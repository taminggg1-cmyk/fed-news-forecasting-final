"""BUILD: Fed tone from Claude Haiku 5.5 class probabilities (paid API, about 27 million tokens, about $2-4 at
batch prices).

Every sentence ChronoBERT scored is masked and sent, 25 per request in the Study 4 random order, through the Message
Batches API. Haiku gives three probabilities per sentence (dovish, neutral, hawkish); tone = P(hawkish) - P(dovish).
Reads:   <data>/corpus.pkl, <data>/study4/sentence_labels.pkl and <data>/doc_stance_haiku.pkl (descriptive checks),
         an Anthropic API key (<data>/secrets/anthropic_api_key.txt or ANTHROPIC_API_KEY).
Writes:  <data>/coral/haiku_state.json (batch ids), <data>/coral/haiku_sentence_probs.pkl (every sentence),
         <data>/coral/doc_stance_haiku_ordinal.pkl (tone per document),
         results/coral/haiku_token_projection.json, results/coral/haiku_index.json.
Run:     python -m fed_news.coral.haiku_ordinal project   (count tokens first)
         python -m fed_news.coral.haiku_ordinal submit
         python -m fed_news.coral.haiku_ordinal collect

Details: CORAL study, section 5 as amended by section 13. Haiku cannot be trained, so CORAL's output form is used:
the class probabilities give the levels P(rank >= 1) = N + H and P(rank >= 2) = H, which are always ordered. Same
sentences, masking, order (seed 0) and request size as Study 4; thinking disabled, effort low, max_tokens 2048.
Replies whose three numbers do not add up to 100 are divided by their sum (counted); a missing sentence, a value
outside [0, 100], a wrong length or a zero sum fails the request, which is resent once.
"""
import argparse
import json
import re

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.features import haiku_prompt as pl
from fed_news.features.haiku_tone import PER_BATCH, chunks, sentences

WORK = config.DATA / "coral"
RESULTS = config.RESULTS / "coral"
OUT_PATH = WORK / "doc_stance_haiku_ordinal.pkl"
STATE = WORK / "haiku_state.json"
OLD_LABELS = config.DATA / "study4" / "sentence_labels.pkl"   # Study 4 labels, for the descriptive checks
OLD_DOC = config.DATA / "doc_stance_haiku.pkl"
MAX_UNLABELLED = 0.01  # share of sentences without probabilities above which collect warns
TOKEN_LIMIT, OUT_PER_SENTENCE, N_COUNT = 35_000_000, 15, 200
SYSTEM = pl.SYSTEM[:pl.SYSTEM.index("Judge only")] + (
    "For each sentence give three whole numbers from 0 to 100 that add up to 100: the probabilities that the "
    "sentence is dovish (D), neutral (N) and hawkish (H). Judge only the text of each sentence. Reply with one JSON "
    "object mapping each sentence number to [D, N, H], and nothing else.")
SETTINGS = {"thinking": {"type": "disabled"}, "output_config": {"effort": "low"}, "max_tokens": 2048}
# Columns of the probability arrays: dovish, neutral, hawkish. Their stored codes (download/stance_labels.LABELS).
COL_TO_CODE = np.array([0, 2, 1])


def request_params(sents: list[str]) -> dict:
    """API parameters for one request (model, system prompt, settings, numbered sentences)."""
    return {"model": pl.MODEL, "system": SYSTEM, **SETTINGS,
            "messages": [{"role": "user", "content": pl.batch_prompt(sents)}]}


def _number(x, i: int) -> float:
    """A JSON number or numeric string as a float; anything else (null, true/false, a list, text) is invalid."""
    if isinstance(x, bool) or not isinstance(x, (int, float, str)):
        raise ValueError(f"sentence {i}: {x!r} is not a number")
    try:
        return float(x)
    except ValueError:
        raise ValueError(f"sentence {i}: {x!r} is not a number") from None


def parse_probs(text: str, n: int) -> tuple[np.ndarray, int]:
    """Read a reply into (n, 3) probabilities (dovish, neutral, hawkish) and the number of rows that were divided by
    their sum; raise ValueError if any sentence is missing or invalid."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise ValueError("no JSON object in the reply")
    obj = {str(k).strip(): v for k, v in json.loads(m.group(0)).items()}
    out, n_norm = np.zeros((n, 3)), 0
    for i in range(1, n + 1):
        v = obj.get(str(i))
        if not isinstance(v, (list, tuple)) or len(v) != 3:
            raise ValueError(f"sentence {i}: {v!r} is not three numbers")
        row = np.array([_number(x, i) for x in v])
        if not np.isfinite(row).all() or (row < 0).any() or (row > 100).any():
            raise ValueError(f"sentence {i}: {v!r} outside [0, 100]")
        total = row.sum()
        if total <= 0:
            raise ValueError(f"sentence {i}: zero sum")
        if abs(total - 100) > 1e-9:
            n_norm += 1
        out[i - 1] = row / total
    return out, n_norm


def tone(p: np.ndarray) -> np.ndarray:
    """P(hawkish) - P(dovish) per sentence."""
    p = np.asarray(p, dtype=float)
    return p[:, 2] - p[:, 0]


def levels_from_probs(p: np.ndarray) -> np.ndarray:
    """CORAL levels P(rank >= 1) = N + H and P(rank >= 2) = H."""
    p = np.asarray(p, dtype=float)
    return np.column_stack([p[:, 1] + p[:, 2], p[:, 2]])


def projected_tokens(sample_inputs, n_requests: int, n_sentences: int) -> int:
    """Estimated total tokens from a sample of requests."""
    return int(round(np.mean(sample_inputs) * n_requests + OUT_PER_SENTENCE * n_sentences))


def agreement(probs: np.ndarray, codes: np.ndarray) -> float:
    """Share of sentences whose most probable class equals the Study 4 label (ties count as a mismatch)."""
    probs, codes = np.asarray(probs, dtype=float), np.asarray(codes)
    ok = ~np.isnan(probs).any(1) & ~pd.isna(codes)
    p, c = probs[ok], codes[ok].astype(int)
    top = p.max(1, keepdims=True)
    unique = (p == top).sum(1) == 1
    return float(np.mean(unique & (COL_TO_CODE[p.argmax(1)] == c)))


def assemble(n: int, parts: list[np.ndarray], answers: dict) -> np.ndarray:
    """(n, 3) probabilities from the answered requests (NaN where unanswered)."""
    probs = np.full((n, 3), np.nan)
    for k, idx in enumerate(parts):
        if answers.get(k) is not None:
            probs[idx] = answers[k]
    return probs


def doc_stance(docs: pd.DataFrame, sent: pd.DataFrame, probs: np.ndarray) -> pd.DataFrame:
    """Mean sentence tone per document and its count of labelled sentences, in the doc_stance.pkl format."""
    s = sent.assign(stance=tone(probs)).dropna(subset=["stance"])
    agg = (s.groupby("doc_id")["stance"].agg(["mean", "size"])
           .rename(columns={"mean": "stance", "size": "n_sent"}).reset_index())
    out = docs[["doc_id", "type", "release"]].merge(agg, on="doc_id", how="left")
    out["n_sent"] = out["n_sent"].fillna(0).astype(int)
    return out.sort_values(["release", "doc_id"]).reset_index(drop=True)


def _state() -> dict:
    """Saved batch ids."""
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def _save_state(state: dict) -> None:
    """Save batch ids."""
    WORK.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=2))


def project(client) -> dict:
    """Count the input tokens of 200 random requests and project the total (step 'project')."""
    _, _, masked = sentences()
    parts = chunks(len(masked))
    pick = np.random.default_rng(1).choice(len(parts), N_COUNT, replace=False)
    counts = [client.messages.count_tokens(**{k: v for k, v in request_params([masked[i] for i in parts[k]]).items()
                                              if k in ("model", "system", "messages")}).input_tokens for k in pick]
    res = {"n_sentences": len(masked), "n_requests": len(parts), "sample_requests": N_COUNT,
           "mean_input_per_request": float(np.mean(counts)), "output_per_sentence": OUT_PER_SENTENCE,
           "projected_total_tokens": projected_tokens(counts, len(parts), len(masked)), "limit": TOKEN_LIMIT}
    res["within_limit"] = res["projected_total_tokens"] <= TOKEN_LIMIT
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "haiku_token_projection.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


def _submit(client, masked, parts, ks, state: dict, key: str) -> None:
    """Send requests in batches of PER_BATCH, saving each batch id at once so a crash never resends a batch. The
    SDK's automatic retries are off for these calls: a retried POST whose first attempt succeeded would create a
    second, paid batch."""
    ks = list(ks)
    sender = client.with_options(max_retries=0)
    ids = state.setdefault(key, [])
    for s in range(len(ids) * PER_BATCH, len(ks), PER_BATCH):
        reqs = [{"custom_id": f"c{k}", "params": request_params([masked[i] for i in parts[k]])}
                for k in ks[s:s + PER_BATCH]]
        ids.append(sender.messages.batches.create(requests=reqs).id)
        _save_state(state)
        print(f"{key}: batch {ids[-1]} with {len(reqs)} requests", flush=True)


def submit(client) -> None:
    """Send every request as message batches (step 'submit'), only if the projection is within the limit."""
    path = RESULTS / "haiku_token_projection.json"
    if not path.exists():
        raise SystemExit("run the 'project' step first")
    proj = json.loads(path.read_text())
    if proj["projected_total_tokens"] > TOKEN_LIMIT:
        raise SystemExit(f"projected {proj['projected_total_tokens']} tokens exceeds {TOKEN_LIMIT}; ask the user")
    _, _, masked = sentences()
    parts = chunks(len(masked))
    state = _state()
    _submit(client, masked, parts, range(len(parts)), state, "first")
    print("submitted", state["first"], flush=True)


def _results(client, batch_ids: list[str], answers: dict, usage: dict, parts, counts: dict) -> list[int]:
    """Read finished batch results into `answers`; return the requests that failed."""
    failed = []
    for r in (r for b in batch_ids for r in client.messages.batches.results(b)):
        k = int(r.custom_id[1:])
        if r.result.type != "succeeded":
            failed.append(k)
            continue
        msg = r.result.message
        usage["input"] += msg.usage.input_tokens
        usage["output"] += msg.usage.output_tokens
        try:
            if msg.stop_reason in ("refusal", "max_tokens"):
                raise ValueError(msg.stop_reason)
            probs, n_norm = parse_probs("".join(b.text for b in msg.content if b.type == "text"), len(parts[k]))
        except (ValueError, TypeError, json.JSONDecodeError):
            failed.append(k)
            continue
        answers[k] = probs
        counts[k] = n_norm
    return failed


def collect(client) -> None:
    """Fetch results, resend failures once, then write the sentence probabilities, the tone per document and the
    summary (step 'collect')."""
    state = _state()
    docs, sent, masked = sentences()
    parts = chunks(len(masked))
    for b in state.get("first", []) + state.get("retry", []):
        if client.messages.batches.retrieve(b).processing_status != "ended":
            raise SystemExit(f"batch {b} still processing")
    answers, usage, counts = {}, {"input": 0, "output": 0}, {}
    failed = _results(client, state["first"], answers, usage, parts, counts)
    if failed and "retry" not in state:
        state["n_failed_first"] = len(failed)
        _submit(client, masked, parts, failed, state, "retry")
        raise SystemExit(f"{len(failed)} requests failed; resent, run collect again when done")
    if "retry" in state:
        _results(client, state["retry"], answers, usage, parts, counts)
    probs = assemble(len(masked), parts, answers)
    sent.assign(masked=masked, p_dovish=probs[:, 0], p_neutral=probs[:, 1], p_hawkish=probs[:, 2]).to_pickle(
        WORK / "haiku_sentence_probs.pkl")
    out = doc_stance(docs, sent, probs)
    out.to_pickle(OUT_PATH)
    old_codes = pd.read_pickle(OLD_LABELS)["code"].to_numpy()
    old_doc = pd.read_pickle(OLD_DOC).set_index("doc_id")["stance"]
    both = out.set_index("doc_id")["stance"].to_frame("new").join(old_doc.rename("old")).dropna()
    n_missing = int(np.isnan(probs[:, 0]).sum())
    res = {"n_sentences": len(probs), "n_unlabelled": n_missing, "share_unlabelled": n_missing / len(probs),
           "n_rows_normalised": int(sum(counts.values())), "n_failed_first": state.get("n_failed_first", 0),
           "tokens": usage, "n_docs_without_labels": int((out["n_sent"] == 0).sum()),
           "mean_p": dict(zip(["dovish", "neutral", "hawkish"], np.nanmean(probs, 0).round(4).tolist())),
           "argmax_agreement_with_study4_labels": agreement(probs, old_codes),
           "doc_tone_correlation_with_study4": float(both["new"].corr(both["old"]))}
    if res["share_unlabelled"] > MAX_UNLABELLED:
        res["warning"] = (f"{n_missing} of {len(probs)} sentences ({res['share_unlabelled']:.1%}) have no "
                          f"probabilities after the retry; their documents' tone uses the remaining sentences")
        print("WARNING:", res["warning"], flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "haiku_index.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


def main(argv=None) -> None:
    """Command line: project, submit or collect."""
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["project", "submit", "collect"])
    step = ap.parse_args(argv).step
    client = pl._client()
    {"project": project, "submit": submit, "collect": collect}[step](client)


if __name__ == "__main__":
    main()
