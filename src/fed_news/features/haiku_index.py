"""Study 4, section 3: Claude Haiku 5.5 stance labels for every sentence ChronoBERT scored
(docs/superpowers/specs/2026-10-12-study4-haiku-index-tuned-design.md).

Steps (each resumable): --project counts the input tokens of 200 random requests and scales them to all;
--submit sends every request through the Message Batches API if the projection is within the token limit;
--collect fetches results, resends failed requests once, and writes doc_stance_haiku.pkl.
"""
import argparse
import json

import numpy as np
import pandas as pd

from fed_news import config
from fed_news.features import stance
from fed_news.features import haiku_prompt as pl
from fed_news.features.haiku_prompt import mask, official_names

OUT_PATH = config.DATA / "doc_stance_haiku.pkl"
WORK = config.DATA / "study4"
RESULTS = config.RESULTS / "study4"
BATCH, SEED = 25, 0
PER_BATCH = 4000
TOKEN_LIMIT, OUT_PER_SENTENCE, N_COUNT = 25_000_000, 8, 200  # design amendment 7.1: 20M raised by the user
SCORE = {0: -1.0, 1: 1.0, 2: 0.0}  # poc_llm_labels.CODES: dovish 0, hawkish 1, neutral 2


def chunks(n: int, batch: int = BATCH, seed: int = SEED) -> list[np.ndarray]:
    order = np.random.default_rng(seed).permutation(n)
    return [order[s:s + batch] for s in range(0, n, batch)]


def request_params(sentences: list[str]) -> dict:
    return {"model": pl.MODEL, "system": pl.SYSTEM, **pl.ATTEMPTS[1],
            "messages": [{"role": "user", "content": pl.batch_prompt(sentences)}]}


def projected_tokens(sample_inputs, n_requests: int, n_sentences: int) -> int:
    return int(round(np.mean(sample_inputs) * n_requests + OUT_PER_SENTENCE * n_sentences))


def assemble(n: int, parts: list[np.ndarray], answers: dict) -> np.ndarray:
    codes = np.full(n, np.nan)
    for k, idx in enumerate(parts):
        if answers.get(k) is not None:
            codes[idx] = answers[k]
    return codes


def doc_stance(docs: pd.DataFrame, sent: pd.DataFrame, codes: np.ndarray) -> pd.DataFrame:
    s = sent.assign(stance=pd.Series(codes).map(SCORE).to_numpy()).dropna(subset=["stance"])
    agg = (s.groupby("doc_id")["stance"].agg(["mean", "size"])
           .rename(columns={"mean": "stance", "size": "n_sent"}).reset_index())
    out = docs[["doc_id", "type", "release"]].merge(agg, on="doc_id", how="left")
    out["n_sent"] = out["n_sent"].fillna(0).astype(int)
    return out.sort_values(["release", "doc_id"]).reset_index(drop=True)


def sentences() -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    corpus = pd.read_pickle(config.DATA / "corpus.pkl")
    docs = corpus[corpus["type"].isin(stance.STANCE_TYPES)].copy()
    sent = stance._sentence_table(docs)
    names = official_names(corpus)
    return docs, sent, [mask(s, names) for s in sent["sentence"]]


def _state() -> dict:
    p = WORK / "state.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _save_state(state: dict) -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "state.json").write_text(json.dumps(state, indent=2))


def project(client) -> dict:
    _, sent, masked = sentences()
    parts = chunks(len(masked))
    pick = np.random.default_rng(1).choice(len(parts), N_COUNT, replace=False)
    counts = [client.messages.count_tokens(**{k: v for k, v in request_params([masked[i] for i in parts[k]]).items()
                                              if k in ("model", "system", "messages")}).input_tokens for k in pick]
    res = {"n_sentences": len(masked), "n_requests": len(parts), "sample_requests": N_COUNT,
           "mean_input_per_request": float(np.mean(counts)),
           "projected_total_tokens": projected_tokens(counts, len(parts), len(masked)), "limit": TOKEN_LIMIT}
    res["within_limit"] = res["projected_total_tokens"] <= TOKEN_LIMIT
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "token_projection.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)
    return res


def _submit(client, masked, parts, ks, state: dict, key: str) -> None:
    """Send requests in batches of PER_BATCH, saving each batch id at once so a crash never resends a batch."""
    ks = list(ks)
    ids = state.setdefault(key, [])
    for s in range(len(ids) * PER_BATCH, len(ks), PER_BATCH):
        reqs = [{"custom_id": f"c{k}", "params": request_params([masked[i] for i in parts[k]])}
                for k in ks[s:s + PER_BATCH]]
        ids.append(client.messages.batches.create(requests=reqs).id)
        _save_state(state)
        print(f"{key}: batch {ids[-1]} with {len(reqs)} requests", flush=True)


def submit(client) -> None:
    proj = json.loads((RESULTS / "token_projection.json").read_text())
    if proj["projected_total_tokens"] > TOKEN_LIMIT:
        raise SystemExit(f"projected {proj['projected_total_tokens']} tokens exceeds {TOKEN_LIMIT}; ask the user")
    _, _, masked = sentences()
    parts = chunks(len(masked))
    state = _state()
    _submit(client, masked, parts, range(len(parts)), state, "first")
    print("submitted", state["first"], flush=True)


def _results(client, batch_ids: list[str], answers: dict, usage: dict, parts) -> list[int]:
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
            answers[k] = pl.parse_labels("".join(b.text for b in msg.content if b.type == "text"), len(parts[k]))
        except (ValueError, json.JSONDecodeError):
            failed.append(k)
    return failed


def collect(client) -> None:
    state = _state()
    docs, sent, masked = sentences()
    parts = chunks(len(masked))
    for b in state.get("first", []) + state.get("retry", []):
        if client.messages.batches.retrieve(b).processing_status != "ended":
            raise SystemExit(f"batch {b} still processing")
    answers, usage = {}, {"input": 0, "output": 0}
    failed = _results(client, state["first"], answers, usage, parts)
    if failed and "retry" not in state:
        state["n_failed_first"] = len(failed)
        _submit(client, masked, parts, failed, state, "retry")
        raise SystemExit(f"{len(failed)} requests failed; resent, run collect again when done")
    if "retry" in state:
        _results(client, state["retry"], answers, usage, parts)
    codes = assemble(len(masked), parts, answers)
    sent.assign(masked=masked, code=codes).to_pickle(WORK / "sentence_labels.pkl")
    out = doc_stance(docs, sent, codes)
    out.to_pickle(OUT_PATH)
    res = {"n_sentences": len(codes), "n_unlabelled": int(np.isnan(codes).sum()),
           "n_failed_first": state.get("n_failed_first", 0), "tokens": usage,
           "n_docs_without_labels": int((out["n_sent"] == 0).sum()),
           "share_hawkish": float(np.nanmean(codes == 1)), "share_dovish": float(np.nanmean(codes == 0))}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "haiku_index.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2), flush=True)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["project", "submit", "collect"])
    step = ap.parse_args(argv).step
    client = pl._client()
    {"project": project, "submit": submit, "collect": collect}[step](client)


if __name__ == "__main__":
    main()
