import json

import numpy as np
import pandas as pd
import pytest

from fed_news.coral import haiku_ordinal as ho
from fed_news.features import haiku_prompt


def test_system_prompt_keeps_the_study4_definitions_and_asks_for_three_numbers():
    definitions = haiku_prompt.SYSTEM[:haiku_prompt.SYSTEM.index("Judge only")]
    assert ho.SYSTEM.startswith(definitions)
    assert "three whole numbers from 0 to 100 that add up to 100" in ho.SYSTEM
    assert "[D, N, H]" in ho.SYSTEM and "H, D or N" not in ho.SYSTEM


def test_request_settings_follow_the_design():
    p = ho.request_params(["a", "b"])
    assert p["model"] == "claude-haiku-5-5" and p["system"] == ho.SYSTEM
    assert p["thinking"] == {"type": "disabled"} and p["output_config"] == {"effort": "low"}
    assert p["max_tokens"] == 2048 and p["messages"][0]["content"] == "1. a\n2. b"


def test_parse_accepts_strings_and_normalises_sums():
    probs, n_norm = ho.parse_probs('```json\n{"1": [10, 60, 30], "2": ["20", "30", "49"]}\n```', 2)
    assert np.allclose(probs[0], [0.1, 0.6, 0.3])
    assert np.allclose(probs.sum(1), 1.0) and n_norm == 1


@pytest.mark.parametrize("reply", ['{"1": [10, 60, 30]}', '{"1": [10, 60, 30], "2": [-5, 60, 45]}',
                                   '{"1": [10, 60, 30], "2": [0, 0, 0]}', '{"1": [10, 60, 30], "2": [10, 90]}',
                                   '{"1": [10, 60, 30], "2": [10, 160, 30]}', 'no json here',
                                   '{"1": [10, 60, 30], "2": [null, 60, 40]}', '{"1": [10, 60, 30], "2": [[1], 60, 39]}',
                                   '{"1": [10, 60, 30], "2": [true, 60, 39]}', '{"1": [10, 60, 30], "2": ["x", 60, 40]}'])
def test_parse_rejects_missing_negative_zero_sum_wrong_length_and_above_100(reply):
    with pytest.raises(ValueError):
        ho.parse_probs(reply, 2)


def test_tone_and_levels():
    p = np.array([[0.1, 0.6, 0.3]])
    assert np.isclose(ho.tone(p)[0], 0.2)
    assert np.allclose(ho.levels_from_probs(p), [[0.9, 0.3]])


def test_projection_rule():
    total = ho.projected_tokens([1600, 1600], 13_750, 343_742)
    assert total == 1600 * 13_750 + 15 * 343_742
    assert total <= ho.TOKEN_LIMIT == 35_000_000


def test_agreement_with_study4_labels_counts_ties_as_mismatch():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.1, 0.8, 0.1], [0.4, 0.2, 0.4]])
    codes = np.array([0, 1, 2, 0])  # stored codes: dovish 0, hawkish 1, neutral 2
    assert ho.agreement(probs, codes) == 0.75


def test_doc_stance_averages_sentence_tone_and_skips_unlabelled():
    docs = pd.DataFrame({"doc_id": ["a", "b"], "type": ["statement"] * 2,
                         "release": pd.to_datetime(["2001-01-01", "2001-02-01"])})
    sent = pd.DataFrame({"doc_id": ["a", "a", "b"], "sentence": ["x", "y", "z"]})
    probs = np.array([[0.1, 0.6, 0.3], [0.5, 0.5, 0.0], [np.nan] * 3])
    out = ho.doc_stance(docs, sent, probs).set_index("doc_id")
    assert np.isclose(out.loc["a", "stance"], (0.2 - 0.5) / 2) and out.loc["a", "n_sent"] == 2
    assert np.isnan(out.loc["b", "stance"]) and out.loc["b", "n_sent"] == 0


class _Batches:
    def __init__(self, log, results=None):
        self.log, self.results_by_id, self.n = log, results or {}, 0

    def create(self, requests):
        self.log.append({"n_requests": len(requests), "max_retries": self.max_retries})
        return type("B", (), {"id": f"new{len(self.log)}"})()

    def retrieve(self, batch_id):
        return type("S", (), {"processing_status": "ended"})()

    def results(self, batch_id):
        return self.results_by_id[batch_id]


class _Client:
    def __init__(self, log, results=None, max_retries=2):
        self.log, self.max_retries = log, max_retries
        self.messages = type("M", (), {})()
        self.messages.batches = _Batches(log, results)
        self.messages.batches.max_retries = max_retries

    def with_options(self, max_retries):
        clone = _Client(self.log, self.messages.batches.results_by_id, max_retries)
        clone.messages.batches = self.messages.batches
        clone.messages.batches.max_retries = max_retries
        return clone


def _isolate(monkeypatch, tmp_path, n_sent=6):
    monkeypatch.setattr(ho, "WORK", tmp_path)
    monkeypatch.setattr(ho, "STATE", tmp_path / "state.json")
    monkeypatch.setattr(ho, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(ho, "OUT_PATH", tmp_path / "doc.pkl")
    docs = pd.DataFrame({"doc_id": ["a", "b"], "type": ["statement"] * 2,
                         "release": pd.to_datetime(["2001-01-01", "2001-02-01"])})
    sent = pd.DataFrame({"doc_id": ["a"] * 3 + ["b"] * (n_sent - 3), "sentence": [f"s{i}" for i in range(n_sent)]})
    monkeypatch.setattr(ho, "sentences", lambda: (docs, sent, [f"m{i}" for i in range(n_sent)]))
    monkeypatch.setattr(ho, "chunks", lambda n: [np.arange(0, 3), np.arange(3, n)])
    monkeypatch.setattr(ho, "PER_BATCH", 1)
    pd.DataFrame({"code": [0, 1, 2, 2, 1, 0]}).to_pickle(tmp_path / "old_labels.pkl")
    pd.DataFrame({"doc_id": ["a", "b"], "stance": [0.1, -0.2]}).to_pickle(tmp_path / "old_doc.pkl")
    monkeypatch.setattr(ho, "OLD_LABELS", tmp_path / "old_labels.pkl")
    monkeypatch.setattr(ho, "OLD_DOC", tmp_path / "old_doc.pkl")
    return docs, sent


def _write_projection(tmp_path, total):
    (tmp_path / "results").mkdir(exist_ok=True)
    (tmp_path / "results" / "haiku_token_projection.json").write_text(json.dumps({"projected_total_tokens": total}))


def test_submit_refuses_without_a_projection_or_above_the_limit(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    log = []
    with pytest.raises(SystemExit, match="project"):
        ho.submit(_Client(log))
    _write_projection(tmp_path, 35_000_001)
    with pytest.raises(SystemExit, match="exceeds"):
        ho.submit(_Client(log))
    assert log == []


def test_submit_sends_each_batch_once_without_sdk_retries_and_resumes(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    _write_projection(tmp_path, 1000)
    log = []
    ho.submit(_Client(log))
    assert len(log) == 2 and all(e["max_retries"] == 0 for e in log)
    ho.submit(_Client(log))  # state already holds both batches: nothing is sent again
    assert len(log) == 2


def _reply(rows):
    text = json.dumps({str(i + 1): r for i, r in enumerate(rows)})
    msg = type("Msg", (), {"stop_reason": "end_turn",
                           "usage": type("U", (), {"input_tokens": 10, "output_tokens": 5})(),
                           "content": [type("T", (), {"type": "text", "text": text})()]})()
    return msg


def _result(custom_id, msg=None):
    res = type("R", (), {"type": "succeeded" if msg else "errored", "message": msg})()
    return type("Row", (), {"custom_id": custom_id, "result": res})()


def test_collect_resends_failures_once_then_writes_tone_and_flags_unlabelled(monkeypatch, tmp_path):
    _isolate(monkeypatch, tmp_path)
    good = _reply([[10, 60, 30], [50, 50, 0], [0, 20, 80]])
    state = {"first": ["b1", "b2"]}
    ho._save_state(state)
    results = {"b1": [_result("c0", good)], "b2": [_result("c1")], "new1": [_result("c1")]}
    log = []
    with pytest.raises(SystemExit, match="resent"):
        ho.collect(_Client(log, results))
    assert len(log) == 1 and log[0]["max_retries"] == 0 and json.loads(ho.STATE.read_text())["retry"] == ["new1"]
    ho.collect(_Client(log, results))  # the retry failed again: sentences 3-5 stay unlabelled
    out = pd.read_pickle(ho.OUT_PATH).set_index("doc_id")
    assert np.isclose(out.loc["a", "stance"], np.mean([0.2, -0.5, 0.8])) and out.loc["b", "n_sent"] == 0
    res = json.loads((tmp_path / "results" / "haiku_index.json").read_text())
    assert res["n_unlabelled"] == 3 and res["share_unlabelled"] == 0.5 and "warning" in res
