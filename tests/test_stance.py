import numpy as np
import pandas as pd
import pytest

from fed_news.features import stance


def test_vintage_rule():
    assert stance.vintage_year(1994) == 1999
    assert stance.vintage_year(2000) == 1999
    assert stance.vintage_year(2016) == 2015
    assert stance.vintage_year(2026) == 2024


def test_classifier_year_maps_early_rows_to_2000():
    assert stance.classifier_year(1996) == 2000
    assert stance.classifier_year(2013) == 2013


def test_model_name():
    assert stance.model_name(2015) == "manelalab/chrono-bert-v1-20151231"


def test_eligible_is_strictly_before_year():
    lab = pd.DataFrame({"sentence": [f"s{i}" for i in range(120)], "label": 0,
                        "year": [1998] * 60 + [2000] * 60})
    assert set(stance.eligible(lab, 2000)["year"]) == {1998}


def test_eligible_raises_when_too_few():
    lab = pd.DataFrame({"sentence": ["a", "b"], "label": [0, 1], "year": [1998, 1999]})
    with pytest.raises(ValueError, match="labels"):
        stance.eligible(lab, 2000)


@pytest.mark.gpu
def test_train_and_score_smoke():
    train = pd.DataFrame({"sentence": ["the committee decided to raise rates"] * 16
                          + ["the committee decided to lower rates"] * 16,
                          "label": [1] * 16 + [0] * 16})
    probs = stance.train_and_score(train, ["inflation pressures warrant tightening"], 1999, 0)
    assert probs.shape == (1, 3)
    assert abs(probs.sum() - 1.0) < 1e-4


SENT = "The committee decided to keep the target range unchanged this quarter."


def _setup(monkeypatch, tmp_path, docs):
    calls = []
    lab = pd.DataFrame({"sentence": [f"label sentence {i}" for i in range(300)],
                        "label": [i % 3 for i in range(300)], "year": 1998,
                        "source": "tdw"})
    corpus = pd.DataFrame(docs, columns=["doc_id", "type", "release", "text"])
    corpus["release"] = pd.to_datetime(corpus["release"])

    def fake_train(train, texts, vintage, seed):
        calls.append((vintage, seed, len(texts)))
        out = np.zeros((len(texts), 3), dtype=np.float32)
        out[:, 1] = 0.7
        out[:, 0] = 0.1
        out[:, 2] = 0.2
        return out

    monkeypatch.setattr(stance.config, "DATA", tmp_path)
    monkeypatch.setattr(stance.config, "RESULTS", tmp_path)
    monkeypatch.setattr(stance, "load_labels", lambda: lab)
    monkeypatch.setattr(stance, "build_corpus", lambda: corpus)
    monkeypatch.setattr(stance, "train_and_score", fake_train)
    return calls


def test_zero_sentence_doc_kept_with_nan(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path, [("a", "statement", "2001-03-01", SENT),
                                   ("b", "speech", "2001-04-01", "too short")])
    out = stance.run_stance()
    b = out[out["doc_id"] == "b"].iloc[0]
    assert b["n_sent"] == 0 and np.isnan(b["stance"])
    a = out[out["doc_id"] == "a"].iloc[0]
    assert a["n_sent"] == 1 and abs(a["stance"] - 0.6) < 1e-6
    assert list(out["doc_id"]) == ["a", "b"]


def test_resume_reuses_matching_part(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, [("a", "statement", "2001-03-01", SENT)])
    stance.run_stance()
    n = len(calls)
    assert n == len(stance.config.STANCE_SEEDS)
    stance.run_stance()
    assert len(calls) == n


def test_resume_recomputes_on_fingerprint_change(monkeypatch, tmp_path):
    docs = [("a", "statement", "2001-03-01", SENT)]
    calls = _setup(monkeypatch, tmp_path, docs)
    stance.run_stance()
    n = len(calls)
    monkeypatch.setattr(stance, "build_corpus", lambda: pd.DataFrame(
        docs + [("c", "minutes", "2001-05-01", SENT)],
        columns=["doc_id", "type", "release", "text"]).assign(
            release=lambda d: pd.to_datetime(d["release"])))
    out = stance.run_stance()
    assert len(calls) == 2 * n
    assert set(out["doc_id"]) == {"a", "c"}
    monkeypatch.setattr(stance.config, "STANCE_LR", 1e-5)
    stance.run_stance()
    assert len(calls) == 3 * n


def test_early_docs_use_year_2000_classifier(monkeypatch, tmp_path):
    calls = _setup(monkeypatch, tmp_path, [("e", "statement", "1996-03-01", SENT)])
    stance.run_stance()
    assert {(v, s) for v, s, _ in calls} == {(1999, s) for s in stance.config.STANCE_SEEDS}
    assert (tmp_path / "stance" / "2000.pkl").exists()
