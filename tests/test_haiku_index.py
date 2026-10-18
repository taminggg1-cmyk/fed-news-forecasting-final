import numpy as np
import pandas as pd

from fed_news.features import haiku_index as hi
from fed_news.features import haiku_prompt as pl


def test_chunks_cover_every_sentence_once_in_random_order():
    parts = hi.chunks(103, batch=25, seed=0)
    flat = np.concatenate(parts)
    assert sorted(flat.tolist()) == list(range(103))
    assert [len(p) for p in parts] == [25, 25, 25, 25, 3]
    assert flat.tolist() != list(range(103))


def test_request_params_use_the_unchanged_attempt_1_labeller():
    p = hi.request_params(["the committee raised rates.", "growth slowed."])
    assert p["model"] == pl.MODEL and p["system"] == pl.SYSTEM
    for k, v in pl.ATTEMPTS[1].items():
        assert p[k] == v
    assert p["messages"] == [{"role": "user", "content": "1. the committee raised rates.\n2. growth slowed."}]


def test_projected_tokens_scales_the_counted_sample():
    assert hi.projected_tokens([1000, 1200], n_requests=10, n_sentences=250) == 1100 * 10 + 8 * 250


def test_assemble_places_answers_and_leaves_missing_as_nan():
    parts = [np.array([2, 0]), np.array([1])]
    codes = hi.assemble(3, parts, {0: [1, 0], 1: None})
    assert codes[2] == 1 and codes[0] == 0 and np.isnan(codes[1])


def test_doc_stance_is_share_hawkish_minus_share_dovish_ignoring_missing():
    docs = pd.DataFrame({"doc_id": ["a", "b", "c"], "type": ["speech", "minutes", "statement"],
                         "release": pd.to_datetime(["2005-01-02", "2005-01-01", "2005-01-03"])})
    sent = pd.DataFrame({"doc_id": ["a", "a", "a", "a", "b", "b"], "sentence": list("uvwxyz")})
    codes = np.array([1, 1, 0, np.nan, 2, 0])  # CODES: dovish 0, hawkish 1, neutral 2
    out = hi.doc_stance(docs, sent, codes)
    assert out["doc_id"].tolist() == ["b", "a", "c"]  # sorted by release
    a = out.set_index("doc_id")
    assert np.isclose(a.loc["a", "stance"], (2 - 1) / 3) and a.loc["a", "n_sent"] == 3
    assert np.isclose(a.loc["b", "stance"], -0.5)
    assert np.isnan(a.loc["c", "stance"]) and a.loc["c", "n_sent"] == 0
