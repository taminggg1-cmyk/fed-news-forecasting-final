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
                                   '{"1": [10, 60, 30], "2": [10, 160, 30]}', 'no json here'])
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
