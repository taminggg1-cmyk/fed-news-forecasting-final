import numpy as np
import pandas as pd
import pytest
import torch
from transformers import AutoModel, BertConfig, BertTokenizerFast

from fed_news.coral import chronobert_coral as cc
from fed_news.coral import core
from fed_news.features import chronobert_tone

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "rates", "raise", "hike", "cut", "ease", "the", "meeting"]


def _tiny():
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(WORDS)})  # transformers 5: vocab, not vocab_file
    torch.manual_seed(0)
    lm = AutoModel.from_config(BertConfig(vocab_size=len(WORDS), hidden_size=16, num_hidden_layers=1,
                                          num_attention_heads=2, intermediate_size=32))
    return cc.StanceNetCoral(lm), tok


def test_labels_are_remapped_to_tone_ranks():
    # stored codes dovish 0, hawkish 1, neutral 2 -> ranks 0, 2, 1
    assert cc.to_ranks([0, 1, 2]).tolist() == [0, 2, 1]


def test_seed_average_of_levels_stays_ordered_and_gives_the_tone():
    lps = [np.array([[0.9, 0.2], [0.4, 0.1]]), np.array([[0.7, 0.6], [0.2, 0.05]])]
    lp, tone = cc.combine_seeds(lps)
    assert np.allclose(lp, [[0.8, 0.4], [0.3, 0.075]])
    assert (lp[:, 0] >= lp[:, 1]).all()
    assert np.allclose(tone, core.tone_from_levels(lp))


def test_ordinal_mae_counts_rank_distance():
    lp = np.array([[0.9, 0.9], [0.1, 0.1], [0.9, 0.1]])  # predicted ranks 2, 0, 1
    assert np.isclose(cc.ordinal_mae(lp, np.array([0, 0, 2])), (2 + 0 + 1) / 3)


def test_saved_model_reloads_and_gives_the_same_levels(tmp_path):
    net, tok = _tiny()
    texts = ["raise rates", "cut rates", "the meeting"]
    before = cc.score(net, tok, texts, "cpu")
    cc.save_model(net, tok, tmp_path / "m", {"vintage": 1999})
    net2, tok2 = cc.load_model(tmp_path / "m")
    assert np.allclose(before, cc.score(net2, tok2, texts, "cpu"), atol=1e-6)
    assert (tmp_path / "m" / "info.json").exists()


def test_fine_tune_learns_a_tiny_ordinal_task_on_cpu(tmp_path):
    net, tok = _tiny()
    sents = ["raise rates", "hike rates", "cut rates", "ease rates", "the meeting", "the rates"] * 10
    ranks = np.array([2, 2, 0, 0, 1, 1] * 10)
    cc.fine_tune(net, tok, sents, ranks, seed=0, device="cpu", epochs=30, lr=1e-3)
    pred = core.predict_rank(cc.score(net, tok, sents[:6], "cpu"))
    assert pred.tolist() == [2, 2, 0, 0, 1, 1]
    assert core.biases_ordered(net.head)


def test_fingerprint_marks_coral_runs():
    train = pd.DataFrame({"sentence": ["a", "b"], "label": [0, 1]})
    hold = pd.DataFrame({"sentence": ["c"], "label": [2]})
    old = chronobert_tone._fingerprint(train, hold, 2005, ["d1"])
    new = cc.fingerprint(train, hold, 2005, ["d1"])
    assert new["coral"] == 1 and {k: new[k] for k in old} == old


@pytest.mark.gpu
def test_one_vintage_trains_scores_and_saves_on_the_gpu(tmp_path):
    train = pd.DataFrame({"sentence": ["inflation pressures warrant higher rates", "the economy is weak",
                                       "the committee met today"] * 20, "label": [1, 0, 2] * 20})
    lp, ordered = cc.train_and_score(train, ["rates should rise", "growth is slowing"], 1999, 0, tmp_path / "s0")
    assert lp.shape == (2, 2) and ((lp >= 0) & (lp <= 1)).all()
    assert isinstance(ordered, bool)
    assert (tmp_path / "s0" / "encoder").is_dir() and (tmp_path / "s0" / "coral_head.pt").exists()


def test_run_refuses_to_train_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA"):
        cc.run()
