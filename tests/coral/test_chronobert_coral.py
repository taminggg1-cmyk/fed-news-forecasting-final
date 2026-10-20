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
    lp, ordered, biases = cc.train_and_score(train, ["rates should rise", "growth is slowing"], 1999, 0,
                                             tmp_path / "s0")
    assert lp.shape == (2, 2) and ((lp >= 0) & (lp <= 1)).all()
    assert isinstance(ordered, bool) and len(biases) == 2
    assert (tmp_path / "s0" / "encoder").is_dir() and (tmp_path / "s0" / "coral_head.pt").exists()


def test_run_refuses_to_train_without_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="CUDA"):
        cc.run()


def test_optimizer_gives_the_biases_their_own_rate_and_no_decay():
    net, _ = _tiny()
    opt = cc.make_optimizer(net, lr=2e-5)
    bias = [g for g in opt.param_groups if any(p is net.head.bias for p in g["params"])][0]
    rest = [g for g in opt.param_groups if g is not bias][0]
    assert bias["lr"] == 1e-2 and bias["weight_decay"] == 0.0
    assert rest["lr"] == 2e-5 and rest["weight_decay"] == 0.01


def test_neutral_is_not_capped_after_fine_tuning_at_the_real_learning_rate():
    net, tok = _tiny()
    sents = ["raise rates", "cut rates", "the meeting", "the rates"] * 15
    ranks = np.array([2, 0, 1, 1] * 15)
    cc.fine_tune(net, tok, sents, ranks, seed=0, device="cpu")  # Study 4 settings: lr 2e-5, 3 epochs
    p_neutral = core.class_probs(cc.score(net, tok, sents, "cpu"))[:, 1]
    assert p_neutral.max() > 0.3  # the start values (1.0, 0.5) would cap it at 0.124



def test_biases_actually_move_at_their_own_learning_rate():
    net, tok = _tiny()
    sents = ["raise rates", "cut rates", "the meeting", "the rates"] * 15
    ranks = np.array([2, 0, 1, 1] * 15)
    core.init_biases(net.head, ranks)
    start = net.head.bias.detach().clone()
    cc.fine_tune(net, tok, sents, ranks, seed=0, device="cpu")  # 3 epochs, 12 steps
    assert (net.head.bias.detach() - start).abs().max() > 0.02  # at lr 2e-5 they could move at most 0.00024


def test_year_report_shows_mean_class_probabilities_and_biases():
    lp = np.array([[0.9, 0.2], [0.4, 0.1], [0.8, 0.7]])
    truth = np.array([1, 0, 2])
    runs = [(lp, True, [1.2, -0.8]), (lp, False, [0.3, 0.5])]
    rep = cc.year_report(lp[:2], truth[:2], lp, runs)
    p = core.class_probs(lp)
    assert np.isclose(rep["mean_p_neutral_all"], p[:, 1].mean())
    assert np.isclose(rep["mean_p_dovish_holdout"], p[:2, 0].mean())
    assert rep["biases_by_seed"] == [[1.2, -0.8], [0.3, 0.5]] and rep["n_seeds_biases_unordered"] == 1
    assert {"accuracy", "macro_f1", "ordinal_mae"} <= set(rep)
