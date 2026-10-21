import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, BertConfig, BertTokenizerFast

from fed_news.coral import chronobert_coral as cc
from fed_news.coral import core
from fed_news.distill import chronobert_distill as cd

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "rates", "raise", "hike", "cut", "ease", "the", "meeting"]


def _tiny():
    tok = BertTokenizerFast(vocab={w: i for i, w in enumerate(WORDS)})
    torch.manual_seed(0)
    lm = AutoModel.from_config(BertConfig(vocab_size=len(WORDS), hidden_size=16, num_hidden_layers=1,
                                          num_attention_heads=2, intermediate_size=32))
    return cc.StanceNetCoral(lm), tok


def _sents(years, per_year=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for y in years:
        for k in range(per_year):
            q = rng.dirichlet([1, 1, 1])
            rows.append({"doc_id": f"speech:{y}0315:d{y}{k // 3}", "sentence": f"s{y}-{k}",
                         "p_dovish": q[0], "p_neutral": q[1], "p_hawkish": q[2]})
    return cd.with_years(pd.DataFrame(rows))


def test_soft_levels_are_neutral_plus_hawkish_and_hawkish():
    q = np.array([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]])
    assert np.allclose(cd.soft_levels(q), [[0.8, 0.3], [0.4, 0.1]])


def test_soft_bias_start_is_the_logit_of_the_mean_soft_levels():
    head = core.CoralHead(4)
    lv = np.array([[0.8, 0.3], [0.4, 0.1]])
    cd.init_biases_soft(head, lv)
    m = lv.mean(0)
    assert np.allclose(head.bias.detach().numpy(), np.log(m / (1 - m)), atol=1e-6)


def test_soft_cross_entropy_matches_its_formula_and_is_smallest_at_the_targets():
    q = np.array([[0.2, 0.5, 0.3], [0.6, 0.3, 0.1]])
    exact = cd.soft_levels(q)
    entropy = float(np.mean(-(q * np.log(q)).sum(1)))
    assert np.isclose(cd.soft_cross_entropy(q, exact), entropy)
    assert cd.soft_cross_entropy(q, np.array([[0.5, 0.5], [0.5, 0.5]])) > entropy


def test_sentence_years_come_from_the_document_ids():
    df = cd.with_years(pd.DataFrame({"doc_id": ["minutes:19940325:ab", "speech:20091231:cd",
                                                "statement:20100127:ef"]}))
    assert df["year"].tolist() == [1994, 2009, 2010]
    assert df["release"].tolist() == list(pd.to_datetime(["1994-03-25", "2009-12-31", "2010-01-27"]))


def test_tuning_uses_half_of_1994_2006_and_validates_on_2007_2009_only():
    s = _sents(range(1994, 2013), per_year=40)
    s.loc[s.index[5], "p_dovish"] = np.nan  # a sentence without Haiku probabilities is never used
    tr, va = cd.tuning_split(s, n_valid=50)
    assert s.loc[tr, "year"].max() <= 2006 and s.loc[va, "year"].between(2007, 2009).all()
    pool = s[(s["year"] <= 2006) & s["p_dovish"].notna()]
    assert len(tr) == round(cd.TUNE_SHARE * len(pool)) and len(va) == 50
    assert s.loc[np.r_[tr, va], "p_dovish"].notna().all()
    tr2, va2 = cd.tuning_split(s, n_valid=50)
    assert list(tr) == list(tr2) and list(va) == list(va2)


def test_plan_scores_every_sentence_once_with_a_model_that_did_not_train_on_it():
    s = _sents(range(1994, 2013))
    s.loc[s.index[-1], "p_hawkish"] = np.nan  # unlabelled, still scored
    plan = cd.plan(s)
    assert set(plan) == {"final", "foldA", "foldB"}
    scored = np.concatenate([p["score"] for p in plan.values()])
    assert sorted(scored) == list(s.index)
    for p in plan.values():
        assert not set(p["train"]) & set(p["score"])
        assert s.loc[p["train"], "p_hawkish"].notna().all()
    yr = s["year"]
    assert yr[plan["final"]["train"]].max() == 2009 and yr[plan["final"]["score"]].min() == 2010
    assert yr[plan["foldA"]["train"]].between(1994, 2001).all()
    assert yr[plan["foldA"]["score"]].between(2002, 2009).all()
    assert yr[plan["foldB"]["train"]].between(2002, 2009).all()
    assert yr[plan["foldB"]["score"]].between(1994, 2001).all()


def test_best_setting_is_lowest_loss_then_fewer_epochs_then_lower_learning_rate():
    rows = [{"lr": 2e-5, "epochs": 2, "val_ce": 0.50000}, {"lr": 4e-5, "epochs": 1, "val_ce": 0.50005},
            {"lr": 1e-5, "epochs": 2, "val_ce": 0.6}]
    assert cd.pick_best(rows) == {"lr": 4e-5, "epochs": 1}  # within 1e-4: fewer epochs wins
    rows.append({"lr": 1e-5, "epochs": 1, "val_ce": 0.50003})
    assert cd.pick_best(rows) == {"lr": 1e-5, "epochs": 1}  # then the lower learning rate
    assert cd.pick_best([{"lr": 1e-5, "epochs": 1, "val_ce": 0.7},
                         {"lr": 4e-5, "epochs": 2, "val_ce": 0.4}]) == {"lr": 4e-5, "epochs": 2}


def test_soft_fine_tune_moves_the_model_towards_the_haiku_probabilities_and_reports_each_epoch():
    net, tok = _tiny()
    texts = ["raise hike rates", "cut ease rates", "the meeting"] * 8
    q = np.array([[0.05, 0.15, 0.8], [0.8, 0.15, 0.05], [0.1, 0.8, 0.1]] * 8)
    before = cd.soft_cross_entropy(q, cc.score(net, tok, texts, "cpu"))
    seen = []
    cd.fine_tune_soft(net, tok, texts, cd.soft_levels(q), seed=0, device="cpu", lr=5e-3, epochs=3, batch=8,
                      on_epoch=lambda e, n: seen.append(e))
    after = cd.soft_cross_entropy(q, cc.score(net, tok, texts, "cpu"))
    assert seen == [1, 2, 3] and after < before - 0.05


def test_scoring_in_length_order_returns_rows_in_the_input_order():
    net, tok = _tiny()
    texts = ["raise hike rates the meeting", "cut", "the meeting", "ease rates"]
    assert np.allclose(cd.score_sorted(net, tok, texts, "cpu", batch=2), cc.score(net, tok, texts, "cpu"),
                       atol=1e-6)


def test_document_tone_is_the_mean_sentence_tone_and_keeps_documents_without_sentences():
    docs = pd.DataFrame({"doc_id": ["a", "b", "c"], "type": ["speech"] * 3,
                         "release": pd.to_datetime(["2010-01-05", "2010-01-04", "2010-02-01"])})
    sent = pd.DataFrame({"doc_id": ["a", "a", "b"], "stance": [0.2, 0.4, -0.1]})
    out = cd.doc_tone(docs, sent)
    assert out["doc_id"].tolist() == ["b", "a", "c"]  # sorted by release
    assert np.allclose(out["stance"].iloc[:2], [-0.1, 0.3]) and np.isnan(out["stance"].iloc[2])
    assert out["n_sent"].tolist() == [1, 2, 0]


def test_fidelity_reports_correlation_cross_entropy_and_agreement():
    q = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7], [0.2, 0.6, 0.2], [0.6, 0.3, 0.1]])
    f = cd.fidelity(q, cd.soft_levels(q))
    assert np.isclose(f["tone_correlation"], 1.0) and np.isclose(f["agreement"], 1.0)
    assert f["n"] == 4 and np.isclose(f["soft_cross_entropy"], cd.soft_cross_entropy(q, cd.soft_levels(q)))


def test_soft_fine_tune_leaves_no_gradients_on_the_model():
    net, tok = _tiny()
    texts = ["raise hike rates", "cut ease rates"] * 4
    q = np.array([[0.05, 0.15, 0.8], [0.8, 0.15, 0.05]] * 4)
    cd.fine_tune_soft(net, tok, texts, cd.soft_levels(q), seed=0, device="cpu", lr=1e-3, epochs=1, batch=4)
    assert all(p.grad is None for p in net.parameters())


def test_a_diverged_setting_with_nan_loss_never_wins():
    rows = [{"lr": 4e-5, "epochs": 2, "val_ce": float("nan")}, {"lr": 2e-5, "epochs": 2, "val_ce": 0.83},
            {"lr": 1e-5, "epochs": 1, "val_ce": 0.84}]
    assert cd.pick_best(rows) == {"lr": 2e-5, "epochs": 2}
