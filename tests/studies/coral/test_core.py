import numpy as np
import torch
from coral_pytorch.dataset import levels_from_labelbatch, proba_to_label
from coral_pytorch.layers import CoralLayer
from coral_pytorch.losses import coral_loss as reference_loss
from scipy.special import expit

from fed_news.studies.coral import core
from fed_news.models.linear import OrderedLogit


def test_loss_levels_and_labels_match_coral_pytorch():
    gen = torch.Generator().manual_seed(0)
    logits = torch.randn(50, 2, generator=gen)
    ranks = torch.randint(0, 3, (50,), generator=gen)
    lv = core.levels(ranks)
    assert torch.equal(lv, levels_from_labelbatch(ranks.tolist(), 3))
    assert torch.allclose(core.coral_loss(logits, lv), reference_loss(logits, lv), atol=1e-6)
    lp = core.level_probs(logits)
    assert np.array_equal(core.predict_rank(lp), proba_to_label(torch.sigmoid(logits)).numpy())


def test_head_matches_coral_layer_and_initial_biases():
    torch.manual_seed(0)
    head = core.CoralHead(4)
    assert head.bias.detach().tolist() == [1.0, 0.5]
    ref = CoralLayer(4, 3)
    with torch.no_grad():
        ref.coral_weights.weight.copy_(head.weight.weight)
        ref.coral_bias.copy_(head.bias)
    x = torch.randn(7, 4)
    assert torch.allclose(head(x), ref(x))


def test_levels_encoding():
    assert core.levels([0, 1, 2]).tolist() == [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]


def test_class_probs_nonnegative_sum_to_one_even_when_levels_cross():
    p = core.class_probs(np.array([[0.3, 0.6], [0.9, 0.2]]))
    assert (p >= 0).all() and np.allclose(p.sum(1), 1.0)
    assert np.allclose(p[1], [0.1, 0.7, 0.2])


def test_tone_rank_map():
    # stored codes: dovish 0, hawkish 1, neutral 2 -> ranks dovish 0, neutral 1, hawkish 2
    assert core.TONE_RANK == {0: 0, 2: 1, 1: 2}
    assert np.isclose(core.tone_from_levels(np.array([[0.9, 0.2]]))[0], 0.2 - 0.1)


def test_biases_ordered_detects_crossing():
    head = core.CoralHead(3)
    assert core.biases_ordered(head)
    with torch.no_grad():
        head.bias.copy_(torch.tensor([0.0, 0.4]))
    assert not core.biases_ordered(head)


def test_linear_coral_equals_ordered_logit():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(2000, 2))
    eta = X @ np.array([1.0, -0.5])
    c1, c2 = expit(-0.5 - eta), expit(1.0 - eta)
    u = rng.random(2000)
    y = np.where(u < c1, 0, np.where(u < c2, 1, 2))
    torch.manual_seed(0)
    head = core.CoralHead(2)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    xt, lv = torch.tensor(X, dtype=torch.float32), core.levels(y)
    for _ in range(2000):
        opt.zero_grad()
        core.coral_loss(head(xt), lv).backward()
        opt.step()
    p_coral = core.class_probs(core.level_probs(head(xt).detach()))
    p_logit = OrderedLogit(0.0).fit(X, y).predict_proba(X)
    assert np.abs(p_coral - p_logit).max() < 0.02
    assert core.biases_ordered(head)


def test_init_biases_sets_logits_of_cumulative_training_shares():
    head = core.CoralHead(4)
    core.init_biases(head, [0] * 30 + [1] * 45 + [2] * 25)
    assert np.allclose(head.bias.detach().numpy(), np.log(np.array([0.70, 0.25]) / np.array([0.30, 0.75])), atol=1e-6)
    core.init_biases(head, [1] * 10)  # no rank 2 at all: share 0 clipped to 0.001
    assert np.isclose(head.bias.detach().numpy()[1], np.log(0.001 / 0.999), atol=1e-5)


def test_param_groups_keep_coral_biases_free_of_weight_decay():
    model = torch.nn.Sequential(torch.nn.Linear(3, 4))
    model.head = core.CoralHead(4)
    groups = core.param_groups(model, lr=1e-3, weight_decay=0.1, bias_lr=1e-2)
    bias_group = [g for g in groups if any(p is model.head.bias for p in g["params"])]
    assert len(bias_group) == 1 and bias_group[0]["weight_decay"] == 0.0 and bias_group[0]["lr"] == 1e-2
    other = [g for g in groups if g is not bias_group[0]][0]
    assert other["weight_decay"] == 0.1 and other["lr"] == 1e-3
    assert sum(len(g["params"]) for g in groups) == len(list(model.parameters()))
