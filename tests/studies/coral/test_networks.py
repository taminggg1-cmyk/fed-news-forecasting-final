import numpy as np
import torch

from fed_news.studies.coral import core, networks


def _ordinal_data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6, 2)).astype(np.float32)
    ranks = np.digitize(X[:, -1, 0], [-0.5, 0.5])
    return X, ranks


def test_coral_models_output_two_logits_and_are_seed_deterministic():
    x = torch.zeros((5, 12, 3))
    for arch in ("lstm", "gru", "tft"):
        a = networks.build_coral_model(arch, 3, 16, 0.0, seed=7)
        b = networks.build_coral_model(arch, 3, 16, 0.0, seed=7)
        a.eval(); b.eval()
        assert a(x).shape == (5, 2)
        assert torch.equal(a(x), b(x))
        assert isinstance(a.head, core.CoralHead)


def test_coral_lstm_learns_an_ordinal_target():
    X, ranks = _ordinal_data()
    model = networks.build_coral_model("lstm", 2, 16, 0.0, seed=0)
    networks.fit_fixed_coral(model, X, ranks, lr=1e-2, epochs=150, device="cpu")
    pred = core.predict_rank(networks.predict_levels(model, X, "cpu"))
    majority = np.bincount(ranks).max() / len(ranks)
    assert (pred == ranks).mean() > majority + 0.2
    assert core.biases_ordered(model.head)


def test_val_log_loss_uses_the_true_class_probability():
    lp = np.array([[0.9, 0.2], [0.5, 0.1]])
    p = core.class_probs(lp)
    assert np.isclose(networks.val_log_loss(lp, np.array([1, 0])), -np.mean(np.log([p[0, 1], p[1, 0]])))


def test_fit_validate_coral_reports_best_epoch_and_log_loss():
    X, ranks = _ordinal_data(200, seed=1)
    Xv, rv = _ordinal_data(80, seed=2)
    model = networks.build_coral_model("gru", 2, 8, 0.0, seed=3)
    best, epoch = networks.fit_validate_coral(model, X, ranks, Xv, rv, lr=1e-2, weight_decay=1e-4, batch_size=32,
                                              max_epochs=40, patience=5, device="cpu", seed=3)
    assert 1 <= epoch <= 40
    refit = networks.build_coral_model("gru", 2, 8, 0.0, seed=3)
    networks.fit_fixed_coral(refit, X, ranks, lr=1e-2, epochs=epoch, device="cpu", batch_size=32, seed=3,
                             weight_decay=1e-4)
    assert abs(networks.val_log_loss(networks.predict_levels(refit, Xv, "cpu"), rv) - best) < 1e-6


def test_fits_start_the_biases_at_the_training_shares():
    X, ranks = _ordinal_data(200)
    model = networks.build_coral_model("lstm", 2, 8, 0.0, seed=0)
    networks.fit_fixed_coral(model, X, ranks, lr=1e-3, epochs=0, device="cpu")
    shares = np.array([(ranks >= 1).mean(), (ranks >= 2).mean()])
    assert np.allclose(model.head.bias.detach().numpy(), np.log(shares / (1 - shares)), atol=1e-5)


def test_middle_class_is_not_capped_at_small_learning_rates():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4, 2)).astype(np.float32)
    ranks = np.where(X[:, -1, 0] > 1.0, 2, np.where(X[:, -1, 0] < -1.0, 0, 1))  # about 70% middle
    model = networks.build_coral_model("gru", 2, 8, 0.0, seed=0)
    networks.fit_fixed_coral(model, X, ranks, lr=1e-4, epochs=3, device="cpu")
    p_mid = core.class_probs(networks.predict_levels(model, X, "cpu"))[:, 1]
    assert p_mid.max() > 0.5  # the start values (1.0, 0.5) would cap it at 0.124
