import numpy as np

from src.models.tree_models import fit_tuned_tree, tune_tree_hyperparameters


def _synthetic_binary_problem(seed: int, n: int = 4000, n_features: int = 6):
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(n, n_features)).astype(np.float64)
    weights = rng.normal(size=n_features)
    logits = matrix @ weights
    prob = 1.0 / (1.0 + np.exp(-logits))
    target = (rng.uniform(size=n) < prob).astype(np.uint8)
    time_ms = np.arange(n, dtype=np.int64) * 1000
    names = [f"f{i}" for i in range(n_features)]
    return matrix, target, names, time_ms


def test_tuning_never_scores_below_the_untuned_baseline():
    """pilot_parameters's own heuristic bucket is always included as a
    candidate, so the winning validation score must be >= the baseline's."""
    matrix, target, names, time_ms = _synthetic_binary_problem(seed=3)
    _, best_score, baseline_score = tune_tree_hyperparameters(
        matrix, target, names, time_ms, n_candidates=8, num_boost_round=40,
    )
    assert best_score >= baseline_score - 1e-9


def test_fit_tuned_tree_returns_a_working_model_on_the_full_window():
    matrix, target, names, time_ms = _synthetic_binary_problem(seed=5)
    model, overrides, val_score, baseline_score = fit_tuned_tree(
        matrix, target, names, time_ms, n_candidates=8, num_boost_round=40,
    )
    pred = model.predict(matrix)
    assert pred.shape == (len(target),)
    assert np.all((pred >= 0) & (pred <= 1))
    assert set(overrides.keys()) <= {"num_leaves", "min_data_in_leaf", "feature_fraction", "lambda_l2"}
    assert np.isfinite(val_score) and np.isfinite(baseline_score)


def test_tune_rejects_invalid_validation_fraction():
    matrix, target, names, time_ms = _synthetic_binary_problem(seed=7, n=200)
    try:
        tune_tree_hyperparameters(matrix, target, names, time_ms, validation_fraction=0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass
