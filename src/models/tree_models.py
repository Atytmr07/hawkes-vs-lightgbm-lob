"""Native LightGBM wrappers with sample-budget-specific regularization."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BinaryTreeModel:
    booster: object
    feature_names: list[str]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        return np.asarray(self.booster.predict(matrix), dtype=np.float64)


def pilot_parameters(sample_count: int, seed: int = 20240330) -> dict[str, object]:
    if sample_count < 20_000:
        leaves, min_leaf, fraction, l2 = 7, 200, 0.65, 10.0
    elif sample_count < 100_000:
        leaves, min_leaf, fraction, l2 = 15, 300, 0.75, 5.0
    else:
        leaves, min_leaf, fraction, l2 = 31, 500, 0.85, 2.0
    return {
        "objective": "binary", "metric": "binary_logloss", "verbosity": -1,
        "num_leaves": leaves, "min_data_in_leaf": min_leaf,
        "feature_fraction": fraction, "lambda_l2": l2,
        "learning_rate": 0.05, "seed": seed, "num_threads": 0,
        "deterministic": True, "force_col_wise": True,
    }


def fit_tree(
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    *,
    num_boost_round: int = 160,
    seed: int = 20240330,
    param_overrides: dict[str, object] | None = None,
) -> BinaryTreeModel:
    """Fit with `pilot_parameters`'s coarse 3-bucket heuristic, or with
    `param_overrides` applied on top of it (used by `tune_tree_hyperparameters`
    to evaluate/refit a specific candidate config without duplicating the
    scale_pos_weight/dataset-construction boilerplate)."""
    import lightgbm as lgb

    positives = max(1, int(target.sum()))
    params = pilot_parameters(len(target), seed)
    if param_overrides:
        params.update(param_overrides)
    params["scale_pos_weight"] = (len(target) - positives) / positives
    dataset = lgb.Dataset(matrix, label=target, feature_name=feature_names, free_raw_data=False)
    booster = lgb.train(params, dataset, num_boost_round=num_boost_round)
    return BinaryTreeModel(booster=booster, feature_names=feature_names)


def _candidate_grid(rng: np.random.Generator, n_candidates: int) -> list[dict[str, object]]:
    """Random draws over a sensible LightGBM regularization space, one
    knob at a time so the search covers the space rather than only its
    corners. `pilot_parameters`'s own bucket choice is NOT included here --
    the caller adds it separately as the guaranteed baseline candidate, so
    tuning can never do worse than the untuned heuristic."""
    leaves_choices = [7, 15, 31, 63, 127]
    min_leaf_choices = [30, 50, 100, 200, 300, 500, 800]
    fraction_choices = [0.6, 0.7, 0.8, 0.9, 1.0]
    l2_choices = [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    seen: set[tuple] = set()
    candidates: list[dict[str, object]] = []
    attempts = 0
    while len(candidates) < n_candidates and attempts < n_candidates * 10:
        attempts += 1
        cand = (
            int(rng.choice(leaves_choices)),
            int(rng.choice(min_leaf_choices)),
            float(rng.choice(fraction_choices)),
            float(rng.choice(l2_choices)),
        )
        if cand in seen:
            continue
        seen.add(cand)
        candidates.append({
            "num_leaves": cand[0], "min_data_in_leaf": cand[1],
            "feature_fraction": cand[2], "lambda_l2": cand[3],
        })
    return candidates


def tune_tree_hyperparameters(
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    time_ms: np.ndarray,
    *,
    validation_fraction: float = 0.15,
    n_candidates: int = 16,
    num_boost_round: int = 160,
    seed: int = 20240330,
) -> tuple[dict[str, object], float, float]:
    """Causal (time-ordered, no shuffling) train/validation split within the
    supplied window; random search over `_candidate_grid` plus
    `pilot_parameters`'s own heuristic bucket as a guaranteed baseline
    candidate; returns (best_param_overrides, validation_pr_auc,
    baseline_validation_pr_auc) so a caller can see the tuning lift
    directly, not just the winning config.

    `time_ms` must already be aligned with `matrix`/`target` row-for-row
    (the same convention every other function in this codebase uses) --
    the split point is `time_ms`'s own order, not re-sorted here, matching
    `run_pilot_learning_curve`'s existing causal-ordering assumption.
    """
    from src.utils.metrics import average_precision

    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    n = len(target)
    split = int(round(n * (1.0 - validation_fraction)))
    split = min(max(split, 1), n - 1)
    tr_slice, va_slice = slice(0, split), slice(split, n)
    train_matrix, train_target = matrix[tr_slice], target[tr_slice]
    valid_matrix, valid_target = matrix[va_slice], target[va_slice]

    rng = np.random.default_rng(seed)
    baseline_overrides: dict[str, object] = {}
    candidates = [baseline_overrides] + _candidate_grid(rng, n_candidates - 1)

    best_overrides: dict[str, object] | None = None
    best_score = -np.inf
    baseline_score: float | None = None
    for overrides in candidates:
        model = fit_tree(
            train_matrix, train_target, feature_names,
            num_boost_round=num_boost_round, seed=seed, param_overrides=overrides,
        )
        score = average_precision(valid_target, model.predict(valid_matrix))
        if overrides is baseline_overrides:
            baseline_score = score
        if score > best_score:
            best_score = score
            best_overrides = overrides

    assert best_overrides is not None and baseline_score is not None
    return best_overrides, float(best_score), float(baseline_score)


def fit_tuned_tree(
    matrix: np.ndarray,
    target: np.ndarray,
    feature_names: list[str],
    time_ms: np.ndarray,
    *,
    validation_fraction: float = 0.15,
    n_candidates: int = 16,
    num_boost_round: int = 160,
    seed: int = 20240330,
) -> tuple[BinaryTreeModel, dict[str, object], float, float]:
    """Tune on a causal validation slice of `matrix`/`target`, then refit
    the winning config on the FULL window (so the returned model is not
    trained on 85% of the data it could have used) -- the same
    tune-then-refit pattern used everywhere else in this codebase's
    causal-split conventions. Returns (model, winning_param_overrides,
    validation_pr_auc, baseline_validation_pr_auc)."""
    overrides, val_score, baseline_score = tune_tree_hyperparameters(
        matrix, target, feature_names, time_ms,
        validation_fraction=validation_fraction, n_candidates=n_candidates,
        num_boost_round=num_boost_round, seed=seed,
    )
    model = fit_tree(matrix, target, feature_names, num_boost_round=num_boost_round, seed=seed, param_overrides=overrides)
    return model, overrides, val_score, baseline_score
