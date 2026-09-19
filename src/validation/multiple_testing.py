"""Day-block Romano-Wolf stepdown implementation (full-window use only)."""

from __future__ import annotations

import numpy as np


def romano_wolf_stepdown(loss_differentials: np.ndarray, day_ids: np.ndarray, *, replications: int = 2000, seed: int = 20240330) -> np.ndarray:
    """Return stepdown adjusted p-values for columns of loss differentials."""
    values = np.asarray(loss_differentials, np.float64)
    days, inverse = np.unique(day_ids, return_inverse=True)
    observed = np.abs(values.mean(axis=0) / np.maximum(values.std(axis=0, ddof=1) / np.sqrt(len(values)), 1e-12))
    centered = values - values.mean(axis=0)
    rng = np.random.default_rng(seed)
    boot = np.empty((replications, values.shape[1]))
    day_rows = [np.flatnonzero(inverse == i) for i in range(len(days))]
    for b in range(replications):
        chosen = rng.integers(0, len(days), len(days))
        sample = np.concatenate([day_rows[i] for i in chosen])
        x = centered[sample]
        boot[b] = np.abs(x.mean(axis=0) / np.maximum(x.std(axis=0, ddof=1) / np.sqrt(len(x)), 1e-12))
    order = np.argsort(-observed)
    adjusted = np.empty(values.shape[1])
    previous = 0.0
    for rank, hypothesis in enumerate(order):
        remaining = order[rank:]
        p = float(np.mean(np.max(boot[:, remaining], axis=1) >= observed[hypothesis]))
        previous = max(previous, p)
        adjusted[hypothesis] = previous
    return adjusted
