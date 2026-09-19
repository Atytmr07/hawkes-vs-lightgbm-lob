"""Minute-block Giacomini-White unconditional CPA test."""

from __future__ import annotations

import numpy as np

from src.utils.metrics import newey_west_mean_test


def minute_block_gw(timestamp_ms: np.ndarray, loss_a: np.ndarray, loss_b: np.ndarray) -> dict[str, float | int | str]:
    differential = np.asarray(loss_a) - np.asarray(loss_b)
    minute = np.asarray(timestamp_ms, np.int64) // 60_000
    unique, inverse = np.unique(minute, return_inverse=True)
    sums = np.bincount(inverse, weights=differential)
    counts = np.bincount(inverse)
    block_means = sums / counts
    mean, statistic, p_value = newey_west_mean_test(block_means)
    return {
        "minute_blocks": int(len(unique)), "mean_loss_differential": mean,
        "gw_statistic": statistic, "p_value": p_value,
        "label": "pilot, low power",
    }
