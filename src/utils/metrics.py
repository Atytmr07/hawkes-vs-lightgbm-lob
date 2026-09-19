"""Dependency-light binary metrics and HAC inference."""

from __future__ import annotations

import numpy as np


def average_precision(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.uint8)
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(-np.asarray(score), kind="stable")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y) + 1)
    return float((precision * sorted_y).sum() / positives)


def brier_score(y_true: np.ndarray, probability: np.ndarray) -> float:
    return float(np.mean((np.asarray(y_true) - np.asarray(probability)) ** 2))


def log_loss_vector(y_true: np.ndarray, probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability), 1e-12, 1 - 1e-12)
    y = np.asarray(y_true)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def newey_west_mean_test(values: np.ndarray, lags: int | None = None) -> tuple[float, float, float]:
    from scipy.stats import norm

    x = np.asarray(values, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return float("nan"), float("nan"), float("nan")
    centered = x - x.mean()
    if lags is None:
        lags = min(n - 1, int(np.floor(4 * (n / 100.0) ** (2 / 9))))
    long_run = float(centered @ centered / n)
    for lag in range(1, lags + 1):
        covariance = float(centered[lag:] @ centered[:-lag] / n)
        long_run += 2 * (1 - lag / (lags + 1)) * covariance
    se = np.sqrt(max(long_run, 0.0) / n)
    statistic = float(x.mean() / se) if se > 0 else float("nan")
    p_value = float(2 * norm.sf(abs(statistic))) if np.isfinite(statistic) else float("nan")
    return float(x.mean()), statistic, p_value
