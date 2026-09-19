"""Fixed-alert-budget recall and lead-time summaries (no PnL)."""

from __future__ import annotations

import numpy as np


def fixed_alert_metrics(y_true: np.ndarray, score: np.ndarray, lead_time_ms: np.ndarray | None = None, rate: float = 0.01) -> dict[str, float]:
    threshold = float(np.quantile(score, 1.0 - rate))
    alerts = score >= threshold
    positives = np.asarray(y_true, bool)
    captured = alerts & positives
    recall = float(captured.sum() / max(1, positives.sum()))
    median_lead = float(np.median(lead_time_ms[captured])) if lead_time_ms is not None and captured.any() else float("nan")
    return {"threshold": threshold, "alert_rate": float(alerts.mean()), "recall": recall, "median_lead_time_ms": median_lead}
