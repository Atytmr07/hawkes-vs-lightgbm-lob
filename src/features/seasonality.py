"""Eight-harmonic time-of-day and asymmetric funding-window regressors."""

from __future__ import annotations

import numpy as np

from dataclasses import dataclass


def seasonality_design(timestamp_ms: np.ndarray) -> tuple[np.ndarray, list[str]]:
    seconds = (timestamp_ms.astype(np.float64) / 1000.0) % 86400.0
    columns = []
    names = []
    for harmonic in range(1, 9):
        angle = 2.0 * np.pi * harmonic * seconds / 86400.0
        columns.extend((np.cos(angle), np.sin(angle)))
        names.extend((f"tod_cos_{harmonic}", f"tod_sin_{harmonic}"))
    for hour in (0, 8, 16):
        funding = hour * 3600.0
        wrapped = (seconds - funding + 43200.0) % 86400.0 - 43200.0
        columns.extend(((wrapped >= -300) & (wrapped < 0), (wrapped >= 0) & (wrapped <= 300)))
        names.extend((f"funding_{hour:02d}_pre", f"funding_{hour:02d}_post"))
    return np.column_stack(columns).astype(np.float64), names


@dataclass
class SeasonalBaseline:
    """Positive eight-harmonic baseline times asymmetric funding jumps."""

    eta0: np.ndarray
    harmonic: np.ndarray
    funding: np.ndarray
    epsilon: float = 1e-9

    def rates(self, timestamp_ms: np.ndarray) -> np.ndarray:
        design, _ = seasonality_design(np.asarray(timestamp_ms, np.int64))
        harmonic_design, funding_design = design[:, :16], design[:, 16:]
        amplitudes = np.sqrt(
            self.harmonic[:, 0::2] ** 2 + self.harmonic[:, 1::2] ** 2 + self.epsilon
        )
        a0 = self.epsilon + np.exp(self.eta0) + amplitudes.sum(axis=1)
        base = a0[None, :] + harmonic_design @ self.harmonic.T
        jumps = np.exp(np.clip(funding_design @ self.funding.T, -20.0, 20.0))
        return np.maximum(base * jumps, self.epsilon)
