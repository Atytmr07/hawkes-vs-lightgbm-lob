"""Validate an order-only and exact-ms-masked Gamma attribution treatment.

This experiment does not overwrite the §14 artifacts.  It uses the default
trades-first event taxonomy as the common event multiset, constructs a pure
quotes-first ordering in memory, and compares ordinary versus ambiguity-masked
Hawkes fits at the existing N=3d endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.features.ewma_bank import feature_names as ewma_names
from src.features.ofi import M0_NAMES
from src.models.hawkes_mle import HawkesFit, fit_hawkes, fit_seasonal_baseline
from src.models.tree_models import fit_tree
from src.utils.metrics import average_precision, brier_score
from src.validation.causality import (
    gamma_change_diagnostic,
    invert_cross_feed_ties,
    touch_trade_collision_diagnostic,
)
from src.validation.learning_curve import _matrix, _target, _times, load_events, load_features


def _iso(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, timezone.utc).isoformat()


def _event_counts(
    aligned_dir: Path,
    start_ms: int,
    end_ms: int,
) -> list[int]:
    start_day = datetime.fromtimestamp(start_ms / 1000, timezone.utc).date()
    end_day = datetime.fromtimestamp((end_ms - 1) / 1000, timezone.utc).date()
    result = np.zeros(10, dtype=np.int64)
    current = start_day
    while current <= end_day:
        path = aligned_dir / f"BTCUSDT-events-{current}.parquet"
        table = pq.read_table(path, columns=["event_time_ns", "event_code"], memory_map=True)
        times = table["event_time_ns"].combine_chunks().to_numpy(zero_copy_only=False)
        codes = table["event_code"].combine_chunks().to_numpy(zero_copy_only=False)
        keep = (times >= start_ms * 1_000_000) & (times < end_ms * 1_000_000)
        result += np.bincount(codes[keep], minlength=10)
        current = date.fromordinal(current.toordinal() + 1)
    return result.tolist()


def _fit_summary(fit: HawkesFit) -> dict[str, object]:
    return {
        "spectral_radius": fit.spectral_radius,
        "stationarity_scale": fit.stationarity_scale,
        "converged_rows": int(sum(fit.converged)),
        "selected_events": fit.selected_events,
        "tie_treatment": fit.tie_treatment,
        "penalized_negative_loglik": fit.penalized_negative_loglik,
        "gamma": fit.gamma.tolist(),
    }


def _m2_score(
    fit: HawkesFit,
    training: pa.Table,
    testing: pa.Table,
) -> dict[str, float]:
    m0_names = list(M0_NAMES)
    train_m0 = _matrix(training, m0_names)
    test_m0 = _matrix(testing, m0_names)
    train_z = _matrix(training, ewma_names())
    test_z = _matrix(testing, ewma_names())
    train_intensity = fit.intensities(
        train_z, sample_time_ms=_times(training)
    ).astype(np.float32)
    test_intensity = fit.intensities(
        test_z, sample_time_ms=_times(testing)
    ).astype(np.float32)
    names = m0_names + [f"hawkes_lambda_{i}" for i in range(10)]
    model = fit_tree(
        np.column_stack((train_m0, train_intensity)), _target(training), names
    )
    prediction = model.predict(np.column_stack((test_m0, test_intensity)))
    target = _target(testing)
    return {
        "pr_auc": average_precision(target, prediction),
        "brier": brier_score(target, prediction),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", type=Path, default=Path("data/features_38day"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned_38day"))
    parser.add_argument(
        "--legacy-book-first-aligned-dir",
        type=Path,
        default=Path("data/parquet/aligned_38day_book_first"),
    )
    parser.add_argument("--output", type=Path, default=Path("data/results/gamma_attribution_validation.json"))
    parser.add_argument("--max-mle-events", type=int, default=100_000)
    args = parser.parse_args()

    features = load_features(args.feature_dir, date(2024, 3, 21), date(2024, 3, 30))
    feature_times = _times(features)
    test_start_ms = int(np.datetime64("2024-03-24", "ms").astype(np.int64))
    boundary = int(feature_times[feature_times < test_start_ms].max()) + 1_000
    requested_lower = boundary - 3 * 86_400_000
    train_mask = (feature_times >= requested_lower) & (feature_times < boundary)
    test_mask = feature_times >= test_start_ms
    training = features.filter(pa.array(train_mask))
    testing = features.filter(pa.array(test_mask))
    observed_lower = int(_times(training).min())

    default_times, default_codes = load_events(args.aligned_dir, observed_lower, boundary)
    inverted_times, inverted_codes = invert_cross_feed_ties(default_times, default_codes)
    if not np.array_equal(np.bincount(default_codes, minlength=10), np.bincount(inverted_codes, minlength=10)):
        raise RuntimeError("order-only inversion changed the event multiset")

    print(f"calibration events: {len(default_codes):,}", flush=True)
    collisions = touch_trade_collision_diagnostic(default_times, default_codes)
    print(f"ambiguous touch/trade groups: {collisions['ambiguous_groups']:,}", flush=True)

    fits: dict[str, HawkesFit] = {}
    for label, times, codes, treatment in (
        ("ordered_trades_first", default_times, default_codes, "ordered"),
        ("ordered_book_first_order_only", inverted_times, inverted_codes, "ordered"),
        ("masked_trades_first", default_times, default_codes, "mask_touch_trade"),
        ("masked_book_first_order_only", inverted_times, inverted_codes, "mask_touch_trade"),
    ):
        print(f"fitting {label}", flush=True)
        fits[label] = fit_hawkes(
            times,
            codes,
            max_events=args.max_mle_events,
            maxiter=2_000,
            tie_treatment=treatment,
        )
        print(
            f"{label}: converged={sum(fits[label].converged)}/10 "
            f"rho={fits[label].spectral_radius:.10f}",
            flush=True,
        )

    ordered_diagnostic = gamma_change_diagnostic(
        fits["ordered_trades_first"].gamma,
        fits["ordered_book_first_order_only"].gamma,
    )
    masked_diagnostic = gamma_change_diagnostic(
        fits["masked_trades_first"].gamma,
        fits["masked_book_first_order_only"].gamma,
    )

    scores: dict[str, dict[str, float]] = {}
    seasonal: dict[str, dict[str, object]] = {}
    for label in ("masked_trades_first", "masked_book_first_order_only"):
        print(f"profiling seasonality and scoring {label}", flush=True)
        fit = fits[label]
        event_codes = default_codes if label == "masked_trades_first" else inverted_codes
        event_times = default_times if label == "masked_trades_first" else inverted_times
        profile = fit_seasonal_baseline(
            fit,
            event_times,
            event_codes,
            max_events=args.max_mle_events,
            maxiter=1_000,
        )
        seasonal[label] = {
            "loglik_improvement": float(profile["loglik_improvement"]),
            "converged_rows": int(sum(profile["converged"])),
        }
        scores[label] = _m2_score(fit, training, testing)

    legacy_path = Path("data/results/pilot_38day_causality_sensitivity.json")
    legacy = json.loads(legacy_path.read_text(encoding="utf-8"))
    default_counts = np.bincount(default_codes, minlength=10).astype(np.int64)
    legacy_book_counts = np.asarray(
        _event_counts(args.legacy_book_first_aligned_dir, observed_lower, boundary),
        dtype=np.int64,
    )
    result = {
        "design": {
            "endpoint": "N=3d, fixed test 2024-03-24 through 2024-03-30",
            "requested_lower": _iso(requested_lower),
            "observed_lower": _iso(observed_lower),
            "boundary": _iso(boundary),
            "train_rows": training.num_rows,
            "test_rows": testing.num_rows,
            "max_mle_events": args.max_mle_events,
            "common_event_multiset": "trades_first taxonomy; only cross-feed order inverted",
        },
        "legacy_confounded_comparison": {
            "gamma_relative_frobenius_change": legacy["gamma_relative_frobenius_change"],
            "m2_pr_auc_delta_book_minus_default": legacy["m2_pr_auc_delta_book_minus_default"],
            "default_event_counts": default_counts.tolist(),
            "legacy_book_first_event_counts": legacy_book_counts.tolist(),
            "event_count_delta_book_minus_default": (legacy_book_counts - default_counts).tolist(),
            "total_event_count_delta": int(legacy_book_counts.sum() - default_counts.sum()),
        },
        "touch_trade_collisions": collisions,
        "fits": {label: _fit_summary(fit) for label, fit in fits.items()},
        "order_only_gamma_diagnostic": ordered_diagnostic,
        "masked_gamma_diagnostic": masked_diagnostic,
        "seasonal_profiles": seasonal,
        "common_features_m2": scores,
        "masked_m2_pr_auc_delta_book_minus_default": (
            scores["masked_book_first_order_only"]["pr_auc"]
            - scores["masked_trades_first"]["pr_auc"]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
