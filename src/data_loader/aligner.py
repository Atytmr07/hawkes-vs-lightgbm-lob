"""Causally align BTCUSDT trades and L1 bookTicker transitions.

Each changed book side is one quote transition.  A bookTicker row can therefore
emit both a bid and an ask event.  Quantity drops fully explained by executions
at the old best price do not emit a duplicate cancellation event.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Literal, Mapping

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


SYMBOL = "BTCUSDT"
JITTER_NS = 100
NS_PER_MS = 1_000_000
MAX_JITTER_RANK = NS_PER_MS // JITTER_NS
FLOAT_EPS = 1e-12

EVENT_TYPES = (
    "bid_qty_up",
    "bid_qty_down",
    "bid_price_up",
    "bid_price_down",
    "ask_qty_up",
    "ask_qty_down",
    "ask_price_up",
    "ask_price_down",
    "trade_buy",
    "trade_sell",
)
EVENT_CODE = {name: code for code, name in enumerate(EVENT_TYPES)}

OUTPUT_SCHEMA = pa.schema(
    [
        ("event_time_ns", pa.int64()),
        ("transaction_time", pa.int64()),
        ("event_code", pa.uint8()),
        ("event_type", pa.string()),
        ("price", pa.float64()),
        ("volume", pa.float64()),
        ("source", pa.string()),
        ("source_id", pa.int64()),
    ]
)


@dataclass
class AlignmentStats:
    trades: int = 0
    quote_events: int = 0
    quote_side_transitions: int = 0
    trade_explained_qty_down: int = 0
    unchanged_quote_sides: int = 0
    unclassified_quote_transitions: int = 0
    raw_book_order_violations: int = 0
    raw_trade_order_violations: int = 0
    event_counts: Counter[str] = field(default_factory=Counter)

    def merge(self, other: "AlignmentStats") -> None:
        for name in (
            "trades",
            "quote_events",
            "quote_side_transitions",
            "trade_explained_qty_down",
            "unchanged_quote_sides",
            "unclassified_quote_transitions",
            "raw_book_order_violations",
            "raw_trade_order_violations",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        self.event_counts.update(other.event_counts)

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["event_counts"] = dict(sorted(self.event_counts.items()))
        return result


def _empty_events() -> dict[str, np.ndarray]:
    return {
        "transaction_time": np.empty(0, dtype=np.int64),
        "event_code": np.empty(0, dtype=np.uint8),
        "price": np.empty(0, dtype=np.float64),
        "volume": np.empty(0, dtype=np.float64),
        "source_id": np.empty(0, dtype=np.int64),
        "side_order": np.empty(0, dtype=np.uint8),
    }


def _concat_event_parts(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not parts:
        return _empty_events()
    return {name: np.concatenate([part[name] for part in parts]) for name in parts[0]}


def _append_quote_part(
    parts: list[dict[str, np.ndarray]],
    mask: np.ndarray,
    *,
    code: int,
    transaction_time: np.ndarray,
    price: np.ndarray,
    volume: np.ndarray,
    update_id: np.ndarray,
    side_order: int,
) -> None:
    if not np.any(mask):
        return
    count = int(mask.sum())
    parts.append(
        {
            "transaction_time": transaction_time[mask],
            "event_code": np.full(count, code, dtype=np.uint8),
            "price": price[mask],
            "volume": volume[mask],
            "source_id": update_id[mask],
            "side_order": np.full(count, side_order, dtype=np.uint8),
        }
    )


def classify_quote_block(
    *,
    transaction_time: np.ndarray,
    update_id: np.ndarray,
    previous_bid_price: np.ndarray,
    previous_bid_qty: np.ndarray,
    previous_ask_price: np.ndarray,
    previous_ask_qty: np.ndarray,
    bid_price: np.ndarray,
    bid_qty: np.ndarray,
    ask_price: np.ndarray,
    ask_qty: np.ndarray,
    trade_time: np.ndarray,
    trade_price: np.ndarray,
    trade_qty: np.ndarray,
    attribution_side: Literal["left", "right"] = "left",
) -> tuple[dict[str, np.ndarray], AlignmentStats]:
    """Classify one complete-time quote block.

    ``attribution_side='left'`` implements trades-before-book: a trade at T is
    assigned to the first quote at T.  ``'right'`` assigns it to the first quote
    strictly after T for the inverted sensitivity rule.
    """

    n = len(transaction_time)
    arrays = (
        update_id,
        previous_bid_price,
        previous_bid_qty,
        previous_ask_price,
        previous_ask_qty,
        bid_price,
        bid_qty,
        ask_price,
        ask_qty,
    )
    if any(len(array) != n for array in arrays):
        raise ValueError("all quote arrays must have the same length")

    bid_trade_volume = np.zeros(n, dtype=np.float64)
    ask_trade_volume = np.zeros(n, dtype=np.float64)
    if len(trade_time):
        target = np.searchsorted(transaction_time, trade_time, side=attribution_side)
        valid = target < n
        target_valid = target[valid]
        price_valid = trade_price[valid]
        qty_valid = trade_qty[valid]
        bid_match = price_valid == previous_bid_price[target_valid]
        ask_match = price_valid == previous_ask_price[target_valid]
        if np.any(bid_match):
            bid_trade_volume += np.bincount(
                target_valid[bid_match], weights=qty_valid[bid_match], minlength=n
            )
        if np.any(ask_match):
            ask_trade_volume += np.bincount(
                target_valid[ask_match], weights=qty_valid[ask_match], minlength=n
            )

    bid_delta = bid_qty - previous_bid_qty
    ask_delta = ask_qty - previous_ask_qty
    bid_same_price = bid_price == previous_bid_price
    ask_same_price = ask_price == previous_ask_price

    masks: dict[str, np.ndarray] = {
        "bid_price_up": bid_price > previous_bid_price,
        "bid_price_down": bid_price < previous_bid_price,
        "ask_price_up": ask_price > previous_ask_price,
        "ask_price_down": ask_price < previous_ask_price,
        "bid_qty_up": bid_same_price & (bid_delta > FLOAT_EPS),
        "ask_qty_up": ask_same_price & (ask_delta > FLOAT_EPS),
    }

    bid_drop = bid_same_price & (bid_delta < -FLOAT_EPS)
    ask_drop = ask_same_price & (ask_delta < -FLOAT_EPS)
    bid_cancel = np.maximum(0.0, -bid_delta - bid_trade_volume)
    ask_cancel = np.maximum(0.0, -ask_delta - ask_trade_volume)
    masks["bid_qty_down"] = bid_drop & (bid_cancel > FLOAT_EPS)
    masks["ask_qty_down"] = ask_drop & (ask_cancel > FLOAT_EPS)
    bid_explained = bid_drop & ~masks["bid_qty_down"]
    ask_explained = ask_drop & ~masks["ask_qty_down"]

    bid_changed = ~(
        bid_same_price & (np.abs(bid_delta) <= FLOAT_EPS)
    )
    ask_changed = ~(
        ask_same_price & (np.abs(ask_delta) <= FLOAT_EPS)
    )
    bid_covered = (
        masks["bid_price_up"]
        | masks["bid_price_down"]
        | masks["bid_qty_up"]
        | masks["bid_qty_down"]
        | bid_explained
    )
    ask_covered = (
        masks["ask_price_up"]
        | masks["ask_price_down"]
        | masks["ask_qty_up"]
        | masks["ask_qty_down"]
        | ask_explained
    )
    unclassified = int((bid_changed & ~bid_covered).sum() + (ask_changed & ~ask_covered).sum())

    parts: list[dict[str, np.ndarray]] = []
    specifications = (
        ("bid_qty_up", bid_price, bid_delta, 0),
        ("bid_qty_down", bid_price, bid_cancel, 0),
        ("bid_price_up", bid_price, bid_qty, 0),
        ("bid_price_down", bid_price, bid_qty, 0),
        ("ask_qty_up", ask_price, ask_delta, 1),
        ("ask_qty_down", ask_price, ask_cancel, 1),
        ("ask_price_up", ask_price, ask_qty, 1),
        ("ask_price_down", ask_price, ask_qty, 1),
    )
    for event_type, prices, volumes, side_order in specifications:
        _append_quote_part(
            parts,
            masks[event_type],
            code=EVENT_CODE[event_type],
            transaction_time=transaction_time,
            price=prices,
            volume=volumes,
            update_id=update_id,
            side_order=side_order,
        )

    events = _concat_event_parts(parts)
    stats = AlignmentStats(
        quote_events=len(events["event_code"]),
        quote_side_transitions=int(bid_changed.sum() + ask_changed.sum()),
        trade_explained_qty_down=int(bid_explained.sum() + ask_explained.sum()),
        unchanged_quote_sides=int((~bid_changed).sum() + (~ask_changed).sum()),
        unclassified_quote_transitions=unclassified,
    )
    stats.event_counts.update(
        {name: int(mask.sum()) for name, mask in masks.items() if np.any(mask)}
    )
    return events, stats


def build_trade_events(
    *,
    transaction_time: np.ndarray,
    trade_id: np.ndarray,
    price: np.ndarray,
    qty: np.ndarray,
    is_buyer_maker: np.ndarray,
) -> tuple[dict[str, np.ndarray], AlignmentStats]:
    count = len(transaction_time)
    codes = np.where(
        is_buyer_maker,
        EVENT_CODE["trade_sell"],
        EVENT_CODE["trade_buy"],
    ).astype(np.uint8)
    events = {
        "transaction_time": transaction_time,
        "event_code": codes,
        "price": price,
        "volume": qty,
        "source_id": trade_id,
        "side_order": np.zeros(count, dtype=np.uint8),
    }
    buy_count = int((~is_buyer_maker).sum())
    sell_count = int(is_buyer_maker.sum())
    stats = AlignmentStats(trades=count)
    stats.event_counts.update({"trade_buy": buy_count, "trade_sell": sell_count})
    return events, stats


def merge_and_jitter_events(
    quote_events: Mapping[str, np.ndarray],
    trade_events: Mapping[str, np.ndarray],
    *,
    tie_rule: Literal["trades_first", "book_first"] = "trades_first",
) -> pa.Table:
    q_count = len(quote_events["event_code"])
    t_count = len(trade_events["event_code"])
    if q_count + t_count == 0:
        return pa.Table.from_pylist([], schema=OUTPUT_SCHEMA)

    transaction_time = np.concatenate(
        [quote_events["transaction_time"], trade_events["transaction_time"]]
    ).astype(np.int64, copy=False)
    event_code = np.concatenate([quote_events["event_code"], trade_events["event_code"]])
    price = np.concatenate([quote_events["price"], trade_events["price"]])
    volume = np.concatenate([quote_events["volume"], trade_events["volume"]])
    source_id = np.concatenate([quote_events["source_id"], trade_events["source_id"]])
    side_order = np.concatenate([quote_events["side_order"], trade_events["side_order"]])

    quote_order, trade_order = ((1, 0) if tie_rule == "trades_first" else (0, 1))
    feed_order = np.concatenate(
        [
            np.full(q_count, quote_order, dtype=np.uint8),
            np.full(t_count, trade_order, dtype=np.uint8),
        ]
    )
    order = np.lexsort((side_order, source_id, feed_order, transaction_time))
    transaction_time = transaction_time[order]
    event_code = event_code[order]
    price = price[order]
    volume = volume[order]
    source_id = source_id[order]

    starts = np.empty(len(transaction_time), dtype=bool)
    starts[0] = True
    starts[1:] = transaction_time[1:] != transaction_time[:-1]
    indices = np.arange(len(transaction_time), dtype=np.int64)
    group_start = np.maximum.accumulate(np.where(starts, indices, 0))
    tie_rank = indices - group_start
    if tie_rank.max(initial=0) >= MAX_JITTER_RANK:
        raise RuntimeError(
            f"at least one millisecond contains {tie_rank.max() + 1:,} events; "
            "100 ns jitter would cross the next real millisecond"
        )
    event_time_ns = transaction_time * NS_PER_MS + tie_rank * JITTER_NS

    names = np.asarray(EVENT_TYPES, dtype=object)[event_code]
    sources = np.where(event_code >= EVENT_CODE["trade_buy"], "trades", "bookTicker")
    return pa.table(
        {
            "event_time_ns": event_time_ns,
            "transaction_time": transaction_time,
            "event_code": event_code,
            "event_type": names,
            "price": price,
            "volume": volume,
            "source": sources,
            "source_id": source_id,
        },
        schema=OUTPUT_SCHEMA,
    )


def _column_numpy(table: pa.Table, name: str) -> np.ndarray:
    return table.column(name).combine_chunks().to_numpy(zero_copy_only=False)


def _sort_inputs(book: pa.Table, trades: pa.Table) -> tuple[pa.Table, pa.Table, int, int]:
    book_time = _column_numpy(book, "transaction_time")
    book_id = _column_numpy(book, "update_id")
    book_bad = (book_time[1:] < book_time[:-1]) | (
        (book_time[1:] == book_time[:-1]) & (book_id[1:] < book_id[:-1])
    )
    raw_book_violations = int(book_bad.sum())
    if raw_book_violations:
        indices = pc.sort_indices(
            book,
            sort_keys=[("transaction_time", "ascending"), ("update_id", "ascending")],
        )
        book = pc.take(book, indices)

    trade_time = _column_numpy(trades, "time")
    trade_id = _column_numpy(trades, "id")
    trade_bad = (trade_time[1:] < trade_time[:-1]) | (
        (trade_time[1:] == trade_time[:-1]) & (trade_id[1:] < trade_id[:-1])
    )
    raw_trade_violations = int(trade_bad.sum())
    if raw_trade_violations:
        indices = pc.sort_indices(
            trades,
            sort_keys=[("time", "ascending"), ("id", "ascending")],
        )
        trades = pc.take(trades, indices)
    return book, trades, raw_book_violations, raw_trade_violations


def _slice_numpy(table: pa.Table, name: str, offset: int, length: int) -> np.ndarray:
    return table.column(name).slice(offset, length).combine_chunks().to_numpy(zero_copy_only=False)


def _trade_slice(
    trade_time: np.ndarray,
    lower: int,
    upper: int,
    *,
    lower_inclusive: bool,
    upper_inclusive: bool,
) -> slice:
    left = np.searchsorted(trade_time, lower, side="left" if lower_inclusive else "right")
    right = np.searchsorted(trade_time, upper, side="right" if upper_inclusive else "left")
    return slice(int(left), int(right))


def align_day(
    *,
    day: date,
    parquet_dir: Path = Path("data/parquet"),
    aligned_dir: Path = Path("data/parquet/aligned"),
    tie_rule: Literal["trades_first", "book_first"] = "trades_first",
    quote_chunk_rows: int = 1_000_000,
    symbol: str = SYMBOL,
) -> tuple[Path, AlignmentStats]:
    trade_path = parquet_dir / "trades" / f"{symbol}-trades-{day.isoformat()}.parquet"
    book_path = parquet_dir / "bookTicker" / f"{symbol}-bookTicker-{day.isoformat()}.parquet"
    if not trade_path.exists() or not book_path.exists():
        raise FileNotFoundError(f"missing input Parquet for {day}: {trade_path}, {book_path}")

    book = pq.read_table(
        book_path,
        columns=[
            "update_id",
            "best_bid_price",
            "best_bid_qty",
            "best_ask_price",
            "best_ask_qty",
            "transaction_time",
        ],
        memory_map=True,
    )
    trades = pq.read_table(
        trade_path,
        columns=["id", "price", "qty", "time", "is_buyer_maker"],
        memory_map=True,
    )
    book, trades, raw_book_bad, raw_trade_bad = _sort_inputs(book, trades)

    book_time = _column_numpy(book, "transaction_time").astype(np.int64, copy=False)
    trade_time = _column_numpy(trades, "time").astype(np.int64, copy=False)
    trade_id = _column_numpy(trades, "id").astype(np.int64, copy=False)
    trade_price = _column_numpy(trades, "price").astype(np.float64, copy=False)
    trade_qty = _column_numpy(trades, "qty").astype(np.float64, copy=False)
    is_buyer_maker = _column_numpy(trades, "is_buyer_maker").astype(bool, copy=False)

    aligned_dir.mkdir(parents=True, exist_ok=True)
    target = aligned_dir / f"{symbol}-events-{day.isoformat()}.parquet"
    temporary = target.with_suffix(target.suffix + ".part")
    temporary.unlink(missing_ok=True)
    writer = pq.ParquetWriter(
        temporary,
        OUTPUT_SCHEMA,
        compression="zstd",
        compression_level=3,
        use_dictionary=["event_type", "source"],
        write_statistics=True,
    )
    total_stats = AlignmentStats(
        raw_book_order_violations=raw_book_bad,
        raw_trade_order_violations=raw_trade_bad,
    )

    try:
        first_quote_time = int(book_time[0])
        leading = _trade_slice(
            trade_time,
            np.iinfo(np.int64).min,
            first_quote_time,
            lower_inclusive=True,
            upper_inclusive=True,
        )
        leading_events, leading_stats = build_trade_events(
            transaction_time=trade_time[leading],
            trade_id=trade_id[leading],
            price=trade_price[leading],
            qty=trade_qty[leading],
            is_buyer_maker=is_buyer_maker[leading],
        )
        if leading_stats.trades:
            writer.write_table(merge_and_jitter_events(_empty_events(), leading_events, tie_rule=tie_rule))
            total_stats.merge(leading_stats)

        start = 1
        while start < len(book_time):
            end = min(start + quote_chunk_rows, len(book_time))
            while end < len(book_time) and book_time[end] == book_time[end - 1]:
                end += 1
            length = end - start
            lower_time = int(book_time[start - 1])
            upper_time = int(book_time[end - 1])

            output_slice = _trade_slice(
                trade_time,
                lower_time,
                upper_time,
                lower_inclusive=False,
                upper_inclusive=True,
            )
            if tie_rule == "trades_first":
                attribution_slice = output_slice
                attribution_side: Literal["left", "right"] = "left"
            else:
                attribution_slice = _trade_slice(
                    trade_time,
                    lower_time,
                    upper_time,
                    lower_inclusive=True,
                    upper_inclusive=False,
                )
                attribution_side = "right"

            previous_bid_price = _slice_numpy(book, "best_bid_price", start - 1, length)
            previous_bid_qty = _slice_numpy(book, "best_bid_qty", start - 1, length)
            previous_ask_price = _slice_numpy(book, "best_ask_price", start - 1, length)
            previous_ask_qty = _slice_numpy(book, "best_ask_qty", start - 1, length)
            quote_events, quote_stats = classify_quote_block(
                transaction_time=book_time[start:end],
                update_id=_slice_numpy(book, "update_id", start, length),
                previous_bid_price=previous_bid_price,
                previous_bid_qty=previous_bid_qty,
                previous_ask_price=previous_ask_price,
                previous_ask_qty=previous_ask_qty,
                bid_price=_slice_numpy(book, "best_bid_price", start, length),
                bid_qty=_slice_numpy(book, "best_bid_qty", start, length),
                ask_price=_slice_numpy(book, "best_ask_price", start, length),
                ask_qty=_slice_numpy(book, "best_ask_qty", start, length),
                trade_time=trade_time[attribution_slice],
                trade_price=trade_price[attribution_slice],
                trade_qty=trade_qty[attribution_slice],
                attribution_side=attribution_side,
            )
            trade_events, trade_stats = build_trade_events(
                transaction_time=trade_time[output_slice],
                trade_id=trade_id[output_slice],
                price=trade_price[output_slice],
                qty=trade_qty[output_slice],
                is_buyer_maker=is_buyer_maker[output_slice],
            )
            merged = merge_and_jitter_events(quote_events, trade_events, tie_rule=tie_rule)
            if merged.num_rows:
                writer.write_table(merged)
            total_stats.merge(quote_stats)
            total_stats.merge(trade_stats)
            if quote_stats.unclassified_quote_transitions:
                raise RuntimeError(
                    f"{quote_stats.unclassified_quote_transitions} unclassified quote transitions "
                    f"in rows {start:,}:{end:,}"
                )
            start = end

        trailing = _trade_slice(
            trade_time,
            int(book_time[-1]),
            np.iinfo(np.int64).max,
            lower_inclusive=False,
            upper_inclusive=True,
        )
        trailing_events, trailing_stats = build_trade_events(
            transaction_time=trade_time[trailing],
            trade_id=trade_id[trailing],
            price=trade_price[trailing],
            qty=trade_qty[trailing],
            is_buyer_maker=is_buyer_maker[trailing],
        )
        if trailing_stats.trades:
            writer.write_table(merge_and_jitter_events(_empty_events(), trailing_events, tie_rule=tie_rule))
            total_stats.merge(trailing_stats)
        writer.close()
        writer = None

        footer_rows = pq.read_metadata(temporary).num_rows
        expected_rows = total_stats.trades + total_stats.quote_events
        if footer_rows != expected_rows:
            raise RuntimeError(
                f"aligned footer has {footer_rows:,} rows, expected {expected_rows:,}"
            )
        if total_stats.trades != len(trades):
            raise RuntimeError(
                f"only {total_stats.trades:,}/{len(trades):,} trades were emitted"
            )
        os.replace(temporary, target)
    except Exception:
        if writer is not None:
            writer.close()
        temporary.unlink(missing_ok=True)
        raise

    stats_path = target.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(total_stats.to_dict(), indent=2) + "\n", encoding="utf-8")
    return target, total_stats


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", type=parse_date, required=True)
    parser.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    parser.add_argument("--aligned-dir", type=Path, default=Path("data/parquet/aligned"))
    parser.add_argument("--tie-rule", choices=("trades_first", "book_first"), default="trades_first")
    parser.add_argument("--quote-chunk-rows", type=int, default=1_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target, stats = align_day(
        day=args.date,
        parquet_dir=args.parquet_dir,
        aligned_dir=args.aligned_dir,
        tie_rule=args.tie_rule,
        quote_chunk_rows=args.quote_chunk_rows,
    )
    print(f"aligned {target}: {target.stat().st_size:,} bytes")
    print(json.dumps(stats.to_dict(), indent=2))


if __name__ == "__main__":
    main()
