import numpy as np

from src.data_loader.aligner import (
    EVENT_CODE,
    EVENT_TYPES,
    build_trade_events,
    classify_quote_block,
    merge_and_jitter_events,
)


def _eight_quote_cases():
    n = 8
    previous_bid_price = np.full(n, 100.0)
    previous_bid_qty = np.full(n, 10.0)
    previous_ask_price = np.full(n, 101.0)
    previous_ask_qty = np.full(n, 20.0)
    bid_price = previous_bid_price.copy()
    bid_qty = previous_bid_qty.copy()
    ask_price = previous_ask_price.copy()
    ask_qty = previous_ask_qty.copy()

    bid_qty[0] = 12.0
    bid_qty[1] = 7.0
    bid_price[2] = 101.0
    bid_price[3] = 99.0
    ask_qty[4] = 23.0
    ask_qty[5] = 17.0
    ask_price[6] = 102.0
    ask_price[7] = 100.0

    return dict(
        transaction_time=np.arange(1, n + 1, dtype=np.int64),
        update_id=np.arange(101, 101 + n, dtype=np.int64),
        previous_bid_price=previous_bid_price,
        previous_bid_qty=previous_bid_qty,
        previous_ask_price=previous_ask_price,
        previous_ask_qty=previous_ask_qty,
        bid_price=bid_price,
        bid_qty=bid_qty,
        ask_price=ask_price,
        ask_qty=ask_qty,
        trade_time=np.empty(0, dtype=np.int64),
        trade_price=np.empty(0, dtype=np.float64),
        trade_qty=np.empty(0, dtype=np.float64),
    )


def test_all_eight_quote_types_are_exhaustive():
    events, stats = classify_quote_block(**_eight_quote_cases())

    assert set(events["event_code"].tolist()) == set(range(8))
    assert stats.quote_events == 8
    assert stats.quote_side_transitions == 8
    assert stats.unchanged_quote_sides == 8
    assert stats.trade_explained_qty_down == 0
    assert stats.unclassified_quote_transitions == 0
    assert sum(stats.event_counts.values()) == 8


def test_trade_volume_is_removed_from_residual_cancellation():
    common = dict(
        transaction_time=np.array([100], dtype=np.int64),
        update_id=np.array([10], dtype=np.int64),
        previous_bid_price=np.array([100.0]),
        previous_bid_qty=np.array([10.0]),
        previous_ask_price=np.array([101.0]),
        previous_ask_qty=np.array([20.0]),
        bid_price=np.array([100.0]),
        bid_qty=np.array([6.0]),
        ask_price=np.array([101.0]),
        ask_qty=np.array([20.0]),
        trade_time=np.array([100], dtype=np.int64),
        trade_price=np.array([100.0]),
    )

    explained, explained_stats = classify_quote_block(
        **common, trade_qty=np.array([4.0])
    )
    assert explained["event_code"].size == 0
    assert explained_stats.trade_explained_qty_down == 1
    assert explained_stats.unclassified_quote_transitions == 0

    residual, residual_stats = classify_quote_block(
        **common, trade_qty=np.array([1.0])
    )
    assert residual["event_code"].tolist() == [EVENT_CODE["bid_qty_down"]]
    assert residual["volume"].tolist() == [3.0]
    assert residual_stats.trade_explained_qty_down == 0
    assert residual_stats.unclassified_quote_transitions == 0


def test_one_book_update_can_emit_bid_and_ask_transitions():
    values = _eight_quote_cases()
    for key in values:
        values[key] = values[key][:1]
    values["bid_qty"] = np.array([12.0])
    values["ask_qty"] = np.array([23.0])

    events, stats = classify_quote_block(**values)

    assert set(events["event_code"].tolist()) == {
        EVENT_CODE["bid_qty_up"],
        EVENT_CODE["ask_qty_up"],
    }
    assert stats.quote_side_transitions == 2
    assert stats.unclassified_quote_transitions == 0


def test_trade_direction_and_default_cross_feed_ordering():
    trade_events, stats = build_trade_events(
        transaction_time=np.array([100, 100], dtype=np.int64),
        trade_id=np.array([2, 1], dtype=np.int64),
        price=np.array([100.0, 100.0]),
        qty=np.array([1.0, 2.0]),
        is_buyer_maker=np.array([True, False]),
    )
    quote_events = {
        "transaction_time": np.array([100], dtype=np.int64),
        "event_code": np.array([EVENT_CODE["bid_qty_up"]], dtype=np.uint8),
        "price": np.array([100.0]),
        "volume": np.array([3.0]),
        "source_id": np.array([50], dtype=np.int64),
        "side_order": np.array([0], dtype=np.uint8),
    }

    merged = merge_and_jitter_events(quote_events, trade_events).to_pydict()

    assert stats.event_counts == {"trade_buy": 1, "trade_sell": 1}
    assert merged["event_type"] == ["trade_buy", "trade_sell", "bid_qty_up"]
    assert merged["source_id"] == [1, 2, 50]
    assert merged["event_time_ns"] == [100_000_000, 100_000_100, 100_000_200]


def test_inverted_cross_feed_ordering_is_deterministic():
    trade_events, _ = build_trade_events(
        transaction_time=np.array([100], dtype=np.int64),
        trade_id=np.array([1], dtype=np.int64),
        price=np.array([100.0]),
        qty=np.array([1.0]),
        is_buyer_maker=np.array([False]),
    )
    quote_events = {
        "transaction_time": np.array([100], dtype=np.int64),
        "event_code": np.array([EVENT_CODE["bid_qty_up"]], dtype=np.uint8),
        "price": np.array([100.0]),
        "volume": np.array([1.0]),
        "source_id": np.array([50], dtype=np.int64),
        "side_order": np.array([0], dtype=np.uint8),
    }

    merged = merge_and_jitter_events(
        quote_events, trade_events, tie_rule="book_first"
    ).to_pydict()

    assert merged["event_type"] == ["bid_qty_up", "trade_buy"]
    assert merged["event_time_ns"] == [100_000_000, 100_000_100]
    assert set(EVENT_TYPES) == {
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
    }
