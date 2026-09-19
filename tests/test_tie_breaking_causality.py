import numpy as np

from src.data_loader.aligner import EVENT_CODE, build_trade_events, merge_and_jitter_events
from src.features.ewma_bank import BETAS, state_before_selected_events_mask_touch_trade
from src.validation.causality import (
    gamma_change_diagnostic,
    invert_cross_feed_ties,
    touch_trade_collision_diagnostic,
)


def test_causality_switch_only_changes_cross_feed_tie_order():
    trades, _ = build_trade_events(
        transaction_time=np.array([7], dtype=np.int64), trade_id=np.array([1]),
        price=np.array([100.0]), qty=np.array([1.0]), is_buyer_maker=np.array([False]),
    )
    quotes = {
        "transaction_time": np.array([7], dtype=np.int64),
        "event_code": np.array([EVENT_CODE["ask_price_up"]], dtype=np.uint8),
        "price": np.array([101.0]), "volume": np.array([2.0]),
        "source_id": np.array([9], dtype=np.int64), "side_order": np.array([1], dtype=np.uint8),
    }
    default = merge_and_jitter_events(quotes, trades, tie_rule="trades_first").to_pydict()
    inverted = merge_and_jitter_events(quotes, trades, tie_rule="book_first").to_pydict()
    assert default["event_type"] == ["trade_buy", "ask_price_up"]
    assert inverted["event_type"] == ["ask_price_up", "trade_buy"]
    assert default["event_time_ns"] == inverted["event_time_ns"]


def test_gamma_diagnostic_separates_trade_quote_directions():
    default = np.eye(10)
    inverted = default.copy()
    inverted[0, 8] += 3.0  # trade_buy -> bid_qty_up
    inverted[9, 1] += 2.0  # bid_qty_down -> trade_sell
    inverted[2, 3] += 1.0  # quote -> quote
    report = gamma_change_diagnostic(default, inverted, top_k=3)
    blocks = report["blocks"]
    assert blocks["trade_to_quote"]["delta_frobenius"] == 3.0
    assert blocks["quote_to_trade"]["delta_frobenius"] == 2.0
    assert blocks["quote_to_quote"]["delta_frobenius"] == 1.0
    assert [term["direction"] for term in report["top_terms"]] == [
        "trade_to_quote", "quote_to_trade", "quote_to_quote"
    ]


def test_order_only_inversion_preserves_multiset_and_masked_states():
    millisecond = 7_000_000
    times = millisecond + np.arange(4, dtype=np.int64) * 100
    codes = np.array([
        EVENT_CODE["trade_buy"], EVENT_CODE["trade_sell"],
        EVENT_CODE["ask_price_up"], EVENT_CODE["bid_price_down"],
    ], dtype=np.uint8)
    inverted_times, inverted_codes = invert_cross_feed_ties(times, codes)
    assert inverted_codes.tolist() == [
        EVENT_CODE["ask_price_up"], EVENT_CODE["bid_price_down"],
        EVENT_CODE["trade_buy"], EVENT_CODE["trade_sell"],
    ]
    assert np.array_equal(np.sort(codes), np.sort(inverted_codes))

    # Use observed millisecond time, not artificial jitter, in the estimator.
    coarse = (times // 1_000_000) * 1_000_000
    selected = np.arange(4, dtype=np.int64)
    default_state = state_before_selected_events_mask_touch_trade(
        coarse, codes, selected, BETAS
    )
    inverted_state = state_before_selected_events_mask_touch_trade(
        coarse, inverted_codes, selected, BETAS
    )
    for event_code in codes:
        left = default_state[codes == event_code]
        right = inverted_state[inverted_codes == event_code]
        assert np.allclose(left, right)

    diagnostic = touch_trade_collision_diagnostic(times, codes)
    assert diagnostic["ambiguous_groups"] == 1
    assert diagnostic["trades_in_ambiguous_groups"] == 2
    assert diagnostic["touches_in_ambiguous_groups"] == 2


def test_touch_mask_does_not_hide_quantity_only_cross_feed_order():
    coarse = np.full(2, 9_000_000, dtype=np.int64)
    codes = np.array([
        EVENT_CODE["trade_buy"], EVENT_CODE["ask_qty_down"],
    ], dtype=np.uint8)
    selected = np.arange(2, dtype=np.int64)
    states = state_before_selected_events_mask_touch_trade(coarse, codes, selected, BETAS)
    trade_buy_beta0 = EVENT_CODE["trade_buy"] * len(BETAS)
    assert states[1, trade_buy_beta0] == 1.0
