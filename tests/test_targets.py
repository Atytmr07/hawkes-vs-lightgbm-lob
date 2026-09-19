import numpy as np

from src.features.targets import target_a, target_b


def test_price_move_and_pure_depletion_are_separate():
    time = np.array([0, 100, 200, 300], dtype=np.int64)
    bid = np.array([100.0, 100.0, 99.0, 100.0])
    ask = np.array([101.0, 101.0, 101.0, 101.0])
    bq = np.array([10.0, 4.0, 4.0, 3.0])
    aq = np.array([10.0, 10.0, 10.0, 10.0])
    sample = np.array([0], dtype=np.int64)
    assert target_a(time, bid, ask, sample, 50, 200).tolist() == [1]
    assert target_b(time, bid, bq, ask, aq, sample, 0.5, 50, 200).tolist() == [0]
