import unittest

from hogan_bot.decision import apply_ml_filter


class MlFilterTests(unittest.TestCase):
    def test_buy_blocked_below_threshold(self):
        self.assertEqual(apply_ml_filter("buy", 0.55, 0.6, 0.4), "hold")

    def test_sell_blocked_above_threshold(self):
        self.assertEqual(apply_ml_filter("sell", 0.41, 0.6, 0.4), "hold")

    def test_signal_kept_when_prob_in_range(self):
        self.assertEqual(apply_ml_filter("buy", 0.72, 0.55, 0.45), "buy")
        self.assertEqual(apply_ml_filter("sell", 0.32, 0.55, 0.45), "sell")


if __name__ == "__main__":
    unittest.main()
