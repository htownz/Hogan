import unittest

from hogan_bot.risk import DrawdownGuard, calculate_position_size


class RiskTests(unittest.TestCase):
    def test_position_size_respects_allocation_cap(self):
        size = calculate_position_size(
            equity_usd=1800,
            price=60000,
            stop_distance_pct=0.01,
            max_risk_per_trade=0.03,
            max_allocation_pct=0.75,
        )
        # allocation cap => 1350 / 60000 = 0.0225 BTC
        self.assertAlmostEqual(size, 0.0225, places=6)

    def test_drawdown_guard_trips_after_limit(self):
        guard = DrawdownGuard(starting_equity=1800, max_drawdown=0.15)
        self.assertTrue(guard.update_and_check(1700))
        self.assertFalse(guard.update_and_check(1500))


if __name__ == "__main__":
    unittest.main()
