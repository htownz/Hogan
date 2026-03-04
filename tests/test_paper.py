import unittest

from hogan_bot.paper import PaperPortfolio


class PaperTests(unittest.TestCase):
    def test_buy_then_sell_updates_cash(self):
        p = PaperPortfolio(cash_usd=1000, fee_rate=0.001)
        self.assertTrue(p.execute_buy("BTC/USD", price=100, qty=2))
        self.assertAlmostEqual(p.cash_usd, 799.8, places=5)

        self.assertTrue(p.execute_sell("BTC/USD", price=110, qty=1))
        self.assertAlmostEqual(p.cash_usd, 909.69, places=5)

    def test_cannot_oversell(self):
        p = PaperPortfolio(cash_usd=1000, fee_rate=0.001)
        p.execute_buy("ETH/USD", price=50, qty=1)
        self.assertFalse(p.execute_sell("ETH/USD", price=50, qty=2))


class TrailingStopTests(unittest.TestCase):
    def _portfolio(self):
        return PaperPortfolio(cash_usd=10_000, fee_rate=0.0)

    def test_no_exit_when_price_above_stop(self):
        p = self._portfolio()
        p.execute_buy("BTC/USD", price=100, qty=1, trailing_stop_pct=0.05)
        exits = p.check_exits({"BTC/USD": 102.0})  # still above 95 stop
        self.assertEqual(exits, [])

    def test_trailing_stop_fires_when_price_drops(self):
        p = self._portfolio()
        p.execute_buy("BTC/USD", price=100, qty=1, trailing_stop_pct=0.05)
        # Price rises to 110 → peak_price = 110; stop = 110 * 0.95 = 104.5
        p.check_exits({"BTC/USD": 110.0})
        # Price drops to 104 — below 104.5 stop
        exits = p.check_exits({"BTC/USD": 104.0})
        self.assertEqual(len(exits), 1)
        self.assertEqual(exits[0], ("BTC/USD", "trailing_stop"))

    def test_peak_price_tracks_high_watermark(self):
        p = self._portfolio()
        p.execute_buy("BTC/USD", price=100, qty=1, trailing_stop_pct=0.10)
        p.check_exits({"BTC/USD": 120.0})  # peak rises to 120
        p.check_exits({"BTC/USD": 115.0})  # price still above stop (108)
        exits = p.check_exits({"BTC/USD": 107.0})  # below stop (108)
        self.assertEqual(exits[0][1], "trailing_stop")

    def test_no_trailing_stop_when_pct_zero(self):
        p = self._portfolio()
        p.execute_buy("BTC/USD", price=100, qty=1, trailing_stop_pct=0.0)
        exits = p.check_exits({"BTC/USD": 10.0})  # price crashed
        self.assertEqual(exits, [])

    def test_take_profit_fires_at_target(self):
        p = self._portfolio()
        p.execute_buy("BTC/USD", price=100, qty=1, take_profit_pct=0.10)
        exits_before = p.check_exits({"BTC/USD": 105.0})  # below 110
        exits_at = p.check_exits({"BTC/USD": 110.0})       # exactly at 110
        self.assertEqual(exits_before, [])
        self.assertEqual(exits_at[0], ("BTC/USD", "take_profit"))

    def test_check_exits_does_not_auto_close(self):
        """check_exits() signals but does NOT sell; caller must call execute_sell."""
        p = self._portfolio()
        p.execute_buy("BTC/USD", price=100, qty=1, trailing_stop_pct=0.05)
        p.check_exits({"BTC/USD": 120.0})  # pump peak
        p.check_exits({"BTC/USD": 50.0})   # trigger stop
        self.assertIn("BTC/USD", p.positions)  # still open

    def test_total_equity_after_exit(self):
        p = self._portfolio()
        p.execute_buy("BTC/USD", price=100, qty=1, trailing_stop_pct=0.05)
        p.check_exits({"BTC/USD": 120.0})
        exits = p.check_exits({"BTC/USD": 100.0})  # below 114 stop
        for sym, _ in exits:
            p.execute_sell(sym, 100.0, p.positions[sym].qty)
        self.assertNotIn("BTC/USD", p.positions)
        self.assertAlmostEqual(p.cash_usd, 10_000.0, places=2)  # fee_rate=0


if __name__ == "__main__":
    unittest.main()
