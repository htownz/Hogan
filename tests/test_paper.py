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


if __name__ == "__main__":
    unittest.main()
