import unittest

from hogan_bot.trade_explainer import _enforce_two_sentences


class TradeExplainerTests(unittest.TestCase):
    def test_enforce_two_sentences_truncates_extra(self):
        text = "One. Two! Three?"
        out = _enforce_two_sentences(text)
        self.assertEqual(out, "One. Two!")

    def test_enforce_two_sentences_expands_single(self):
        text = "Only one sentence"
        out = _enforce_two_sentences(text)
        self.assertIn("Only one sentence.", out)
        self.assertEqual(out.count("."), 2)


if __name__ == "__main__":
    unittest.main()
