import unittest

from data.labels import is_explicitly_excluded, normalize_rating


class LabelPolicyTest(unittest.TestCase):
    def test_maps_only_explicit_ratings(self):
        self.assertEqual(normalize_rating("TRUE"), "supported")
        self.assertEqual(normalize_rating("Miscaptioned"), "refuted")
        self.assertEqual(normalize_rating("No evidence"), "insufficient_evidence")

    def test_rejects_ambiguous_ratings(self):
        self.assertIsNone(normalize_rating("Mostly true"))
        self.assertTrue(is_explicitly_excluded("Half-true"))


if __name__ == "__main__":
    unittest.main()

