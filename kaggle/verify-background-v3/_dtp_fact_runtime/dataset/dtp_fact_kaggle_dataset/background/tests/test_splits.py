import unittest
from dataclasses import replace

from data.splits import assert_no_group_leakage, partition
from data.synthetic import build_synthetic_records


class SplitTest(unittest.TestCase):
    def test_expected_temporal_partitions(self):
        claims, _ = build_synthetic_records()
        splits = partition(claims)
        self.assertEqual({key: len(value) for key, value in splits.items()}, {"train": 2, "validation": 1, "test": 3})
        assert_no_group_leakage(claims)

    def test_detects_event_leakage(self):
        claims, _ = build_synthetic_records()
        leaked = replace(claims[-1], event_id=claims[0].event_id)
        with self.assertRaises(ValueError):
            assert_no_group_leakage([*claims[:-1], leaked])


if __name__ == "__main__":
    unittest.main()

