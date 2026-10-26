import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from retrieval.open_archives import (
    ArchiveObservation,
    find_pre_cutoff_observation,
    wayback_pre_cutoff_observation,
)


class OpenArchivesTest(unittest.TestCase):
    def test_finds_a_pre_cutoff_observation(self):
        indexes = [
            {
                "id": "CC-MAIN-2025-10",
                "from": "2025-03-01T00:00:00Z",
                "to": "2025-03-10T00:00:00Z",
            }
        ]
        observation = ArchiveObservation(
            url="https://example.org/report",
            timestamp="20250305120000",
            status="200",
            mime="text/html",
            filename="file.warc.gz",
            offset="1",
            length="2",
        )
        with patch(
            "retrieval.open_archives.common_crawl_observations",
            return_value=[observation],
        ):
            match, queried = find_pre_cutoff_observation(
                "https://example.org/report",
                datetime(2025, 3, 1, tzinfo=timezone.utc),
                datetime(2025, 3, 8, tzinfo=timezone.utc),
                indexes=indexes,
            )
        self.assertEqual(match, observation)
        self.assertEqual(queried, ["CC-MAIN-2025-10"])

    def test_accepts_timezone_free_catalogue_dates(self):
        indexes = [
            {
                "id": "CC-MAIN-2025-10",
                "from": "2025-03-01",
                "to": "2025-03-10",
            }
        ]
        observation = ArchiveObservation(
            url="https://example.org/report",
            timestamp="20250305120000",
            status="200",
            mime="text/html",
            filename="file.warc.gz",
            offset="1",
            length="2",
        )
        with patch(
            "retrieval.open_archives.common_crawl_observations",
            return_value=[observation],
        ):
            match, queried = find_pre_cutoff_observation(
                "https://example.org/report",
                datetime(2025, 3, 1, tzinfo=timezone.utc),
                datetime(2025, 3, 8, tzinfo=timezone.utc),
                indexes=indexes,
            )
        self.assertEqual(match, observation)
        self.assertEqual(queried, ["CC-MAIN-2025-10"])

    def test_wayback_capture_must_fall_inside_window(self):
        with patch(
            "retrieval.open_archives._get_json",
            return_value=[
                ["timestamp", "original", "statuscode", "digest"],
                ["20230802204913", "https://example.org/report", "200", "ABC"],
            ],
        ):
            match = wayback_pre_cutoff_observation(
                "https://example.org/report",
                datetime(2023, 8, 2, 12, tzinfo=timezone.utc),
                datetime(2023, 8, 3, tzinfo=timezone.utc),
            )
        self.assertIsNotNone(match)
        self.assertEqual(match.timestamp, "20230802204913")


if __name__ == "__main__":
    unittest.main()
