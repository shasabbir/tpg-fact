import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from retrieval.open_archives import ArchiveObservation, find_pre_cutoff_observation


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


if __name__ == "__main__":
    unittest.main()
