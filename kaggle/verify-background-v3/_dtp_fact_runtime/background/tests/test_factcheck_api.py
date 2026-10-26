import unittest
from io import BytesIO
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlparse
from unittest.mock import patch

from data.factcheck_api import FactCheckApiClient, FactCheckApiError, candidates_from_pages


class FactCheckApiTest(unittest.TestCase):
    def test_paginates_and_flattens_reviews_without_images(self):
        calls = []

        def fake_fetch(url):
            calls.append(url)
            query = parse_qs(urlparse(url).query)
            if "pageToken" not in query:
                return {
                    "claims": [
                        {
                            "text": "An image shows an event in Texas",
                            "claimant": "Example account",
                            "claimDate": "2025-03-01T00:00:00Z",
                            "claimReview": [
                                {
                                    "publisher": {"name": "Example", "site": "example.org"},
                                    "url": "https://example.org/check/1",
                                    "title": "Example check",
                                    "reviewDate": "2025-03-03T00:00:00Z",
                                    "textualRating": "False",
                                    "languageCode": "en",
                                }
                            ],
                        }
                    ],
                    "nextPageToken": "second",
                }
            return {"claims": []}

        client = FactCheckApiClient("private-key", fetch_json=fake_fetch)
        pages = client.search_pages(publisher_site="example.org")
        candidates = candidates_from_pages(pages)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["publisher_site"], "example.org")
        self.assertNotIn("claim_image_url", candidates[0])

    def test_filters_out_of_range_year(self):
        pages = [
            {
                "claims": [
                    {
                        "text": "Old claim",
                        "claimDate": "2019-01-01T00:00:00Z",
                        "claimReview": [
                            {
                                "url": "https://example.org/old",
                                "reviewDate": "2019-01-02T00:00:00Z",
                            }
                        ],
                    }
                ]
            }
        ]
        self.assertEqual(candidates_from_pages(pages), [])

    def test_http_error_is_informative_and_redacts_key(self):
        key = "private-test-key"
        body = BytesIO(
            b'{"error":{"message":"API disabled for private-test-key"}}'
        )
        error = HTTPError(
            "https://example.invalid",
            403,
            "Forbidden",
            hdrs=None,
            fp=body,
        )
        client = FactCheckApiClient(key, retries=0)
        with patch("data.factcheck_api.urlopen", side_effect=error):
            with self.assertRaises(FactCheckApiError) as context:
                client.search_pages(publisher_site="example.org", max_pages=1)
        self.assertIn("API disabled", str(context.exception))
        self.assertNotIn(key, str(context.exception))
        self.assertIn("[REDACTED]", str(context.exception))


if __name__ == "__main__":
    unittest.main()
