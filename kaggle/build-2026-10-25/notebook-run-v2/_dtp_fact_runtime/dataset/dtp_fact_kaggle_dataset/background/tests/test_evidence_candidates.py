import unittest

from retrieval.evidence_candidates import default_evidence_query, gdelt_articles_to_review_rows


class EvidenceCandidatesTest(unittest.TestCase):
    def test_default_query_removes_common_words(self):
        query = default_evidence_query("This image is from the Texas flood in 2025")
        self.assertEqual(query, "image Texas flood 2025")

    def test_excludes_fact_check_domain_and_deduplicates(self):
        articles = [
            {"url": "https://fact.example/check", "title": "Fact check"},
            {"url": "https://news.example/report", "title": "Report"},
            {"url": "https://news.example/report", "title": "Report copy"},
        ]
        rows = gdelt_articles_to_review_rows(
            "claim-1",
            "2025-01-02T00:00:00Z",
            "Texas flood",
            articles,
            excluded_domains={"fact.example"},
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["domain"], "news.example")


if __name__ == "__main__":
    unittest.main()
