import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_search_bridge import merge_results


def row(link, title="", snippet=""):
    return {"link": link, "title": title, "snippet": snippet}


class TrustModelTests(unittest.TestCase):
    def test_a_model_url_with_no_source_behind_it_is_dropped(self):
        results = merge_results([], [row("https://invented.example/x", "Fake", "text")], 5)
        self.assertEqual(results, [])

    def test_an_enrichment_fills_in_missing_text(self):
        results = merge_results(
            [row("https://example.com/a")],
            [row("https://example.com/a", "Title", "Snippet")],
            5,
        )
        self.assertEqual(
            results,
            [{"link": "https://example.com/a", "title": "Title", "snippet": "Snippet"}],
        )

    def test_provider_text_wins_over_model_text(self):
        results = merge_results(
            [row("https://example.com/a", "Real title", "Real snippet")],
            [row("https://example.com/a", "Model title", "Model snippet")],
            5,
        )
        self.assertEqual(results[0]["title"], "Real title")
        self.assertEqual(results[0]["snippet"], "Real snippet")

    def test_order_follows_the_sources(self):
        results = merge_results(
            [row("https://example.com/1"), row("https://example.com/2")], [], 5
        )
        self.assertEqual(
            [item["link"] for item in results],
            ["https://example.com/1", "https://example.com/2"],
        )

    def test_duplicate_sources_collapse_to_one_row(self):
        results = merge_results(
            [row("https://example.com/a?utm_source=x"), row("https://example.com/a")], [], 5
        )
        self.assertEqual(len(results), 1)

    def test_scheme_and_www_variants_collapse_to_one_row(self):
        results = merge_results(
            [row("http://www.example.com/a")], [row("https://example.com/a", "Title")], 5
        )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Title")

    def test_truncation_happens_after_merging(self):
        sources = [row("https://example.com/%d" % n) for n in range(5)]
        self.assertEqual(len(merge_results(sources, [], 2)), 2)

    def test_rows_without_a_link_are_dropped(self):
        self.assertEqual(merge_results([row("")], [], 5), [])


if __name__ == "__main__":
    unittest.main()
