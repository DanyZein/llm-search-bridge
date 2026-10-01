import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_search_bridge import PROMPT_TEMPLATE, render_prompt


class PromptTests(unittest.TestCase):
    def test_json_braces_survive_rendering(self):
        # Regression. This template used to be rendered with str.format, which
        # read {"link": ...} as a replacement field and raised KeyError before
        # any request was sent, so every search failed.
        prompt = render_prompt("anything", 5)
        self.assertIn('{"link":', prompt)
        self.assertIn('"snippet"', prompt)

    def test_query_and_count_are_substituted(self):
        prompt = render_prompt("who won the race", 7)
        self.assertIn("Query: who won the race", prompt)
        self.assertIn("at most 7 results", prompt)
        self.assertNotIn("$query", prompt)
        self.assertNotIn("$count", prompt)

    def test_a_query_containing_a_placeholder_is_inserted_verbatim(self):
        # Substituted values are not rescanned, so a query that looks like a
        # placeholder cannot corrupt the rest of the prompt.
        prompt = render_prompt("price of $count items", 3)
        self.assertIn("Query: price of $count items", prompt)

    def test_the_template_has_only_the_two_expected_placeholders(self):
        self.assertIn("$query", PROMPT_TEMPLATE)
        self.assertIn("$count", PROMPT_TEMPLATE)
        self.assertNotIn("$link", PROMPT_TEMPLATE)


if __name__ == "__main__":
    unittest.main()
