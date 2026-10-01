import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_search_bridge import canon_key, clean_url


class CleanUrlTests(unittest.TestCase):
    def test_tracking_parameters_are_removed(self):
        url = "https://example.com/a?utm_source=x&utm_medium=y&gclid=z&fbclid=w&keep=1"
        self.assertEqual(clean_url(url), "https://example.com/a?keep=1")

    def test_repeated_parameters_keep_the_first_value(self):
        self.assertEqual(
            clean_url("https://example.com/a?x=1&x=2"), "https://example.com/a?x=1"
        )

    def test_fragment_is_removed(self):
        self.assertEqual(
            clean_url("https://example.com/a#section"), "https://example.com/a"
        )

    def test_default_ports_are_removed(self):
        self.assertEqual(clean_url("https://example.com:443/a"), "https://example.com/a")
        self.assertEqual(clean_url("http://example.com:80/a"), "http://example.com/a")

    def test_other_ports_are_kept(self):
        self.assertEqual(
            clean_url("https://example.com:8443/a"), "https://example.com:8443/a"
        )

    def test_host_and_scheme_are_lowercased(self):
        self.assertEqual(clean_url("HTTPS://Example.COM/A"), "https://example.com/A")

    def test_an_empty_path_becomes_a_slash(self):
        self.assertEqual(clean_url("https://example.com"), "https://example.com/")

    def test_www_is_kept_for_display_and_dropped_for_the_key(self):
        self.assertEqual(
            clean_url("https://www.example.com/a"), "https://www.example.com/a"
        )
        self.assertEqual(canon_key("https://www.example.com/a"), "https://example.com/a")

    def test_www_does_not_change_the_key(self):
        # Scheme differences are handled when merging, not in the key itself.
        self.assertEqual(
            canon_key("https://www.example.com/a"), canon_key("https://example.com/a")
        )

    def test_malformed_input_is_returned_unchanged(self):
        self.assertEqual(clean_url("not a url"), "not a url")
        self.assertEqual(clean_url(""), "")


if __name__ == "__main__":
    unittest.main()
