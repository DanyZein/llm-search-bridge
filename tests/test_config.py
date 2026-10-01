import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llm_search_bridge import ConfigError, load_config

REQUIRED = {
    "BRIDGE_API_KEY": "key",
    "BRIDGE_BASE_URL": "https://provider.example",
    "BRIDGE_MODEL": "model",
}


class ConfigTests(unittest.TestCase):
    def test_every_missing_required_variable_is_reported_at_once(self):
        with self.assertRaises(ConfigError) as caught:
            load_config({})
        message = str(caught.exception)
        self.assertIn("BRIDGE_API_KEY", message)
        self.assertIn("BRIDGE_BASE_URL", message)
        self.assertIn("BRIDGE_MODEL", message)

    def test_defaults(self):
        cfg = load_config(dict(REQUIRED))
        self.assertEqual(cfg.flavor, "anthropic")
        self.assertEqual(cfg.host, "127.0.0.1")
        self.assertEqual(cfg.port, 8899)
        self.assertEqual(cfg.timeout, 180.0)
        self.assertEqual(cfg.max_results, 10)
        self.assertEqual(cfg.web_search_tool, "web_search_20250305")
        self.assertIsNone(cfg.max_uses)
        self.assertEqual(cfg.auth_token, "")

    def test_trailing_slash_is_removed_from_the_base_url(self):
        cfg = load_config(dict(REQUIRED, BRIDGE_BASE_URL="https://provider.example/"))
        self.assertEqual(cfg.base_url, "https://provider.example")

    def test_unknown_flavor_is_rejected_with_the_valid_names(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(dict(REQUIRED, BRIDGE_FLAVOR="gemini"))
        message = str(caught.exception)
        self.assertIn("anthropic", message)
        self.assertIn("openai", message)

    def test_flavor_is_case_insensitive(self):
        cfg = load_config(dict(REQUIRED, BRIDGE_FLAVOR="OpenAI"))
        self.assertEqual(cfg.flavor, "openai")

    def test_numbers_are_validated(self):
        with self.assertRaises(ConfigError) as caught:
            load_config(
                dict(
                    REQUIRED,
                    BRIDGE_PORT="eight",
                    BRIDGE_MAX_RESULTS="0",
                    BRIDGE_TIMEOUT="jiffy",
                )
            )
        message = str(caught.exception)
        self.assertIn("BRIDGE_PORT", message)
        self.assertIn("BRIDGE_MAX_RESULTS", message)
        self.assertIn("BRIDGE_TIMEOUT", message)

    def test_port_above_the_valid_range_is_rejected(self):
        with self.assertRaises(ConfigError):
            load_config(dict(REQUIRED, BRIDGE_PORT="70000"))

    def test_optional_values_are_trimmed(self):
        cfg = load_config(
            dict(
                REQUIRED,
                BRIDGE_AUTH_TOKEN=" s3cret ",
                BRIDGE_MAX_USES="3",
                BRIDGE_HOST="0.0.0.0",
            )
        )
        self.assertEqual(cfg.auth_token, "s3cret")
        self.assertEqual(cfg.max_uses, 3)
        self.assertEqual(cfg.host, "0.0.0.0")

    def test_legacy_variable_names_do_not_satisfy_the_new_ones(self):
        legacy = {
            "DEEPSEEK_API_KEY": "key",
            "DEEPSEEK_BASE_URL": "https://provider.example",
            "DEEPSEEK_MODEL": "model",
        }
        with self.assertRaises(ConfigError):
            load_config(legacy)


if __name__ == "__main__":
    unittest.main()
