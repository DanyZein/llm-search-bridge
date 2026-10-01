import io
import json
import os
import socket
import sys
import unittest
import urllib.error
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_search_bridge as bridge
from llm_search_bridge import Config, UpstreamError, http_post, run_search

FIXTURES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")


def fixture(name):
    with open(os.path.join(FIXTURES, name), encoding="utf-8") as handle:
        return json.load(handle)


def make_config(**overrides):
    values = dict(
        flavor="anthropic",
        api_key="test-key",
        base_url="https://provider.example",
        model="test-model",
        timeout=5.0,
    )
    values.update(overrides)
    return Config(**values)


class AnthropicFlavorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config()

    def test_request_targets_the_messages_api(self):
        url, headers, payload = bridge.anthropic_build_request(self.cfg, "prompt", 5)
        self.assertEqual(url, "https://provider.example/v1/messages")
        self.assertEqual(headers["x-api-key"], "test-key")
        self.assertEqual(headers["anthropic-version"], "2023-06-01")
        self.assertEqual(payload["model"], "test-model")
        self.assertEqual(payload["tools"][0]["name"], "web_search")
        self.assertEqual(payload["tools"][0]["type"], "web_search_20250305")
        self.assertNotIn("max_uses", payload["tools"][0])

    def test_max_uses_is_sent_only_when_configured(self):
        _, _, payload = bridge.anthropic_build_request(
            make_config(max_uses=3), "prompt", 5
        )
        self.assertEqual(payload["tools"][0]["max_uses"], 3)

    def test_sources_come_from_the_tool_result_blocks(self):
        parsed = bridge.anthropic_parse_response(self.cfg, fixture("anthropic_ok.json"))
        self.assertEqual(
            [item["link"] for item in parsed.sources],
            [
                "https://example.com/a?utm_source=newsletter&gclid=abc123",
                "https://www.example.org/b",
            ],
        )
        self.assertEqual(parsed.sources[0]["title"], "Page A")
        self.assertEqual(parsed.warnings, [])

    def test_the_model_answer_becomes_enrichment(self):
        parsed = bridge.anthropic_parse_response(self.cfg, fixture("anthropic_ok.json"))
        self.assertEqual(len(parsed.enrichments), 3)
        self.assertTrue(
            any("invented.example" in item["link"] for item in parsed.enrichments)
        )

    def test_a_response_without_tool_blocks_warns(self):
        parsed = bridge.anthropic_parse_response(
            self.cfg, fixture("anthropic_no_results.json")
        )
        self.assertEqual(parsed.sources, [])
        self.assertTrue(
            any("web_search_tool_result" in note for note in parsed.warnings)
        )

    def test_a_failed_search_reports_the_error_code(self):
        parsed = bridge.anthropic_parse_response(
            self.cfg, fixture("anthropic_tool_error.json")
        )
        self.assertEqual(parsed.sources, [])
        self.assertTrue(
            any("max_uses_exceeded" in note for note in parsed.warnings)
        )

    def test_error_detail_reads_the_nested_message(self):
        raw = json.dumps(
            {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}}
        ).encode()
        self.assertEqual(
            bridge.anthropic_error_detail(raw), "rate_limit_error: slow down"
        )

    def test_error_detail_falls_back_to_the_raw_body(self):
        self.assertEqual(
            bridge.anthropic_error_detail(b"<html>gateway error</html>"),
            "<html>gateway error</html>",
        )


class OpenAIFlavorTests(unittest.TestCase):
    def setUp(self):
        self.cfg = make_config(flavor="openai")

    def test_request_targets_the_responses_api(self):
        url, headers, payload = bridge.openai_build_request(self.cfg, "prompt", 5)
        self.assertEqual(url, "https://provider.example/v1/responses")
        self.assertEqual(headers["authorization"], "Bearer test-key")
        self.assertEqual(payload["tools"], [{"type": "web_search"}])
        self.assertEqual(payload["include"], ["web_search_call.results"])
        self.assertEqual(payload["input"], "prompt")

    def test_raw_results_become_sources_with_their_snippets(self):
        parsed = bridge.openai_parse_response(self.cfg, fixture("openai_ok.json"))
        links = {item["link"] for item in parsed.sources}
        self.assertEqual(links, {"https://example.com/a", "https://example.net/c"})
        # Only the raw result rows carry snippets, and both of them do. The
        # duplicate rows that action.sources contributes stay bare.
        with_snippets = [item for item in parsed.sources if item["snippet"]]
        self.assertEqual(len(with_snippets), 2)
        self.assertEqual(
            sorted(item["snippet"] for item in with_snippets),
            ["Provider snippet for A.", "Provider snippet for C."],
        )
        self.assertEqual(parsed.warnings, [])

    def test_parsing_does_not_depend_on_the_position_of_items(self):
        # The fixture lists message, then reasoning, then web_search_call.
        body = fixture("openai_ok.json")
        self.assertEqual(body["output"][0]["type"], "message")
        parsed = bridge.openai_parse_response(self.cfg, body)
        self.assertTrue(parsed.sources)

    def test_citations_are_enrichment_only(self):
        parsed = bridge.openai_parse_response(self.cfg, fixture("openai_ok.json"))
        self.assertTrue(
            any("invented.example/z" in item["link"] for item in parsed.enrichments)
        )
        self.assertFalse(
            any("invented.example" in item["link"] for item in parsed.sources)
        )

    def test_citations_without_a_search_call_produce_no_sources(self):
        parsed = bridge.openai_parse_response(
            self.cfg, fixture("openai_annotations_only.json")
        )
        self.assertEqual(parsed.sources, [])
        self.assertTrue(any("web_search_call" in note for note in parsed.warnings))

    def test_missing_raw_results_warns_about_the_include_parameter(self):
        parsed = bridge.openai_parse_response(self.cfg, fixture("openai_no_include.json"))
        self.assertEqual(parsed.sources, [])
        self.assertTrue(any("include=" in note for note in parsed.warnings))

    def test_error_detail_reads_code_and_message(self):
        raw = json.dumps(
            {"error": {"message": "bad key", "code": "invalid_api_key"}}
        ).encode()
        self.assertEqual(bridge.openai_error_detail(raw), "invalid_api_key: bad key")


class PipelineTests(unittest.TestCase):
    def test_the_whole_pipeline_runs_from_one_injected_fetch(self):
        calls = []

        def fake_fetch(url, headers, payload, timeout, error_detail):
            calls.append({"url": url, "headers": headers, "payload": payload})
            return fixture("anthropic_ok.json")

        results, warnings = run_search(make_config(), "example query", 5, fetch=fake_fetch)

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["url"], "https://provider.example/v1/messages")
        self.assertIn("example query", calls[0]["payload"]["messages"][0]["content"])
        self.assertEqual(warnings, [])
        self.assertEqual([item["link"] for item in results],
                         ["https://example.com/a", "https://www.example.org/b"])
        self.assertEqual(results[0]["snippet"], "About A.")

    def test_the_openai_pipeline_keeps_provider_snippets(self):
        results, _ = run_search(
            make_config(flavor="openai"),
            "example query",
            5,
            fetch=lambda *args: fixture("openai_ok.json"),
        )
        self.assertEqual([item["link"] for item in results],
                         ["https://example.com/a", "https://example.net/c"])
        self.assertEqual(results[0]["snippet"], "Provider snippet for A.")

    def test_model_text_alone_never_produces_results(self):
        body = {
            "content": [
                {
                    "type": "text",
                    "text": '[{"link": "https://invented.example/x", "title": "T", "snippet": "S"}]',
                }
            ]
        }
        results, warnings = run_search(
            make_config(), "query", 5, fetch=lambda *args: body
        )
        self.assertEqual(results, [])
        self.assertTrue(warnings)

    def test_no_provider_results_returns_an_empty_list(self):
        results, warnings = run_search(
            make_config(),
            "query",
            5,
            fetch=lambda *args: fixture("anthropic_no_results.json"),
        )
        self.assertEqual(results, [])
        self.assertTrue(warnings)


class TransportTests(unittest.TestCase):
    def _http_error(self, code, payload):
        return urllib.error.HTTPError(
            "https://provider.example/v1/messages",
            code,
            "error",
            {},
            io.BytesIO(json.dumps(payload).encode()),
        )

    def test_an_http_error_carries_the_flavor_specific_detail(self):
        error = self._http_error(
            429,
            {"type": "error", "error": {"type": "rate_limit_error", "message": "slow down"}},
        )
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaises(UpstreamError) as caught:
                http_post(
                    "https://provider.example/v1/messages",
                    {},
                    {},
                    5.0,
                    bridge.anthropic_error_detail,
                )
        self.assertEqual(caught.exception.status, 429)
        self.assertEqual(caught.exception.detail, "rate_limit_error: slow down")

    def test_a_timeout_is_reported_as_a_timeout(self):
        with mock.patch("urllib.request.urlopen", side_effect=socket.timeout("late")):
            with self.assertRaises(UpstreamError) as caught:
                http_post("https://provider.example", {}, {}, 5.0, bridge.anthropic_error_detail)
        self.assertIn("did not answer", str(caught.exception))

    def test_a_non_json_body_is_rejected(self):
        class Answer:
            def read(self):
                return b"<html>gateway</html>"

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        with mock.patch("urllib.request.urlopen", return_value=Answer()):
            with self.assertRaises(UpstreamError) as caught:
                http_post("https://provider.example", {}, {}, 5.0, bridge.anthropic_error_detail)
        self.assertIn("not JSON", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
