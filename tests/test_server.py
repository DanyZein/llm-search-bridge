import json
import os
import sys
import threading
import unittest
import urllib.request
from http.server import ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import llm_search_bridge as bridge
from llm_search_bridge import Config, UpstreamError, handle_search


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


def request_body(payload):
    return json.dumps(payload).encode()


def fake_search(cfg, query, count):
    return [{"link": "https://example.com/a", "title": "A", "snippet": "S"}], []


class HandleSearchTests(unittest.TestCase):
    def test_a_token_is_required_when_configured(self):
        cfg = make_config(auth_token="secret")
        for header in ("", "Bearer wrong", "secret", "Basic secret"):
            status, _ = handle_search(cfg, header, request_body({"query": "x"}), fake_search)
            self.assertEqual(status, 401, "header %r should not be accepted" % header)

    def test_the_right_token_is_accepted(self):
        cfg = make_config(auth_token="secret")
        status, payload = handle_search(cfg, "Bearer secret", request_body({"query": "x"}), fake_search)
        self.assertEqual(status, 200)
        self.assertEqual(payload[0]["link"], "https://example.com/a")

    def test_no_token_configured_means_no_auth(self):
        status, _ = handle_search(make_config(), "", request_body({"query": "x"}), fake_search)
        self.assertEqual(status, 200)

    def test_invalid_json_is_a_400(self):
        status, payload = handle_search(make_config(), "", b"{not json", fake_search)
        self.assertEqual(status, 400)
        self.assertIn("error", payload)

    def test_a_body_that_is_not_an_object_is_a_400(self):
        status, _ = handle_search(make_config(), "", b"[1, 2]", fake_search)
        self.assertEqual(status, 400)

    def test_an_empty_query_returns_an_empty_list(self):
        status, payload = handle_search(make_config(), "", request_body({"query": "   "}), fake_search)
        self.assertEqual((status, payload), (200, []))

    def test_count_is_clamped_to_the_configured_ceiling(self):
        seen = {}

        def recording_search(cfg, query, count):
            seen["count"] = count
            return [], []

        cfg = make_config(max_results=3)
        handle_search(cfg, "", request_body({"query": "x", "count": 50}), recording_search)
        self.assertEqual(seen["count"], 3)
        handle_search(cfg, "", request_body({"query": "x"}), recording_search)
        self.assertEqual(seen["count"], 3)
        handle_search(cfg, "", request_body({"query": "x", "count": "many"}), recording_search)
        self.assertEqual(seen["count"], 3)

    def test_an_upstream_error_becomes_a_502(self):
        def failing_search(cfg, query, count):
            raise UpstreamError(
                "provider returned HTTP 429",
                detail="rate_limit_error: slow down",
                status=429,
            )

        status, payload = handle_search(make_config(), "", request_body({"query": "x"}), failing_search)
        self.assertEqual(status, 502)
        self.assertEqual(payload["error"], "provider returned HTTP 429")
        self.assertIn("slow down", payload["detail"])

    def test_the_api_key_is_scrubbed_from_error_details(self):
        def failing_search(cfg, query, count):
            raise UpstreamError("boom", detail="the key test-key was rejected")

        status, payload = handle_search(make_config(), "", request_body({"query": "x"}), failing_search)
        self.assertNotIn("test-key", payload["detail"])

    def test_results_are_returned_as_a_bare_json_array(self):
        status, payload = handle_search(make_config(), "", request_body({"query": "x"}), fake_search)
        self.assertEqual(status, 200)
        self.assertIsInstance(payload, list)
        self.assertEqual(json.loads(json.dumps(payload)), payload)


class ServerSmokeTest(unittest.TestCase):
    """One request over a real loopback socket, with the search injected."""

    def test_health_check_and_search(self):
        class TestHandler(bridge.Handler):
            config = make_config()
            search = staticmethod(fake_search)

        server = ThreadingHTTPServer(("127.0.0.1", 0), TestHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            port = server.server_address[1]

            with urllib.request.urlopen("http://127.0.0.1:%d/" % port, timeout=5) as answer:
                health = json.loads(answer.read())
            self.assertTrue(health["ok"])
            self.assertEqual(health["flavor"], "anthropic")

            request = urllib.request.Request(
                "http://127.0.0.1:%d/search" % port,
                data=request_body({"query": "anything", "count": 2}),
                headers={"content-type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=5) as answer:
                results = json.loads(answer.read())
            self.assertEqual(results[0]["link"], "https://example.com/a")
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
