#!/usr/bin/env python3
"""
llm-search-bridge: use an LLM provider's native web search as a plain search
API, so applications like Open WebUI can use it as their search engine.

The bridge takes a search request, asks the provider's own web search tool to
run the query, and returns the results the provider actually retrieved. Links
that the model wrote itself are never returned, so a caller cannot be handed a
hallucinated citation. See merge_results() and the README.

HTTP contract:
    POST /search   {"query": "...", "count": 5}
        -> 200  [{"link": "...", "title": "...", "snippet": "..."}]
    GET  /         -> 200  {"ok": true, ...}   (health check)

The path of a POST is not checked, so any URL path can be configured in the
client.

Configuration comes from the environment. Run with --check to print the
resolved settings and exit. .env.example has ready-made provider presets.

Environment variables:
    BRIDGE_FLAVOR        anthropic | openai            (default: anthropic)
    BRIDGE_API_KEY       provider key                  (required)
    BRIDGE_BASE_URL      e.g. https://api.deepseek.com/anthropic
    BRIDGE_MODEL         e.g. deepseek-flash           (required)
    BRIDGE_AUTH_TOKEN    shared secret clients must send (default: no auth)
    BRIDGE_HOST          bind address                  (default: 127.0.0.1)
    BRIDGE_PORT          bind port                     (default: 8899)
    BRIDGE_TIMEOUT       upstream timeout, seconds     (default: 180)
    BRIDGE_MAX_RESULTS   ceiling applied to "count"    (default: 10)
    BRIDGE_WEB_SEARCH_TOOL  anthropic flavor only      (default below)
    BRIDGE_MAX_USES      anthropic flavor only, caps searches per request
"""

import hmac
import json
import os
import re
import socket
import string
import sys
import time
import traceback
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable, Dict, List, Mapping, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

__version__ = "1.0.0"

USER_AGENT = "llm-search-bridge/" + __version__

# Anthropic-format headers are versioned, and this one has been stable since
# the Messages API launched.
ANTHROPIC_VERSION = "2023-06-01"

# The Messages API requires an explicit budget even though we only want a
# short JSON answer.
ANTHROPIC_MAX_TOKENS = 3000

# Query parameters that never change which page you land on. Removed when
# cleaning a URL and when computing the key used to detect duplicates.
TRACKING_EXACT = {"gclid", "fbclid", "msclkid", "igshid", "mc_cid", "mc_eid", "ref_src"}

# string.Template placeholders ($query, $count), not str.format: the text
# below contains literal JSON braces, which str.format would try to read as
# replacement fields and fail on.
PROMPT_TEMPLATE = """You are the web search back end for another application.
Call the web_search tool to search the web for the query at the end of this
message. Always search, even if you think you already know the answer.

Then reply with one JSON array and nothing else. No prose, no code fences.
Each element looks like this:

{"link": "the exact result URL", "title": "the page title", "snippet": "one or two sentences about the page"}

Rules:
- Include "snippet" for every element. Never leave it empty.
- Write snippets in the language of the query, and prefer sources in that
  language.
- Use URLs and titles exactly as they appear in the search results. Do not
  invent, edit, or extend them.
- Only use URLs that the search returned. If a URL is not in the results, do
  not include it.
- For questions about current prices, rates, scores or schedules, prefer the
  most recently dated pages and avoid archive pages.
- Return at most $count results, best first. If there are no results, return [].

Query: $query"""


class ConfigError(Exception):
    """The environment does not describe a usable configuration."""


class UpstreamError(Exception):
    """The provider, or the connection to it, failed."""

    def __init__(self, message: str, detail: str = "", status: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.detail = detail
        self.status = status


@dataclass(frozen=True)
class Config:
    """Everything the bridge needs to talk to one provider."""

    flavor: str
    api_key: str
    base_url: str
    model: str
    auth_token: str = ""
    host: str = "127.0.0.1"
    port: int = 8899
    timeout: float = 180.0
    max_results: int = 10
    web_search_tool: str = "web_search_20250305"
    max_uses: Optional[int] = None


def _int_from_env(
    env: Mapping[str, str],
    name: str,
    default: Optional[int],
    minimum: int,
    problems: List[str],
) -> Optional[int]:
    """Read a whole number, recording the problem instead of raising."""
    raw = (env.get(name) or "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        problems.append("%s must be a whole number (got %r)" % (name, raw))
        return default
    if value < minimum:
        problems.append("%s must be at least %d (got %d)" % (name, minimum, value))
        return default
    return value


def load_config(env: Optional[Mapping[str, str]] = None) -> Config:
    """Build a Config from the environment.

    Reports every problem at once so a failed start does not turn into one
    restart per typo.
    """
    env = os.environ if env is None else env
    problems: List[str] = []

    flavor = (env.get("BRIDGE_FLAVOR") or "anthropic").strip().lower()
    if flavor not in FLAVORS:
        problems.append(
            "BRIDGE_FLAVOR must be one of %s (got %r)"
            % (", ".join(sorted(FLAVORS)), flavor)
        )

    api_key = (env.get("BRIDGE_API_KEY") or "").strip()
    if not api_key:
        problems.append("BRIDGE_API_KEY is required")

    base_url = (env.get("BRIDGE_BASE_URL") or "").strip().rstrip("/")
    if not base_url:
        problems.append("BRIDGE_BASE_URL is required")

    model = (env.get("BRIDGE_MODEL") or "").strip()
    if not model:
        problems.append("BRIDGE_MODEL is required")

    timeout = _int_from_env(env, "BRIDGE_TIMEOUT", 180, 1, problems)
    port = _int_from_env(env, "BRIDGE_PORT", 8899, 1, problems)
    max_results = _int_from_env(env, "BRIDGE_MAX_RESULTS", 10, 1, problems)
    max_uses = _int_from_env(env, "BRIDGE_MAX_USES", None, 1, problems)

    if port and port > 65535:
        problems.append("BRIDGE_PORT must be 65535 or less (got %d)" % port)

    if problems:
        raise ConfigError("; ".join(problems))

    return Config(
        flavor=flavor,
        api_key=api_key,
        base_url=base_url,
        model=model,
        auth_token=(env.get("BRIDGE_AUTH_TOKEN") or "").strip(),
        host=(env.get("BRIDGE_HOST") or "127.0.0.1").strip(),
        port=port,
        timeout=float(timeout),
        max_results=max_results,
        web_search_tool=(
            env.get("BRIDGE_WEB_SEARCH_TOOL") or "web_search_20250305"
        ).strip(),
        max_uses=max_uses,
    )


def _clean_url(url: str, drop_www: bool) -> str:
    """Normalize a URL: lowercase scheme and host, drop tracking parameters
    and the fragment, and drop a default port. Malformed input is returned
    unchanged rather than dropped, so the caller decides what to do with it.
    """
    url = (url or "").strip()
    try:
        parts = urlsplit(url)
    except ValueError:
        return url
    if not parts.netloc:
        return url

    scheme = parts.scheme.lower() or "https"
    netloc = parts.netloc.lower()
    if netloc.endswith(":443") and scheme == "https":
        netloc = netloc[: -len(":443")]
    if netloc.endswith(":80") and scheme == "http":
        netloc = netloc[: -len(":80")]
    if drop_www and netloc.startswith("www."):
        netloc = netloc[4:]

    kept, seen = [], set()
    for key, value in parse_qsl(parts.query, keep_blank_values=True):
        lowered = key.lower()
        if lowered.startswith("utm_") or lowered in TRACKING_EXACT or lowered in seen:
            continue
        seen.add(lowered)
        kept.append((key, value))

    return urlunsplit((scheme, netloc, parts.path or "/", urlencode(kept), ""))


def clean_url(url: str) -> str:
    """The URL as it will be returned to the caller."""
    return _clean_url(url, drop_www=False)


def canon_key(url: str) -> str:
    """The key used to decide that two URLs point at the same page."""
    return _clean_url(url, drop_www=True)


def _no_scheme(key: str) -> str:
    return key.split("://", 1)[1] if "://" in key else key


def merge_results(
    sources: List[dict], enrichments: List[dict], count: int
) -> List[dict]:
    """Combine provider results and model text into the returned list.

    Trust model: only rows from `sources` can appear in the output. Rows from
    `enrichments` may fill in a missing title or snippet on a URL that a
    source already contributed, and are dropped otherwise. This is what keeps
    a model from inventing a link, and it is why provider-supplied text wins
    over model text when both describe the same page.

    `sources` also sets the order, which is the provider's ranking, not the
    model's.
    """
    order: List[str] = []
    meta: Dict[str, dict] = {}
    index: Dict[str, str] = {}

    def resolve(key: str) -> Optional[str]:
        if key in index:
            return index[key]
        return index.get(_no_scheme(key))

    def add(link, title, snippet, trusted: bool) -> None:
        key = canon_key(link)
        if not key:
            return
        existing = resolve(key)
        if existing is not None:
            row = meta[existing]
            if not row["title"] and title:
                row["title"] = title
            if not row["snippet"] and snippet:
                row["snippet"] = snippet
            return
        if not trusted:
            return  # a model URL that no provider result backs
        meta[key] = {
            "link": clean_url(link),
            "title": title or "",
            "snippet": snippet or "",
        }
        order.append(key)
        index[key] = key
        index[_no_scheme(key)] = key

    for row in sources:
        add(row.get("link"), row.get("title"), row.get("snippet"), trusted=True)
    for row in enrichments:
        add(row.get("link"), row.get("title"), row.get("snippet"), trusted=False)

    return [meta[key] for key in order][:count]


@dataclass
class FlavorResponse:
    """What a flavor parser hands back to the shared pipeline.

    sources: provider-attested search results. The only rows that can reach
        the output.
    enrichments: anything the model authored (answer JSON, citation
        annotations). Can fill an empty title or snippet on a source, never
        add a row.
    warnings: operator-facing notes about degraded responses, logged to
        stderr.
    """

    sources: List[dict] = field(default_factory=list)
    enrichments: List[dict] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class Flavor:
    """One provider wire format.

    build_request(cfg, prompt, count) -> (url, headers, payload)
    parse_response(cfg, body)         -> FlavorResponse
    error_detail(raw_error_body)      -> a single line for the logs

    All three are pure functions, so a flavor can be tested from a recorded
    response with no network and no key.
    """

    name: str
    build_request: Callable
    parse_response: Callable
    error_detail: Callable


def render_prompt(query: str, count: int) -> str:
    return string.Template(PROMPT_TEMPLATE).substitute(query=query, count=count)


def parse_json_array(text: str) -> List[dict]:
    """Pull a JSON array of result objects out of a model answer.

    Models wrap the array in code fences or add a sentence around it often
    enough that a bare json.loads() is not enough. Returns [] when nothing
    usable is found.
    """
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("[")
    if start == -1:
        return []
    end = text.rfind("]")
    chunk = text[start : end + 1] if end > start else text[start:]
    try:
        data = json.loads(chunk)
    except ValueError:
        return []
    if isinstance(data, dict):
        data = data.get("results", [])
    if not isinstance(data, list):
        return []
    items = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        link = entry.get("link") or entry.get("url")
        if not link:
            continue
        items.append(
            {
                "link": str(link),
                "title": str(entry.get("title") or ""),
                "snippet": str(entry.get("snippet") or ""),
            }
        )
    return items


def _decode(raw) -> str:
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", "replace")
    return raw or ""


def _one_line(text) -> str:
    """Collapse whitespace so a provider error body cannot flood the logs."""
    return " ".join(str(text).split())[:800]


def _error_message_from(body_text: str) -> Optional[str]:
    """Both wire formats nest the human-readable message under error.message."""
    try:
        parsed = json.loads(body_text)
    except ValueError:
        return None
    error = parsed.get("error") if isinstance(parsed, dict) else None
    if not isinstance(error, dict) or not error.get("message"):
        return None
    kind = error.get("type") or error.get("code")
    message = str(error["message"])
    return "%s: %s" % (kind, message) if kind else message


# ---------------------------------------------------------------------------
# anthropic flavor: the Messages API with a server-side web_search tool.
# Anthropic, DeepSeek's /anthropic endpoint, and other compatible gateways.
# ---------------------------------------------------------------------------


def anthropic_build_request(cfg: Config, prompt: str, count: int):
    tool = {"type": cfg.web_search_tool, "name": "web_search"}
    if cfg.max_uses is not None:
        tool["max_uses"] = cfg.max_uses
    payload = {
        "model": cfg.model,
        "max_tokens": ANTHROPIC_MAX_TOKENS,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [tool],
    }
    headers = {
        "content-type": "application/json",
        "x-api-key": cfg.api_key,
        "anthropic-version": ANTHROPIC_VERSION,
    }
    return cfg.base_url + "/v1/messages", headers, payload


def anthropic_parse_response(cfg: Config, body: dict) -> FlavorResponse:
    response = FlavorResponse()
    for block in body.get("content") or []:
        if block.get("type") != "web_search_tool_result":
            continue
        content = block.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "web_search_result" and item.get("url"):
                    response.sources.append(
                        {
                            "link": item["url"],
                            "title": item.get("title") or "",
                            "snippet": "",
                        }
                    )
        elif isinstance(content, dict):
            # A failed search is reported in the same block position, with an
            # error object instead of a result list.
            code = content.get("error_code") or content.get("type") or "unknown"
            response.warnings.append("provider reported a failed web search (%s)" % code)

    text = "".join(
        block.get("text", "")
        for block in (body.get("content") or [])
        if block.get("type") == "text"
    )
    response.enrichments = parse_json_array(text)

    if not response.sources and not response.warnings:
        response.warnings.append("no web_search_tool_result blocks in the response")
    elif response.sources and not response.enrichments:
        response.warnings.append("model answer had no JSON array; snippets will be empty")
    return response


def anthropic_error_detail(raw) -> str:
    body = _decode(raw)
    return _one_line(_error_message_from(body) or body)


# ---------------------------------------------------------------------------
# openai flavor: the Responses API with the server-side web_search tool.
# The include parameter is what makes the raw result set available, and that
# raw result set is the only trusted source for this flavor.
# ---------------------------------------------------------------------------


def openai_build_request(cfg: Config, prompt: str, count: int):
    payload = {
        "model": cfg.model,
        "input": prompt,
        "tools": [{"type": "web_search"}],
        "include": ["web_search_call.results"],
    }
    headers = {
        "content-type": "application/json",
        "authorization": "Bearer " + cfg.api_key,
    }
    return cfg.base_url + "/v1/responses", headers, payload


def _openai_raw_results(item: dict) -> List[dict]:
    """The retrieve list lives at item["results"], but some gateways nest it
    under action. Check both instead of assuming."""
    for candidate in (item.get("results"), (item.get("action") or {}).get("results")):
        if isinstance(candidate, list):
            return [row for row in candidate if isinstance(row, dict)]
    return []


def openai_parse_response(cfg: Config, body: dict) -> FlavorResponse:
    response = FlavorResponse()
    text_parts: List[str] = []
    saw_search_call = False

    # Parse by item type. The output array also carries reasoning items, and
    # their position is not fixed.
    for item in body.get("output") or []:
        if not isinstance(item, dict):
            continue
        kind = item.get("type")

        if kind == "web_search_call":
            saw_search_call = True
            for row in _openai_raw_results(item):
                if row.get("url"):
                    response.sources.append(
                        {
                            "link": row["url"],
                            "title": row.get("title") or "",
                            "snippet": row.get("snippet") or "",
                        }
                    )
            for source in (item.get("action") or {}).get("sources") or []:
                if isinstance(source, dict) and source.get("url"):
                    response.sources.append(
                        {
                            "link": source["url"],
                            "title": source.get("title") or "",
                            "snippet": "",
                        }
                    )

        elif kind == "message":
            content = item.get("content")
            if isinstance(content, str):
                text_parts.append(content)
            for part in (content if isinstance(content, list) else []):
                if not isinstance(part, dict):
                    continue
                if part.get("text"):
                    text_parts.append(part["text"])
                for note in part.get("annotations") or []:
                    # A citation is something the model chose to write, so it
                    # is enrichment, not a source.
                    if isinstance(note, dict) and note.get("url"):
                        response.enrichments.append(
                            {
                                "link": note["url"],
                                "title": note.get("title") or "",
                                "snippet": "",
                            }
                        )

    response.enrichments.extend(parse_json_array("".join(text_parts)))

    if not saw_search_call:
        response.warnings.append("no web_search_call item; the model did not run a search")
    elif not response.sources:
        response.warnings.append(
            "response carried no raw search results; the provider may not support "
            "include=[web_search_call.results]"
        )
    return response


def openai_error_detail(raw) -> str:
    body = _decode(raw)
    return _one_line(_error_message_from(body) or body)


FLAVORS: Dict[str, Flavor] = {
    "anthropic": Flavor(
        "anthropic",
        anthropic_build_request,
        anthropic_parse_response,
        anthropic_error_detail,
    ),
    "openai": Flavor(
        "openai",
        openai_build_request,
        openai_parse_response,
        openai_error_detail,
    ),
}


def http_post(
    url: str,
    headers: Dict[str, str],
    payload: dict,
    timeout: float,
    error_detail: Callable[[bytes], str],
) -> dict:
    """POST JSON and return the decoded response body.

    Every failure mode leaves here as an UpstreamError, so the layers above
    do not have to know about urllib.
    """
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"content-type": "application/json", "user-agent": USER_AGENT, **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as answer:
            raw = answer.read()
    except urllib.error.HTTPError as exc:  # subclass of URLError, so it goes first
        body = b""
        try:
            body = exc.read()
        except Exception:
            pass
        raise UpstreamError(
            "provider returned HTTP %d" % exc.code,
            detail=error_detail(body),
            status=exc.code,
        )
    except (TimeoutError, socket.timeout):
        raise UpstreamError("provider did not answer within %gs" % timeout)
    except urllib.error.URLError as exc:
        raise UpstreamError("could not reach the provider", detail=_one_line(repr(exc.reason)))
    except OSError as exc:
        raise UpstreamError("connection to the provider failed", detail=_one_line(repr(exc)))

    try:
        body = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        raise UpstreamError("provider returned a body that is not JSON", detail=_one_line(raw))
    if not isinstance(body, dict):
        raise UpstreamError("provider returned JSON that is not an object", detail=_one_line(raw))
    return body


def run_search(
    cfg: Config,
    query: str,
    count: int,
    fetch: Callable = http_post,
) -> Tuple[List[dict], List[str]]:
    """Ask the provider's search tool and return (results, warnings).

    `fetch` is a parameter so tests can drive the whole pipeline offline.
    """
    flavor = FLAVORS[cfg.flavor]
    prompt = render_prompt(query, count)
    url, headers, payload = flavor.build_request(cfg, prompt, count)
    body = fetch(url, headers, payload, cfg.timeout, flavor.error_detail)
    parsed = flavor.parse_response(cfg, body)
    # merge_results drops enrichments that no source backs, so a response
    # with no provider results returns [] rather than model-suggested links.
    return merge_results(parsed.sources, parsed.enrichments, count), parsed.warnings


def _authorized(expected: str, header: str) -> bool:
    prefix = "Bearer "
    if not header.startswith(prefix):
        return False
    return hmac.compare_digest(header[len(prefix) :], expected)


def _mask(secret: str) -> str:
    if not secret:
        return ""
    if len(secret) <= 8:
        return "*" * len(secret)
    return secret[:4] + "..." + secret[-4:]


def _scrub(text: str, secret: str) -> str:
    """Keep a key out of logs and responses even if it turns up in an error
    string we did not write."""
    if secret and secret in text:
        return text.replace(secret, "***")
    return text


def describe_config(cfg: Config) -> dict:
    return {
        "flavor": cfg.flavor,
        "base_url": cfg.base_url,
        "model": cfg.model,
        "api_key": _mask(cfg.api_key),
        "auth_token": _mask(cfg.auth_token),
        "host": cfg.host,
        "port": cfg.port,
        "timeout": cfg.timeout,
        "max_results": cfg.max_results,
        "web_search_tool": cfg.web_search_tool if cfg.flavor == "anthropic" else None,
        "max_uses": cfg.max_uses if cfg.flavor == "anthropic" else None,
    }


def handle_search(
    cfg: Config,
    authorization: str,
    body: bytes,
    search: Optional[Callable] = None,
) -> Tuple[int, object]:
    """Answer one search request. Returns (status, payload)."""
    search = run_search if search is None else search

    if cfg.auth_token and not _authorized(cfg.auth_token, authorization or ""):
        return 401, {"error": "unauthorized"}

    try:
        request = json.loads(body.decode("utf-8") or "{}")
    except (ValueError, UnicodeDecodeError):
        return 400, {"error": "invalid JSON body"}
    if not isinstance(request, dict):
        return 400, {"error": "JSON body must be an object"}

    query = str(request.get("query") or "").strip()
    if not query:
        return 200, []

    try:
        count = int(request.get("count") or 5)
    except (TypeError, ValueError):
        count = 5
    count = max(1, min(count, cfg.max_results))

    started = time.time()
    try:
        results, warnings = search(cfg, query, count)
    except UpstreamError as exc:
        detail = _scrub(exc.detail, cfg.api_key)
        print("[bridge] %r upstream error: %s (%s)" % (query, exc.message, detail), file=sys.stderr)
        return 502, {"error": exc.message, "detail": detail}
    except Exception as exc:  # a bug in the bridge, not a provider problem
        traceback.print_exc()
        return 502, {"error": "internal error", "detail": repr(exc)}

    for warning in warnings:
        print("[bridge] %r warning: %s" % (query, warning), file=sys.stderr)

    filled = sum(1 for row in results if row.get("snippet"))
    print(
        "[bridge] %r -> %d results, %d with snippets, %.1fs"
        % (query, len(results), filled, time.time() - started),
        file=sys.stderr,
    )
    return 200, results


class Handler(BaseHTTPRequestHandler):
    server_version = "llm-search-bridge/" + __version__

    config: Config = None  # type: ignore[assignment]  # set by main()
    # staticmethod so the call does not receive self
    search = staticmethod(run_search)

    def _reply(self, status: int, payload) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except BrokenPipeError:
            pass

    def do_GET(self) -> None:
        self._reply(
            200,
            {
                "ok": True,
                "service": "llm-search-bridge",
                "version": __version__,
                "flavor": self.config.flavor,
            },
        )

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = 0
        body = self.rfile.read(length) if length > 0 else b""
        status, payload = handle_search(
            self.config, self.headers.get("Authorization", ""), body, self.search
        )
        self._reply(status, payload)

    def log_message(self, fmt, *args) -> None:
        pass  # one line per search is already printed by handle_search


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    try:
        config = load_config()
    except ConfigError as exc:
        print("[bridge] configuration error: %s" % exc, file=sys.stderr)
        return 1

    if "--check" in argv:
        print(json.dumps(describe_config(config), indent=2))
        return 0

    if config.flavor != "anthropic":
        for name in ("BRIDGE_WEB_SEARCH_TOOL", "BRIDGE_MAX_USES"):
            if os.environ.get(name):
                print(
                    "[bridge] warning: %s only applies to the anthropic flavor" % name,
                    file=sys.stderr,
                )

    Handler.config = config
    server = ThreadingHTTPServer((config.host, config.port), Handler)
    print(
        "[bridge] listening on %s:%d -> %s (model %s, flavor %s)"
        % (config.host, config.port, config.base_url, config.model, config.flavor),
        file=sys.stderr,
    )
    print(
        "[bridge] set this as the external search URL: http://<this-host>:%d/search"
        % config.port,
        file=sys.stderr,
    )
    if config.host not in ("127.0.0.1", "localhost", "::1") and not config.auth_token:
        print(
            "[bridge] warning: bound to %s with no BRIDGE_AUTH_TOKEN, so anyone who "
            "can reach this port can spend your provider key" % config.host,
            file=sys.stderr,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
