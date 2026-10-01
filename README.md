# llm-search-bridge

Use an LLM provider's own web search as a plain search API, so applications
like [Open WebUI](https://github.com/open-webui/open-webui) can use it as their
search engine. Standard library only, no dependencies, one file.

If your model provider already searches the web for you, you do not need to buy
a second search API from Serper, Brave, Tavily or anyone else. This bridge
exposes that built-in search over the small HTTP contract Open WebUI expects
from an "external" search engine.

It runs on a home server as the search engine for a self-hosted Open WebUI
instance.

## How it works

```
Open WebUI                        llm-search-bridge                     provider
    |                                     |                                 |
    |  POST /search                       |                                 |
    |  {"query": "...", "count": 5}       |                                 |
    |------------------------------------>|                                 |
    |                                     |  POST /v1/messages              |
    |                                     |  (web_search tool enabled)      |
    |                                     |-------------------------------->|
    |                                     |                                 |
    |                                     |  search results + model answer  |
    |                                     |<--------------------------------|
    |                                     |                                 |
    |  [{"link", "title", "snippet"}]     |                                 |
    |<------------------------------------|                                 |
```

A request moves through four stages:

1. `build_request` turns the query into a provider call, with that provider's
   web search tool enabled. One function per provider wire format.
2. `http_post` sends it and normalizes every failure into one error type.
3. `parse_response` reads the provider's answer and splits it into two piles:
   results the provider retrieved, and text the model wrote.
4. `merge_results` combines the piles under the trust model below, and is the
   only function that produces output.

Both wire formats share everything except stages 1 and 3, so the interesting
logic is written once.

## The trust model

This is the part worth reading.

**A URL is returned only if it arrived inside the provider's own search result
structure. The model can add text, never links.**

Models occasionally invent a citation that looks plausible and resolves to
nothing. A search bridge that hands those to a chat application is worse than
one that returns fewer results, because the failure is invisible: the citation
renders like any other. So the two inputs are treated differently:

| Flavor | Where a URL comes from | Classification |
| --- | --- | --- |
| anthropic | `web_search_tool_result` blocks | trusted, sets the order |
| anthropic | the model's JSON answer | enrichment only |
| openai | `web_search_call.results[]` (via `include`) | trusted, sets the order |
| openai | `web_search_call.action.sources[]` | trusted, lower fidelity |
| openai | `message.annotations[]` citations | enrichment only |
| openai | the model's JSON answer | enrichment only |

An enrichment may fill in an empty title or snippet on a URL that a trusted
result already contributed, and is discarded otherwise. When both describe the
same page, the provider's text wins.

Citations are classified as enrichment rather than results because a citation
is something the model chose to write, while the `include` parameter is what
buys the provider's raw result set. The asymmetry of the two possible mistakes
settles the argument: treating a real result as untrusted loses a title, but
treating a model-written URL as trusted is exactly the failure this project
exists to prevent.

The same logic drives the degradation path. If the provider returns no raw
results, the bridge returns an empty array and logs a warning. It does not fall
back to the model's suggested links.

Two smaller pieces of the same idea:

- URLs are canonicalized before they are compared, so `http` and `https`,
  `www` and non-`www`, tracking parameters, default ports and fragments do not
  turn one page into three results.
- Rows that a provider sends twice collapse into one.

## Quickstart

```sh
git clone https://github.com/DanyZein/llm-search-bridge.git
cd llm-search-bridge

cp .env.example .env      # then edit it, or set the variables your own way
set -a; . ./.env; set +a

python3 llm_search_bridge.py --check    # prints the resolved settings, key masked
python3 llm_search_bridge.py
```

Requires Python 3.8 or newer. There is nothing to install.

Then wire it into Open WebUI (see below) and check it answers:

```sh
curl -s localhost:8899 -X POST -H 'Content-Type: application/json' \
  -d '{"query": "who won the last F1 race", "count": 3}'
```

## Configuration

Everything comes from the environment. The script never reads `.env` itself, so
load it however you prefer (systemd `EnvironmentFile`, `set -a; . .env`, Docker
`--env-file`, and so on).

| Variable | Default | Meaning |
| --- | --- | --- |
| `BRIDGE_FLAVOR` | `anthropic` | Which wire format to speak: `anthropic` or `openai`. |
| `BRIDGE_API_KEY` | required | Provider key. |
| `BRIDGE_BASE_URL` | required | Provider base URL, for example `https://api.deepseek.com/anthropic`. |
| `BRIDGE_MODEL` | required | Model name. No default, because vendor model names change. |
| `BRIDGE_AUTH_TOKEN` | empty | Shared secret clients must send as `Authorization: Bearer <value>`. Empty disables authentication. |
| `BRIDGE_HOST` | `127.0.0.1` | Bind address. Set `0.0.0.0` only if the client is on another host or in a container, and set a token when you do. |
| `BRIDGE_PORT` | `8899` | Bind port. |
| `BRIDGE_TIMEOUT` | `180` | Upstream timeout in seconds. |
| `BRIDGE_MAX_RESULTS` | `10` | Ceiling applied to the `count` the client asks for. |
| `BRIDGE_WEB_SEARCH_TOOL` | `web_search_20250305` | anthropic flavor only. The provider's tool version string. Newer ones exist, but not every compatible endpoint accepts them. |
| `BRIDGE_MAX_USES` | unset | anthropic flavor only. Caps how many searches the model may run per request. |

## Wiring it into Open WebUI

1. Admin Panel -> Settings -> Web Search.
2. Set **Web Search Engine** to `external`.
3. Set **External Search URL** to `http://<host>:8899/search`.
4. Put the value of `BRIDGE_AUTH_TOKEN` in **External Search API Key**.
5. Save, then run a query that needs current information.

If Open WebUI runs in a container, `localhost` inside that container is not the
host, so use the host's LAN address or put both on the same Docker network, and
remember to set `BRIDGE_HOST=0.0.0.0`.

Open WebUI's own result count maps to the `count` field of the request. The
bridge clamps it to `BRIDGE_MAX_RESULTS`. Any URL path is accepted, and `GET /`
answers a health check.

## Provider compatibility

| Provider | `BRIDGE_FLAVOR` | Notes |
| --- | --- | --- |
| DeepSeek | `anthropic` | `BRIDGE_BASE_URL=https://api.deepseek.com/anthropic`. This is the setup the bridge was written for. |
| Anthropic | `anthropic` | Native web search. Some accounts need a different `BRIDGE_WEB_SEARCH_TOOL` version string. |
| OpenAI | `openai` | Responses API with the `web_search` tool. Returns real titles, URLs and snippets, so this flavor does not depend on the model for snippets at all. |

What cannot work, and why:

- **OpenAI-compatible chat endpoints** (OpenRouter, Groq, Together, Fireworks,
  vLLM, LM Studio, llama.cpp and friends). They serve `/v1/chat/completions`.
  The `openai` flavor needs `/v1/responses` with a provider-run `web_search`
  tool and the `include` parameter. A gateway that proxies chat completions
  only has nowhere to run the search.
- **Local models.** There is no search to bridge. The model would have to fetch
  the web itself, which is a different program.
- **Providers that answer with a prose summary only.** If the API does not
  expose the URLs it retrieved, the trust model has nothing to stand on.
- **Providers with a different grounding schema** (Gemini grounding, xAI Live
  Search, and similar). They are not supported out of the box, but adding a
  flavor is three small functions. See below.

## Adding a provider

A flavor is three pure functions and one dictionary entry:

```python
def my_build_request(cfg, prompt, count):
    return url, headers, payload       # where to send the search request

def my_parse_response(cfg, body):
    return FlavorResponse(sources=[...], enrichments=[...], warnings=[...])

def my_error_detail(raw_error_body):
    return "one line for the logs"
```

`sources` may only contain URLs that the provider's own search response
carries. `enrichments` may contain anything the model wrote. Everything else,
including canonicalization and merging, is shared, and no network or key is
needed to test a flavor: feed `parse_response` a recorded response.

## Deploying as a service

`deploy/llm-search-bridge.service` is an example systemd unit.

```sh
sudo useradd --system --no-create-home llm-search-bridge
sudo mkdir -p /opt/llm-search-bridge
sudo cp llm_search_bridge.py /opt/llm-search-bridge/

sudo install -m 600 -o root -g root /dev/null /etc/llm-search-bridge.env
sudo editor /etc/llm-search-bridge.env      # the variables from the table above
sudo chmod 600 /etc/llm-search-bridge.env

sudo cp deploy/llm-search-bridge.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now llm-search-bridge
journalctl -u llm-search-bridge -f          # one log line per search
```

The environment file should be readable only by root, since it holds the
provider key. The unit runs with `ProtectSystem=strict`, which is why it also
sets `PYTHONDONTWRITEBYTECODE=1`: the service cannot write to `/opt` anyway.

## Running the tests

```sh
python3 -m unittest discover -s tests -t . -v
```

No network, no API key, no fixtures to record. Provider responses live in
`tests/fixtures/`, and the search call is injected, so the suite covers the
trust model, both parsers, URL canonicalization, configuration handling and the
HTTP layer without leaving the process.

## Limitations

- One provider key per process. Run a second instance for a second provider.
- Every search is a billable model call, and takes seconds, not milliseconds.
- No caching, no retries, no rate limiting. A provider 429 reaches the client
  as a 502 with the provider's message in `detail`.
- Results follow the provider's ranking. The model's answer only supplies
  titles and snippets, so a "best first" hint from the model does not reorder
  anything.
- On the `anthropic` flavor, snippets come from the model's JSON answer. If the
  model ignores that instruction, results are still returned, with empty
  snippets and a warning in the log.
- The bridge never opens the result pages, so it cannot check that a page still
  exists or that its title matches.
- Authentication is one shared token. There is no per-user identity.
- Queries are written to stderr and sent to the provider. Nothing else is
  stored.

## License

MIT. See [LICENSE](LICENSE).
