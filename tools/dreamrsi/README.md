# Dream-RSI capture and direct replay

`capture.py` is a temporary, credential-free HTTP boundary for bounded Claude Code
research traffic. It forwards the request route and exact transfer-decoded body to a
local NInfer endpoint, streams the response back, and writes one restricted capsule
per request. `replay.py` sends a capsule directly to an HTTP endpoint, without Claude
Code or the capture proxy.

Raw capsules contain prompts and model output. Keep the capture directory on FNN,
with access limited to the local research owner. The proxy creates owner-only
directories and files, publishes capsules atomically, then removes write permission.
It never forwards or stores Authorization, Cookie, API-key, proxy-authorization, or
other credential-like headers. It rejects request routes with credential-like query
parameters. Semantic headers such as `anthropic-version`, `anthropic-beta`,
`content-type`, `accept`, and SDK version/retry headers are kept. Capsule JSONL
descriptors contain hashes and operational metadata, never prompt or response text.
The restricted capsule preserves safe User-Agent and request-ID headers; descriptors
carry the parsed CLI version and method/path while omitting query strings.

## Capture

Bind the research listener only to FNN's tailnet address and use an explicit research
port. The evaluator owns the NInfer service lifecycle; do not start a second model
service from this tool. Claude Code may send a `GET` or `HEAD /api/hello` probe
before its Messages request. The capture proxy answers that liveness probe locally;
it does not forward it, create a capsule, or consume the request limit.

```bash
python3.11 tools/dreamrsi/capture.py \
  --listen-host 100.119.241.20 --listen-port 18080 \
  --upstream http://127.0.0.1:18081 \
  --capture-dir /home/aramirezfamily/.local/share/ninfer-e0007-01a0e759-evaluator-20260928/captures \
  --descriptor-jsonl /home/aramirezfamily/.local/share/ninfer-e0007-01a0e759-evaluator-20260928/descriptors.jsonl \
  --max-requests 2 --max-seconds 300 --max-output-tokens 128 --cache-state unknown
```

After the evaluator reports the baseline endpoint healthy, point one bounded Claude
Code call at the research listener from the Mac. `--setting-sources project,local`
keeps user-level URL or model settings from replacing the explicit research route;
`--model sonnet` selects a model name Claude Code accepts. The CLI request body keeps
that resolved model field unchanged, while NInfer Serve executes the model loaded at
startup. The research endpoint does not authenticate. For Claude Code API mode, use
a nonsecret API-key placeholder; the capture proxy strips its auth headers before
forwarding or storage.

```bash
CLAUDE_CODE_MAX_OUTPUT_TOKENS=96 \
ANTHROPIC_API_KEY=nonsecret-placeholder \
ANTHROPIC_BASE_URL=http://100.119.241.20:18080 \
claude --setting-sources project,local --model sonnet --max-turns 1 \
  -p 'A bounded research prompt' --verbose --output-format text
```

Confirm the captured `request.semantics.output_budget` before replay; the proxy rejects
the request with HTTP 413 if it exceeds the configured guard. The 96-token CLI setting
is validated against the actual request body, so the evaluator can use that exact value
for paired replay.

Set `--cache-state cold` or `--cache-state warm` only when the evaluator can establish
that state for the captured run. The default explicitly records `unknown`. The body
is inspected only for non-text metadata such as model, `stream`, `max_tokens`, and
whether prompt cache controls are present. Extra `--bucket-label` values are copied
to sanitized descriptors; do not put prompts, user identifiers, or secrets in labels.
`--max-output-tokens N` rejects a request with HTTP 413 if the body has no integer
`max_tokens` or requests more than `N`. The guard does not rewrite or store a rejected
body, and rejected attempts do not consume `--max-requests` capacity. This is useful
when a client has no request-token CLI option; accepted capsules still contain the
exact body the client sent.

Each `*.capsule/` contains:

| File | Contents |
|---|---|
| `manifest.json` | Schema version, capture time, cache state, route, safe headers, body hashes, request and response identities, stream state, chunk offsets/timing, and completion/cancellation status |
| `request.body` | Exact request body bytes after HTTP transfer framing is removed |
| `response.body` | Exact response body bytes after HTTP transfer framing is removed |

The optional descriptor JSONL follows the evaluator contract: `schema_version`,
`capsule_id`, `capsule_path`, nonempty observed `bucket_labels`, `request_sha256`,
and optional `output_budget`/`metadata`. Keep raw capsule directories on FNN; only
copy sanitized descriptor JSONL to the repository when needed for evaluation.

## Direct replay

Replay one capsule against the running model service. This command does not invoke
Claude Code and does not require the capture listener:

```bash
python3.11 tools/dreamrsi/replay.py \
  /path/to/restricted/captures/CAPSULE_ID.capsule \
  --upstream http://127.0.0.1:18081 \
  --response-output /path/to/restricted/replay-response.body
```

`replay.py` also supports a flat FNN tools directory when `capture.py` is its sibling;
the repository package does not need to be installed on that host.

The command prints a JSON summary with status, response identity, response byte
count, SHA-256, and completeness. If `--response-output` is omitted, response bytes
are consumed and hashed without being written to another file. Replay validates the
request body hash before sending and re-applies credential filtering to stored
headers.

## Exploratory derivation

`derive.py` creates a separate owner-only request capsule from the bounded rejected
Sonnet capture when the rejection is caused by `thinking.display:"omitted"`. It
changes only that JSON field to `"summarized"`, records the parent and derived request
hashes plus the exact JSON-path diff, and marks the result
`semantic_transform:true` and `production_compatible:false`. This changes reasoning
visibility semantics; the original capsule is never modified. The derived capsule
has an empty response marked `derived_not_replayed` and can be sent through
`replay.py` by the evaluator. Its public JSONL descriptor contains sanitized
metadata and hashes, while the parent's raw filesystem location appears only in the
owner-only manifest.

```bash
python3.11 tools/dreamrsi/derive.py \
  /path/to/restricted/captures/REJECTED_CAPSULE_ID.capsule
```

The deterministic derived capsule ID prevents accidental overwrite; a repeated
invocation reports that the capsule already exists. The command accepts only a
completed HTTP 400 `POST /v1/messages` capsule with model `claude-sonnet-5`,
`max_tokens:96`, adaptive thinking, and omitted display.

For the H04b validation variant, pass `--effort-override xhigh`. This variant starts
from the immutable original and adds the declared
`output_config.effort:"high"` → `"xhigh"` change alongside the thinking-display
change. Its private manifest and sanitized descriptor also record the H04a request
hash, allowing the two exploratory variants to be compared without altering either.

```bash
python3.11 tools/dreamrsi/derive.py \
  /path/to/restricted/captures/ORIGINAL_REJECTED_CAPSULE_ID.capsule \
  --effort-override xhigh
```

## Tests

The focused tests use loopback fake HTTP endpoints and do not load a model or use a
GPU:

```bash
python3.11 -m unittest tools.dreamrsi.test_capture_replay
```
