# Slow streaming requests: heartbeat, deadlines and delivery

This change reuses the existing **per-token heartbeat**. It does not add a second
proxy, change public endpoints, inject model messages or require client config.

## What is fixed

- Before upstream headers: the heartbeat writes a legal SSE comment and flushes
  as soon as its configured delay expires. `Stop()` cancels **and joins** its worker
  before another response writer can take over. Repeated start/stop is safe.
- After upstream headers: the response loop is the only writer. If that token
  opted into heartbeat, the same loop emits comments during first-data wait and
  gaps. Heartbeat bytes are not model output, usage or success evidence.
- Upstream header waits are cancelled if a streaming client disconnects. After
  headers, the existing output drain/usage settlement behavior remains, but no
  new retry is sent and delivery is logged as failed after disconnect.
- Stream errors use valid JSON/SSE in the caller's protocol. They do not receive
  an extra `[DONE]` success terminator or a second JSON response.
- Closed producer channels are handled explicitly. A post-header upstream that
  never sends data is subject to the existing stream idle timeout too.
- Network timeout/cancellation is reported as `upstream_timeout` (504) or
  `client_disconnected` (499), not rewritten as fictional load saturation merely
  because the retry budget was exhausted. Unknown network failures retain the
  existing bounded retry policy.
- Upload read failures are cached as failures rather than reread from a closed
  body. Logs record stage, actual/declared bytes and timing, never request content.

## Existing settings

A token may enable the existing setting, for example:

```json
{"heartbeat":{"enabled":true,"timeout_seconds":5}}
```

This affects that token only. No behavior is enabled for unrelated tokens.
Keep `timeout_seconds` comfortably below an edge proxy's idle-read timeout.
Once an SSE heartbeat commits HTTP200, subsequent failures are stream errors;
HTTP status alone must **not** be used as business-success statistics.

The upstream budgets remain independent:

- `response_header_timeout`: upstream response-header wait;
- `stream_idle_timeout`: first data after headers and gaps between upstream lines;
- `relay_timeout`: total upstream request lifetime;
- `RetryTimeOut`/`RetryTimes`: budget/count for **additional** attempts, not a promise
  of that many retries after a long first request. Do not blindly replay slow work.

For a slow upstream behind an edge with a 120-second idle limit, heartbeat is the
first fix. If header wait is intentionally extended, retain an overall deadline,
verify the path end-to-end, and do not silently extend clients' hard time limits.
Large uploads can still fail before heartbeat starts: request bodies must be read
and stream mode validated first. The ingress metrics distinguish this case.

## Verification

```bash
go test -race ./relay ./relay/relay_util ./common/requester ./common ./providers/gemini ./types
go vet ./relay ./relay/relay_util ./common/requester ./common ./providers/base
```

Tests use real HTTP upstream/downstream connections with short delays. Release
acceptance must additionally cross the actual edge idle window: a synthetic
upstream delaying headers for >120 seconds, a >120-second mid-stream gap, an
upstream error after heartbeat, cancellation and genuine multi-step model tools.
No business conversation should be replayed just to exercise transport liveness.

The diagnostic fields `relay_error_code`, `stream_failed`, `client_disconnected`,
`failure_stage`, `body_read_bytes/ms`, `prompt_count_ms`, `upstream_headers_ms`,
`provider_send_ms` and `first_stream_data_ms` distinguish gateway processing,
upstream delay and downstream delivery. They do not prove the remote client
processed a response, even when the gateway completed HTTP200.
