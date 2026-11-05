# Gemini tool calls through the OpenAI endpoint

This fork keeps the existing client-facing endpoints. OpenAI clients continue to
use `/v1/chat/completions`; the Gemini provider owns the protocol conversion.
No client-specific proxy or client model configuration change is required.

## Signatures

Gemini can require the `thoughtSignature` associated with a function call when
its result is submitted. OpenAI clients generally do not round-trip this native
field. Before emitting a converted response, Done Hub assigns a unique tool-call
ID and stores the real signature in its existing memory/Redis cache. The next
request restores it using that exact call ID, authenticated user, channel,
credential fingerprint, model and a hash of the function name/arguments.

- Identical calls in concurrent sessions do **not** share signatures.
- Real signatures supplied by native Gemini clients are left unchanged.
- The cache reuses the existing bounded 1 MiB local cache, or the already configured
  Redis backend. TTL is 24 hours; local capacity eviction or a process restart can
  expire entries sooner. No new database/schema, cache service or background worker.
- Responses and logs never expose the cache keys or signature values.
- Streamed tool calls keep unique IDs and indices, including parallel calls;
  `tool_calls` is the terminal reason, not an extra contradictory `stop`.

### Opt-in missing-signature fallback

For a channel whose upstream is verified to accept the escape hatch, enable the
Gemini channel plugin **缺失思考签名兼容**:

```json
{"gemini_signature_autofill":{"enable":true}}
```

This inserts `skip_thought_signature_validator` when there is no recoverable
signature. It is **off by default**. It also permits exactly one corrective resend
when the final upstream HTTP request is rejected with a recognized signature
validation **400**, including `Invalid thought signature`. The first send preserves
real signatures; the resend replaces only signatures on model parts, uses the
same URL/credential, and never mutates shared history/signature caches.

Both sends share the existing request timeout budget; this does not enable global
channel retries or extend timeouts. Client cancellation before response headers
stops the exchange. Successful stream bodies own the remaining context until
closed, preserving bounded output/usage settlement after headers. A stream that
has already returned HTTP 200 is never replayed, including in-stream errors.
A disabled plugin, unrelated error, missing replaceable signature or second
rejection returns the error rather than repeating the request.

This escape hatch is not a guarantee of preserved reasoning: bypassing validation
may lose the original thinking state. Keep it off on unverified backends. It
applies to the type-25 Gemini provider (Chat conversion and native Gemini), not
`use_openai_api` or every provider that happens to embed Gemini helpers.
No user message is injected and no tool result is removed to hide an error.

Old histories from other clients/channels, cache eviction/restart and parallel
calls with unsigned parts may require this fallback. If the backend rejects the
escape hatch, use a fresh history/native signature-preserving client or fix the
upstream; do not silently strip semantic content.

## Tool-result roles

Converted OpenAI `tool`/legacy `function` results use Gemini-native **`role: "user"`**
with `functionResponse` parts. Consecutive parallel results stay in one result turn,
paired by call ID/name; plain user text is not merged into it. Replaying results as
OpenAI's legacy `role: "function"` is not a valid assumption for Gemini streaming
backends: the same signed synthetic history failed with that role and succeeded
when only the result role was changed to `user`.

Failed streaming endpoints can themselves return `data: {"detail": ...}` rather
than an application/json error. The Gemini error handler extracts that known
error envelope with the same redaction rules instead of losing the diagnostic.

## Tool schemas

The normalizer visits schema positions only. A property named `const`, `$ref` or
`deprecated` is never treated as a schema keyword. It removes unsupported
annotations (`$schema`, `$comment`, `examples`, `deprecated`) and the redundant
unrestricted map pattern `patternProperties: {"^.*$":{}}` when safe. It preserves
`additionalProperties`, reference definitions, `const`, conditions and restrictive
patterns. Unsupported semantic constraints may still be rejected upstream; the
upstream's diagnostic is returned rather than silently weakening the tool schema.

The gateway parses Google errors and string `detail` errors from Gemini gateways.
Validation-detail arrays are not echoed because they can include private input.
Upstream credentials are redacted from diagnostics.

## Validation and packaging

```bash
go test -race ./providers/gemini ./types
go vet ./providers/gemini ./types
# Run after committing, from a clean checkout; does not mutate lock files.
hack/scripts/build-linux-static.sh v1.23.22-gemini.1
```

The release script builds the complete frontend using `web/yarn.lock`, verifies Go
modules and builds Linux/amd64 with `CGO_ENABLED=0`, embedding the source version,
commit and build time. Never ship a stub or a partial HTTP mirror of `web/build`.
Keep the previous binary and channel configuration for rollback. Switch only after
JSON, SSE, parallel tools and native tool-result round trips pass against an
isolated database copy without a secondary proxy/fallback channel masking failures.

## Scope

GPT protocol selection is a separate capability: [multi-upstream compatibility](upstream-compatibility.md)
describes the opt-in Chat-to-Responses bridge for OpenAI/custom channels, reusing
the existing converter rather than introducing provider-specific model aliases.
Neither feature certifies model identity/quality from third-party aliases or
response-ID prefixes. A supported adapter does not prove a particular account
has access to a model.
