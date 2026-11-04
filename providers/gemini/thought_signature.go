package gemini

import (
	"bytes"
	"crypto/sha256"
	"done-hub/common"
	"done-hub/common/cache"
	"done-hub/common/logger"
	"done-hub/common/utils"
	"done-hub/model"
	"done-hub/types"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

const (
	// Real signatures are preserved automatically. This opt-in only permits the
	// upstream's documented escape hatch when a signature cannot be recovered.
	ThoughtSignatureAutofillPlugin = "gemini_signature_autofill"
	signatureCacheTTL              = 24 * time.Hour
)

type cachedThoughtSignature struct {
	Signature string
	CallHash  string
}

func signatureCallHash(name string, args map[string]interface{}) string {
	if args == nil {
		args = map[string]interface{}{}
	}
	raw, _ := json.Marshal(args)
	sum := sha256.Sum256(append([]byte(name+"\x00"), raw...))
	return fmt.Sprintf("%x", sum)
}

func SignatureAutofillEnabled(channel *model.Channel) bool {
	if channel == nil || channel.Plugin == nil {
		return false
	}
	enabled, _ := channel.Plugin.Data()[ThoughtSignatureAutofillPlugin]["enable"].(bool)
	return enabled
}

// Reuse the existing bounded memory/Redis cache. Credentials are hashed, never
// stored in cache keys or logs. A generated client-facing call ID identifies one
// occurrence, even when concurrent sessions call the same function/arguments.
func thoughtSignatureScope(channel *model.Channel, ctx *gin.Context, modelName string) string {
	if channel == nil || ctx == nil || ctx.GetInt("id") <= 0 {
		return ""
	}
	credential := sha256.Sum256([]byte(channel.Key))
	return fmt.Sprintf("%d:%d:%s:%x", channel.Id, ctx.GetInt("id"), modelName, credential)
}

func thoughtSignatureKey(scope, callID string) string {
	if scope == "" || callID == "" {
		return ""
	}
	key := sha256.Sum256([]byte(scope + "\x00" + callID))
	return fmt.Sprintf("gemini:thought_signature:%x", key)
}

// Called before conversion exposes tool calls to the client, not after a stream
// callback. Native Gemini responses are not rewritten and retain their signatures.
func preserveFunctionCallSignatures(scope string, candidates []GeminiChatCandidate) {
	if scope == "" {
		return
	}
	for i := range candidates {
		for j := range candidates[i].Content.Parts {
			part := &candidates[i].Content.Parts[j]
			if part.FunctionCall == nil {
				continue
			}
			// Never trust upstream IDs to be globally unique (some gateways reuse them).
			part.FunctionCall.clientID = "call_" + utils.GetUUID()
			var signature string
			if json.Unmarshal(part.ThoughtSignature, &signature) != nil || signature == "" {
				continue
			}
			entry := cachedThoughtSignature{Signature: signature, CallHash: signatureCallHash(part.FunctionCall.Name, part.FunctionCall.Args)}
			if err := cache.SetCache(thoughtSignatureKey(scope, part.FunctionCall.clientID), entry, signatureCacheTTL); err != nil {
				logger.SysError("[Gemini] signature cache write failed; subsequent request may require fallback")
			}
		}
	}
}

// Only model function-call parts may receive a recovered signature. Existing
// signatures, malformed input and tool-result payloads are never overwritten.
func restoreThoughtSignatures(data []byte, scope string, allowFallback bool) ([]byte, int, int, error) {
	if !gjson.ValidBytes(data) {
		return data, 0, 0, nil
	}
	restored, fallback := 0, 0
	for i, content := range gjson.GetBytes(data, "contents").Array() {
		if content.Get("role").String() != "model" {
			continue
		}
		for j, part := range content.Get("parts").Array() {
			call := part.Get("functionCall")
			if !call.IsObject() {
				call = part.Get("function_call")
			}
			if !call.IsObject() || part.Get("thoughtSignature").Exists() || part.Get("thought_signature").Exists() {
				continue
			}
			var signature string
			if key := thoughtSignatureKey(scope, call.Get("id").String()); key != "" {
				entry, err := cache.GetCache[cachedThoughtSignature](key)
				var args map[string]interface{}
				argsJSON := call.Get("args")
				validArgs := !argsJSON.Exists() || json.Unmarshal([]byte(argsJSON.Raw), &args) == nil
				if err == nil && validArgs && entry.CallHash == signatureCallHash(call.Get("name").String(), args) {
					signature = entry.Signature
				}
			}
			if signature == "" {
				if !allowFallback {
					continue
				}
				signature = SkipThoughtSignatureValidator
				fallback++
			} else {
				restored++
			}
			path := fmt.Sprintf("contents.%d.parts.%d.thoughtSignature", i, j)
			updated, err := sjson.SetBytes(data, path, signature)
			if err != nil {
				return nil, 0, 0, err
			}
			data = updated
		}
	}
	return data, restored, fallback, nil
}

// Run on the final HTTP body, after custom parameters and every raw/map/struct
// conversion path. Do not store channel-specific fallback signatures in the
// shared processed-body cache: a retry may select a different provider/channel.
func (p *GeminiProvider) prepareThoughtSignatures(req *http.Request) *types.OpenAIErrorWithStatusCode {
	if req == nil || req.Body == nil {
		return nil
	}
	data, err := io.ReadAll(req.Body)
	_ = req.Body.Close()
	if err != nil {
		return common.ErrorWrapper(err, "read_gemini_request_failed", http.StatusInternalServerError)
	}
	scope := thoughtSignatureScope(p.GetChannel(), p.GetContext(), p.OriginalModel)
	data, restored, fallback, err := restoreThoughtSignatures(data, scope, SignatureAutofillEnabled(p.GetChannel()))
	if err != nil {
		return common.ErrorWrapper(err, "restore_gemini_signature_failed", http.StatusInternalServerError)
	}
	req.Body = io.NopCloser(bytes.NewReader(data))
	req.ContentLength = int64(len(data))
	req.GetBody = func() (io.ReadCloser, error) { return io.NopCloser(bytes.NewReader(data)), nil }
	if restored+fallback > 0 {
		// Counts only: signatures, arguments and credentials must never enter logs.
		logger.SysLog(fmt.Sprintf("[Gemini] signatures restored=%d fallback=%d channel=%d", restored, fallback, p.GetChannel().Id))
	}
	return nil
}
