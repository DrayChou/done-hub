package gemini

import (
	"encoding/json"
	"sync"
	"time"

	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"

	"done-hub/model"
)

// thoughtSignature 回填
//
// 背景：Gemini 3/4 返回 functionCall 时会附带 thoughtSignature，下一次请求必须把该签名
// 原样放回同一个 functionCall part，否则上游 400 INVALID_ARGUMENT。
//
// 走 OpenAI 协议的客户端（Pi / Claude Code / 各类 SDK）拿不到也回传不了这个字段，
// Done Hub 在 Gemini ⇄ OpenAI 之间转换时它会被丢弃，于是工具循环第二轮必挂。
//
// 本文件在渠道内部补上这一段：
//   - 响应：把上游返回的签名按 (函数名 + 规范化参数) 与 id 记入缓存
//   - 请求：发现 functionCall 缺签名时回填；缓存未命中则回填官方哨兵
//
// 默认关闭，需在渠道 plugin 中显式开启：
//   {"gemini_signature_autofill": {"enable": true}}
//
// 之所以要显式开启：哨兵值 skip_thought_signature_validator 只被部分网关接受，
// 官方 Gemini / Vertex 会判定为非法签名（见 clean_bytes.go 中的历史说明）。

const (
	// ThoughtSignatureAutofillPlugin 是渠道插件里的开关名
	ThoughtSignatureAutofillPlugin = "gemini_signature_autofill"
	// signatureCacheTTL 覆盖正常工具循环的间隔即可
	signatureCacheTTL = 30 * time.Minute
	signatureCacheMax = 5000
)

type signatureEntry struct {
	signature string
	at        time.Time
}

var (
	signatureMu    sync.RWMutex
	signatureCache = map[string]signatureEntry{}
)

// SignatureAutofillEnabled 判断渠道是否开启签名回填
func SignatureAutofillEnabled(channel *model.Channel) bool {
	if channel == nil || channel.Plugin == nil {
		return false
	}
	plugin := channel.Plugin.Data()
	item, ok := plugin[ThoughtSignatureAutofillPlugin]
	if !ok {
		return false
	}
	enable, _ := item["enable"].(bool)
	return enable
}

func signatureArgsKey(name string, args map[string]interface{}) string {
	if name == "" {
		return ""
	}
	// encoding/json 序列化 map 时按 key 排序，结果稳定，可直接作为缓存键
	raw, err := json.Marshal(args)
	if err != nil {
		return ""
	}
	return "a:" + name + "|" + string(raw)
}

func storeSignature(code string, signature string) {
	if code == "" || signature == "" {
		return
	}
	signatureMu.Lock()
	signatureCache[code] = signatureEntry{signature: signature, at: time.Now()}
	// 简单容量控制：超限时清理过期项，仍超限则整体清空（缓存可丢，不可泄漏内存）
	if len(signatureCache) > signatureCacheMax {
		now := time.Now()
		for key, entry := range signatureCache {
			if now.Sub(entry.at) > signatureCacheTTL {
				delete(signatureCache, key)
			}
		}
		if len(signatureCache) > signatureCacheMax {
			signatureCache = map[string]signatureEntry{}
		}
	}
	signatureMu.Unlock()
}

func loadSignature(codes ...string) string {
	now := time.Now()
	signatureMu.RLock()
	defer signatureMu.RUnlock()
	for _, code := range codes {
		if code == "" {
			continue
		}
		entry, ok := signatureCache[code]
		if !ok || now.Sub(entry.at) > signatureCacheTTL {
			continue
		}
		return entry.signature
	}
	return ""
}

// RememberSignatures 记录一次响应里所有带签名的 functionCall
func RememberSignatures(candidates []GeminiChatCandidate) {
	for _, candidate := range candidates {
		for _, part := range candidate.Content.Parts {
			if part.FunctionCall == nil || len(part.ThoughtSignature) == 0 {
				continue
			}
			var signature string
			if err := json.Unmarshal(part.ThoughtSignature, &signature); err != nil {
				continue
			}
			if signature == "" {
				continue
			}
			storeSignature("id:"+part.FunctionCall.Id, signature)
			storeSignature(signatureArgsKey(part.FunctionCall.Name, part.FunctionCall.Args), signature)
		}
	}
}

// fillMissingSignatures 遍历 contents，为缺签名的 functionCall 回填缓存值或官方哨兵
func fillMissingSignatures(data []byte) ([]byte, int) {
	contents := gjson.GetBytes(data, "contents")
	if !contents.Exists() {
		return data, 0
	}

	filled := 0
	for i, content := range contents.Array() {
		parts := content.Get("parts")
		if !parts.Exists() {
			continue
		}
		for j, part := range parts.Array() {
			if !part.Get("functionCall").Exists() {
				continue
			}
			if part.Get("thoughtSignature").Exists() {
				continue
			}
			name := part.Get("functionCall.name").String()
			code := "id:" + part.Get("functionCall.id").String()
			argsCode := ""
			if args := part.Get("functionCall.args"); args.Exists() {
				argsCode = signatureArgsKey(name, args.Value().(map[string]interface{}))
				if argsCode == "" {
					argsCode = "a:" + name
				}
			}
			signature := loadSignature(code, argsCode)
			if signature == "" {
				// 缓存未命中：用官方哨兵跳过校验，保证工具循环能继续
				signature = SkipThoughtSignatureValidator
			}
			path := "contents." + itoa(i) + ".parts." + itoa(j) + ".thoughtSignature"
			updated, err := sjson.SetBytes(data, path, signature)
			if err != nil {
				continue
			}
			data = updated
			filled++
		}
	}
	return data, filled
}

// InjectMissingSignatures 供请求构造阶段调用
func InjectMissingSignatures(data []byte) ([]byte, int) {
	return fillMissingSignatures(data)
}

func itoa(v int) string {
	if v == 0 {
		return "0"
	}
	var buf [20]byte
	pos := len(buf)
	for v > 0 {
		pos--
		buf[pos] = byte('0' + v%10)
		v /= 10
	}
	return string(buf[pos:])
}

// FillSignaturesInRequest 在结构体请求路径上回填缺失的 thoughtSignature
// 返回回填数量。OpenAI 协议进来的 chat 请求走这条路径，而不是原始字节路径。
func FillSignaturesInRequest(request *GeminiChatRequest) int {
	if request == nil {
		return 0
	}
	filled := 0
	for i := range request.Contents {
		parts := request.Contents[i].Parts
		for j := range parts {
			part := &parts[j]
			if part.FunctionCall == nil || len(part.ThoughtSignature) > 0 {
				continue
			}
			code := "id:" + part.FunctionCall.Id
			argsCode := signatureArgsKey(part.FunctionCall.Name, part.FunctionCall.Args)
			signature := loadSignature(code, argsCode)
			if signature == "" {
				signature = SkipThoughtSignatureValidator
			}
			raw, err := json.Marshal(signature)
			if err != nil {
				continue
			}
			part.ThoughtSignature = raw
			filled++
		}
	}
	return filled
}
