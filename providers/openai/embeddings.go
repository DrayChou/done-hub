package openai

import (
	"bytes"
	"context"
	"done-hub/common"
	"done-hub/common/config"
	"done-hub/common/requester"
	"done-hub/model"
	"done-hub/providers/base"
	"done-hub/types"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"golang.org/x/sync/errgroup"
)

const (
	embeddingBatchLimit       = 32
	embeddingBatchConcurrency = 4
)

func (p *OpenAIProvider) CreateEmbeddings(request *types.EmbeddingRequest) (*types.EmbeddingResponse, *types.OpenAIErrorWithStatusCode) {
	if items, standard := standardEmbeddingInputs(request.Input); standard && isMultimodalEmbeddingChannel(p.Channel, request.Model) {
		if len(items) == 0 {
			return nil, common.StringErrorWrapperLocal("embedding input must not be empty", "invalid_request_error", http.StatusBadRequest)
		}
		if len(items) > embeddingBatchLimit {
			return nil, common.StringErrorWrapperLocal(fmt.Sprintf("embedding batch is limited to %d inputs", embeddingBatchLimit), "invalid_request_error", http.StatusBadRequest)
		}
		return p.createEmbeddingBatch(request, items)
	}

	req, errWithCode := p.GetRequestTextBody(config.RelayModeEmbeddings, request.Model, request)
	if errWithCode != nil {
		return nil, errWithCode
	}
	defer req.Body.Close()

	response := &OpenAIProviderRawEmbeddingsResponse{}
	// Keep the complete upstream body so a vendor-native data object can be
	// passed through byte-for-byte after its usage has been parsed.
	resp, errWithCode := p.Requester.SendRequest(req, response, true)
	if errWithCode != nil {
		return nil, errWithCode
	}

	openaiErr := ErrorHandle(&response.OpenAIErrorResponse)
	if openaiErr != nil {
		errWithCode = &types.OpenAIErrorWithStatusCode{
			OpenAIError: *openaiErr,
			StatusCode:  http.StatusBadRequest,
		}
		return nil, errWithCode
	}

	rawBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, common.ErrorWrapper(err, "read_embeddings_response_failed", http.StatusBadGateway)
	}
	data, nativeObject, err := decodeEmbeddingData(response.Data)
	if err != nil {
		return nil, common.ErrorWrapper(err, "decode_embeddings_data_failed", http.StatusBadGateway)
	}

	if response.Usage != nil {
		*p.Usage = *response.Usage
	}

	result := &types.EmbeddingResponse{
		Object: response.Object,
		Data:   data,
		Model:  response.Model,
		Usage:  response.Usage,
	}

	// A vendor-native data object cannot be represented by the standard
	// []Embedding response without dropping multi/sparse vectors. Preserve the
	// original bytes; standard array responses keep the existing struct path.
	if nativeObject && p.Context != nil {
		if patched, changed := base.UnifyModelInJSONBytes(p.Context, rawBody, "model"); changed {
			rawBody = patched
		}
		p.storeOpenAIUpstreamHeaders(resp.Header)
		p.Context.Set(config.GinRawResponseBodyKey, rawBody)
	}

	return result, nil
}

// standardEmbeddingInputs recognizes only the OpenAI standard string forms.
// Arrays containing typed objects remain on the native multimodal path.
func standardEmbeddingInputs(input any) ([]string, bool) {
	switch value := input.(type) {
	case string:
		return []string{value}, true
	case []string:
		return value, true
	case []any:
		items := make([]string, 0, len(value))
		for _, item := range value {
			text, ok := item.(string)
			if !ok {
				return nil, false
			}
			items = append(items, text)
		}
		return items, true
	default:
		return nil, false
	}
}

// isMultimodalEmbeddingChannel checks the configured Embedding route, rather
// than guessing from a hostname or model name alone.
func isMultimodalEmbeddingChannel(channel *model.Channel, modelName string) bool {
	if channel == nil || channel.Plugin == nil || !strings.HasPrefix(modelName, "doubao-embedding-vision") {
		return false
	}
	customize := channel.Plugin.Data()["customize"]
	path, _ := customize["3"].(string)
	return strings.Contains(path, "/embeddings/multimodal")
}

func (p *OpenAIProvider) createEmbeddingBatch(request *types.EmbeddingRequest, items []string) (*types.EmbeddingResponse, *types.OpenAIErrorWithStatusCode) {
	url, errWithCode := p.GetSupportedAPIUri(config.RelayModeEmbeddings)
	if errWithCode != nil {
		return nil, errWithCode
	}
	fullURL := p.GetFullRequestURL(url, request.Model)
	headers := p.GetRequestHeaders()
	baseCtx := context.Background()
	if p.Context != nil && p.Context.Request != nil {
		baseCtx = p.Context.Request.Context()
	}

	data := make([]types.Embedding, len(items))
	usages := make([]*types.Usage, len(items))
	errs := make([]*types.OpenAIErrorWithStatusCode, len(items))
	group, ctx := errgroup.WithContext(baseCtx)
	group.SetLimit(embeddingBatchConcurrency)
	for index, text := range items {
		index, text := index, text
		group.Go(func() error {
			embedding, usage, itemErr := p.sendEmbeddingBatchItem(ctx, fullURL, headers, request, text, index)
			if itemErr != nil {
				errs[index] = itemErr
				return itemErr
			}
			data[index] = embedding
			usages[index] = usage
			return nil
		})
	}
	_ = group.Wait()
	for index, itemErr := range errs {
		if itemErr != nil {
			wrapped := *itemErr
			wrapped.Message = fmt.Sprintf("embedding batch item %d failed: %s", index+1, itemErr.Message)
			return nil, &wrapped
		}
	}

	total := &types.Usage{}
	for _, usage := range usages {
		mergeEmbeddingUsage(total, usage)
	}
	if p.Usage == nil {
		p.Usage = &types.Usage{}
	}
	*p.Usage = *total

	return &types.EmbeddingResponse{
		Object: "list",
		Data:   data,
		Model:  request.Model,
		Usage:  total,
	}, nil
}

func (p *OpenAIProvider) sendEmbeddingBatchItem(ctx context.Context, fullURL string, headers map[string]string, request *types.EmbeddingRequest, text string, index int) (types.Embedding, *types.Usage, *types.OpenAIErrorWithStatusCode) {
	payload := map[string]any{
		"model": request.Model,
		"input": []map[string]string{{"type": "text", "text": text}},
	}
	if request.Dimensions > 0 {
		payload["dimensions"] = request.Dimensions
	}
	if request.EncodingFormat != "" {
		payload["encoding_format"] = request.EncodingFormat
	}
	if request.User != "" {
		payload["user"] = request.User
	}

	r := requester.NewHTTPRequester(p.Channel.GetProxy(), RequestErrorHandle)
	r.Context = ctx
	req, err := r.NewRequest(http.MethodPost, fullURL, r.WithBody(payload), r.WithHeader(headers))
	if err != nil {
		return types.Embedding{}, nil, common.ErrorWrapperLocal(err, "new_embeddings_request_failed", http.StatusInternalServerError)
	}
	defer req.Body.Close()

	rawResponse := &OpenAIProviderRawEmbeddingsResponse{}
	resp, errWithCode := r.SendRequest(req, rawResponse, true)
	if errWithCode != nil {
		return types.Embedding{}, nil, errWithCode
	}
	if _, err := io.ReadAll(resp.Body); err != nil {
		return types.Embedding{}, nil, common.ErrorWrapperLocal(err, "read_embeddings_response_failed", http.StatusBadGateway)
	}

	var native struct {
		Embedding json.RawMessage `json:"embedding"`
	}
	if err := json.Unmarshal(rawResponse.Data, &native); err != nil || len(native.Embedding) == 0 {
		return types.Embedding{}, nil, common.StringErrorWrapperLocal("native embedding response has no dense embedding", "decode_embeddings_data_failed", http.StatusBadGateway)
	}
	var dense []json.RawMessage
	if err := json.Unmarshal(native.Embedding, &dense); err != nil {
		return types.Embedding{}, nil, common.ErrorWrapperLocal(err, "decode_embeddings_data_failed", http.StatusBadGateway)
	}

	return types.Embedding{Object: "embedding", Embedding: native.Embedding, Index: index}, rawResponse.Usage, nil
}

func mergeEmbeddingUsage(total, usage *types.Usage) {
	if total == nil || usage == nil {
		return
	}
	total.PromptTokens += usage.PromptTokens
	total.CompletionTokens += usage.CompletionTokens
	total.TotalTokens += usage.TotalTokens
	total.PromptTokensDetails.TextTokens += usage.PromptTokensDetails.TextTokens
	total.PromptTokensDetails.ImageTokens += usage.PromptTokensDetails.ImageTokens
	total.PromptTokensDetails.AudioTokens += usage.PromptTokensDetails.AudioTokens
	total.PromptTokensDetails.CachedTokens += usage.PromptTokensDetails.CachedTokens
}

// decodeEmbeddingData distinguishes the standard OpenAI array from a native
// vendor object. Only an object is eligible for raw response passthrough.
func decodeEmbeddingData(raw json.RawMessage) ([]types.Embedding, bool, error) {
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) == 0 {
		return nil, false, errors.New("embeddings response has no data")
	}
	switch trimmed[0] {
	case '[':
		var data []types.Embedding
		if err := json.Unmarshal(trimmed, &data); err != nil {
			return nil, false, err
		}
		return data, false, nil
	case '{':
		var object map[string]json.RawMessage
		if err := json.Unmarshal(trimmed, &object); err != nil {
			return nil, false, err
		}
		if len(object["embedding"]) == 0 {
			return nil, false, errors.New("native embeddings response has no embedding")
		}
		return nil, true, nil
	default:
		return nil, false, errors.New("embeddings response data must be an array or object")
	}
}
