package openai

import (
	"bytes"
	"done-hub/common"
	"done-hub/common/config"
	"done-hub/providers/base"
	"done-hub/types"
	"encoding/json"
	"errors"
	"io"
	"net/http"
)

func (p *OpenAIProvider) CreateEmbeddings(request *types.EmbeddingRequest) (*types.EmbeddingResponse, *types.OpenAIErrorWithStatusCode) {
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
