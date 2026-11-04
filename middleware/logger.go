package middleware

import (
	"done-hub/common/logger"
	"done-hub/metrics"
	"strings"
	"time"

	"go.uber.org/zap"
	"go.uber.org/zap/zapcore"

	"github.com/gin-gonic/gin"
)

func SetUpLogger(server *gin.Engine) {

	server.Use(GinzapWithConfig())
}

func GinzapWithConfig() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()
		path := c.Request.URL.Path
		query := c.Request.URL.RawQuery
		// 如果query的值包含key=sk-fsdfsdfsdf 则把sk-fsdfsdfsdf脱敏处理
		if query != "" {
			if i := strings.Index(query, "key=sk-"); i >= 0 {
				start := i + 4 // "key=" length
				end := strings.Index(query[start:], "&")
				if end == -1 {
					end = len(query)
				} else {
					end += start
				}
				query = query[:start] + "sk-***" + query[end:]
			}
		}
		c.Next()
		end := time.Now()
		latency := end.Sub(start)
		requestID := c.GetString(logger.RequestIdKey)
		userID := c.GetInt("id")

		fields := []zapcore.Field{
			zap.Int("status", c.Writer.Status()),
			zap.String("request_id", requestID),
			zap.String("method", c.Request.Method),
			zap.String("path", path),
			zap.String("query", query),
			zap.String("ip", c.ClientIP()),
			zap.String("user-agent", c.Request.UserAgent()),
			zap.Duration("latency", latency),
			zap.Int("user_id", userID),
			zap.String("original_model", c.GetString("original_model")),
			zap.String("new_model", c.GetString("new_model")),
			zap.Int("token_id", c.GetInt("token_id")),
			zap.String("token_name", c.GetString("token_name")),
			zap.Int("channel_id", c.GetInt("channel_id")),
			// Transport status alone is not business success after SSE heartbeats.
			zap.Any("relay_error_code", func() any { v, _ := c.Get("relay_error_code"); return v }()),
			zap.Bool("client_disconnected", c.GetBool("client_disconnected")),
			zap.String("failure_stage", c.GetString("request_failure_stage")),
			zap.Int64("content_length", c.Request.ContentLength),
			zap.Int64("body_read_bytes", c.GetInt64("request_body_bytes")),
			zap.Int64("body_read_ms", c.GetInt64("request_body_read_ms")),
			zap.Int64("first_stream_data_ms", c.GetInt64("stream_first_data_ms")),
			zap.Int64("prompt_count_ms", c.GetInt64("prompt_count_ms")),
			zap.Int64("upstream_headers_ms", c.GetInt64("upstream_headers_ms")),
			zap.Int64("provider_send_ms", c.GetInt64("provider_send_ms")),
			zap.Bool("stream_failed", c.GetBool("stream_failed")),
		}

		if len(c.Errors) > 0 {
			// Append error field if this is an erroneous request.
			for _, e := range c.Errors.Errors() {
				logger.Logger.Error(e, fields...)
			}
		} else {
			logger.Logger.Info("GIN request", fields...)
		}
		metrics.RecordHttp(c, latency)
	}
}
