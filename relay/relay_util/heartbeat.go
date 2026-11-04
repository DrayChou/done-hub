package relay_util

import (
	"context"
	"done-hub/common/requester"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/gin-gonic/gin"
)

const HeartbeatStreamText = ": ping\n\n"
const HeartbeatJsonText = "\n"

// HeartbeatConfig controls the existing, opt-in token heartbeat.
type HeartbeatConfig struct {
	TimeoutSeconds  int
	IntervalSeconds int
}

// Heartbeat owns the response writer only while waiting for upstream headers.
// Stop joins the worker before the normal stream/JSON writer takes ownership.
// A heartbeat is transport liveness, never model output or billing evidence.
type Heartbeat struct {
	isStream bool
	config   HeartbeatConfig
	ctx      context.Context
	cancel   context.CancelFunc
	c        *gin.Context

	mu         sync.Mutex
	started    bool
	stopped    bool
	done       chan struct{}
	headerSent atomic.Bool
}

func NewHeartbeat(isStream bool, config HeartbeatConfig, c *gin.Context) *Heartbeat {
	if config.IntervalSeconds < 5 {
		config.IntervalSeconds = 5
	}
	if config.TimeoutSeconds < 0 {
		config.TimeoutSeconds = 0
	}
	parent := context.Background()
	if c != nil && c.Request != nil {
		parent = c.Request.Context()
	}
	ctx, cancel := context.WithCancel(parent)
	return &Heartbeat{isStream: isStream, config: config, ctx: ctx, cancel: cancel, c: c, done: make(chan struct{})}
}

func (h *Heartbeat) HasWrittenHeader() bool { return h.headerSent.Load() }

func (h *Heartbeat) Start() {
	h.mu.Lock()
	defer h.mu.Unlock()
	if h.started || h.stopped {
		return
	}
	h.started = true // establish ownership before starting the asynchronous work
	go h.run()
}

func (h *Heartbeat) run() {
	defer close(h.done)
	timer := time.NewTimer(time.Duration(h.config.TimeoutSeconds) * time.Second)
	defer timer.Stop()
	select {
	case <-h.ctx.Done():
		return
	case <-timer.C:
	}
	if !h.write() {
		return
	}
	ticker := time.NewTicker(time.Duration(h.config.IntervalSeconds) * time.Second)
	defer ticker.Stop()
	for {
		select {
		case <-h.ctx.Done():
			return
		case <-ticker.C:
			if !h.write() {
				return
			}
		}
	}
}

func (h *Heartbeat) write() bool {
	if h.ctx.Err() != nil || h.c == nil {
		return false
	}
	if !h.headerSent.Load() {
		if h.isStream {
			requester.SetEventStreamHeaders(h.c)
		} else {
			h.c.Writer.Header().Set("Content-Type", "application/json")
			h.c.Writer.WriteHeader(http.StatusOK)
		}
	}
	text := HeartbeatJsonText
	if h.isStream {
		text = HeartbeatStreamText
	}
	// Bound a stalled client write too; Stop must not join an indefinitely
	// blocked writer. Existing deployments without ResponseController support
	// retain their server timeout behavior.
	control := http.NewResponseController(h.c.Writer)
	_ = control.SetWriteDeadline(time.Now().Add(10 * time.Second))
	n, err := h.c.Writer.Write([]byte(text))
	if n > 0 || h.c.Writer.Written() {
		h.headerSent.Store(true)
	}
	if err != nil {
		_ = control.SetWriteDeadline(time.Time{})
		return false
	}
	err = control.Flush()
	_ = control.SetWriteDeadline(time.Time{})
	if err != nil {
		return false
	}
	h.headerSent.Store(true)
	return true
}

func (h *Heartbeat) Stop() {
	h.mu.Lock()
	h.stopped = true
	started := h.started
	h.cancel()
	h.mu.Unlock()
	if started {
		<-h.done
	}
}

func (h *Heartbeat) Close() { h.Stop() }

func (h *Heartbeat) IsSafeWriteStream() bool {
	h.Stop()
	return h.isStream && h.HasWrittenHeader()
}
