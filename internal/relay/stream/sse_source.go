package stream

import (
	"context"
	"io"
	"strings"
	"sync"

	"github.com/tmaxmax/go-sse"
)

// SSESource wraps an SSE event stream (tmaxmax/go-sse).
type SSESource struct {
	reader    io.ReadCloser
	cfg       *sse.ReadConfig
	events    chan sseReadResult
	done      chan struct{}
	closeOnce sync.Once
	closeErr  error
	block     bool
}

type sseReadResult struct {
	event sse.Event
	err   error
}

// NewSSESource creates a source from an HTTP response body.
func NewSSESource(reader io.ReadCloser, maxEventSize int) *SSESource {
	cfg := &sse.ReadConfig{MaxEventSize: maxEventSize}
	if maxEventSize <= 0 {
		cfg.MaxEventSize = 32 * 1024 * 1024 // 32MB default
	}

	s := &SSESource{
		reader: reader,
		cfg:    cfg,
		events: make(chan sseReadResult, 1),
		done:   make(chan struct{}),
	}

	// Start reading in background
	go s.readLoop()

	return s
}

// NewSSEBlockSource returns complete SSE blocks, including the optional event
// field. It is used by compatibility paths that must rewrite an event while
// preserving its wire shape.
func NewSSEBlockSource(reader io.ReadCloser, maxEventSize int) *SSESource {
	s := NewSSESource(reader, maxEventSize)
	s.block = true
	return s
}

func (s *SSESource) readLoop() {
	defer close(s.events)
	for ev, err := range sse.Read(s.reader, s.cfg) {
		select {
		case s.events <- sseReadResult{event: ev, err: err}:
			if err != nil {
				return
			}
		case <-s.done:
			return
		}
	}
}

// ReadEvent reads the next SSE event data.
func (s *SSESource) ReadEvent(ctx context.Context) ([]byte, error) {
	select {
	case result, ok := <-s.events:
		if !ok {
			return nil, io.EOF
		}
		if result.err != nil {
			return nil, result.err
		}
		if s.block {
			return formatSSEBlock(result.event), nil
		}
		return []byte(result.event.Data), nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

func formatSSEBlock(event sse.Event) []byte {
	var builder strings.Builder
	if event.Type != "" {
		builder.WriteString("event: ")
		builder.WriteString(event.Type)
		builder.WriteByte('\n')
	}
	for _, line := range strings.Split(event.Data, "\n") {
		builder.WriteString("data: ")
		builder.WriteString(line)
		builder.WriteByte('\n')
	}
	builder.WriteByte('\n')
	return []byte(builder.String())
}

// Close releases the underlying reader.
func (s *SSESource) Close() error {
	s.closeOnce.Do(func() {
		close(s.done)
		if s.reader != nil {
			s.closeErr = s.reader.Close()
		}
	})
	return s.closeErr
}
