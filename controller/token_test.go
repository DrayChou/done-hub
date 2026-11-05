package controller

import (
	"done-hub/model"
	"testing"
)

func TestTokenHeartbeatDelayMatchesRuntime(t *testing.T) {
	for _, seconds := range []int{0, 4, 5, 30, 90, 91} {
		setting := &model.TokenSetting{Heartbeat: model.HeartbeatSetting{Enabled: true, TimeoutSeconds: seconds}}
		err := validateTokenSetting(setting)
		wantValid := seconds >= 5 && seconds <= 90
		if (err == nil) != wantValid {
			t.Fatalf("heartbeat delay=%d accepted=%t want=%t", seconds, err == nil, wantValid)
		}
	}
	if err := validateTokenSetting(&model.TokenSetting{}); err != nil {
		t.Fatalf("disabled heartbeat changed behavior: %v", err)
	}
}
