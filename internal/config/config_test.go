package config

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg == nil {
		t.Fatal("DefaultConfig returned nil")
	}

	if cfg.AppName != "P.A.R.T.N.E.R" {
		t.Errorf("Expected AppName 'P.A.R.T.N.E.R', got %s", cfg.AppName)
	}

	if cfg.Version == "" {
		t.Error("Version not set")
	}

	if cfg.Model.InputSize != 768 {
		t.Errorf("Expected InputSize 768, got %d", cfg.Model.InputSize)
	}
}

func TestConfigValidation(t *testing.T) {
	cfg := DefaultConfig()

	// Valid config should pass
	if err := cfg.Validate(); err != nil {
		t.Errorf("Valid config failed validation: %v", err)
	}

	// Test invalid FPS
	cfg.Vision.CaptureFPS = 0
	if err := cfg.Validate(); err == nil {
		t.Error("Expected error for invalid FPS")
	}
	cfg.Vision.CaptureFPS = 10

	// Test invalid batch size
	cfg.Model.BatchSize = 0
	if err := cfg.Validate(); err == nil {
		t.Error("Expected error for invalid batch size")
	}
	cfg.Model.BatchSize = 32

	// Test invalid learning rate
	cfg.Model.LearningRate = 0
	if err := cfg.Validate(); err == nil {
		t.Error("Expected error for invalid learning rate")
	}
	cfg.Model.LearningRate = 0.001

	// Test invalid CPU usage
	cfg.Performance.MaxCPUUsage = 150
	if err := cfg.Validate(); err == nil {
		t.Error("Expected error for invalid CPU usage")
	}
}

func TestConfigSaveLoad(t *testing.T) {
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "test_config.json")

	// Create and save config
	cfg := DefaultConfig()
	cfg.AppName = "TestApp"

	if err := cfg.Save(configPath); err != nil {
		t.Fatalf("Failed to save config: %v", err)
	}

	// Check file exists
	if _, err := os.Stat(configPath); os.IsNotExist(err) {
		t.Fatal("Config file was not created")
	}

	// Load config
	loaded, err := Load(configPath)
	if err != nil {
		t.Fatalf("Failed to load config: %v", err)
	}

	if loaded.AppName != "TestApp" {
		t.Errorf("Expected AppName 'TestApp', got %s", loaded.AppName)
	}
}

func TestLoadOrDefault(t *testing.T) {
	// Test with non-existent file
	cfg := LoadOrDefault("nonexistent.json")
	if cfg == nil {
		t.Fatal("LoadOrDefault returned nil")
	}

	if cfg.AppName != "P.A.R.T.N.E.R" {
		t.Error("LoadOrDefault did not return default config")
	}

	// Test with existing file
	tmpDir := t.TempDir()
	configPath := filepath.Join(tmpDir, "config.json")

	testCfg := DefaultConfig()
	testCfg.AppName = "CustomName"
	testCfg.Save(configPath)

	loaded := LoadOrDefault(configPath)
	if loaded.AppName != "CustomName" {
		t.Error("LoadOrDefault did not load existing config")
	}
}

func TestEnsureDirectories(t *testing.T) {
	tmpDir := t.TempDir()

	cfg := DefaultConfig()
	cfg.Interface.LogPath = filepath.Join(tmpDir, "logs", "test.log")
	cfg.Model.ModelPath = filepath.Join(tmpDir, "models", "test.model")
	cfg.Training.DBPath = filepath.Join(tmpDir, "data", "test.db")

	if err := cfg.EnsureDirectories(); err != nil {
		t.Fatalf("Failed to ensure directories: %v", err)
	}

	// Check directories were created
	dirs := []string{
		filepath.Join(tmpDir, "logs"),
		filepath.Join(tmpDir, "models"),
		filepath.Join(tmpDir, "data"),
	}

	for _, dir := range dirs {
		if _, err := os.Stat(dir); os.IsNotExist(err) {
			t.Errorf("Directory was not created: %s", dir)
		}
	}
}

func TestConfigFieldsPresent(t *testing.T) {
	cfg := DefaultConfig()

	// Test all major config sections exist
	if cfg.Vision.CaptureFPS == 0 {
		t.Error("Vision config not initialized")
	}

	if cfg.Model.BatchSize == 0 {
		t.Error("Model config not initialized")
	}

	if cfg.Training.ReplayBufferSize == 0 {
		t.Error("Training config not initialized")
	}

	if cfg.Interface.LogLevel == "" {
		t.Error("Interface config not initialized")
	}

	if cfg.Performance.MaxCPUUsage == 0 {
		t.Error("Performance config not initialized")
	}
}

func TestScreenRegion(t *testing.T) {
	cfg := DefaultConfig()

	region := cfg.Vision.ScreenRegion

	if region.Width <= 0 || region.Height <= 0 {
		t.Error("Invalid screen region dimensions")
	}

	if region.X < 0 || region.Y < 0 {
		t.Error("Invalid screen region coordinates")
	}
}
