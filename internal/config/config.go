package config

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// Config represents the application configuration
type Config struct {
	AppName     string            `json:"app_name"`
	Version     string            `json:"version"`
	Vision      VisionConfig      `json:"vision"`
	Model       ModelConfig       `json:"model"`
	Training    TrainingConfig    `json:"training"`
	Interface   InterfaceConfig   `json:"interface"`
	Performance PerformanceConfig `json:"performance"`
}

// VisionConfig contains vision capture settings
type VisionConfig struct {
	ScreenRegion  Region  `json:"screen_region"`
	CaptureFPS    int     `json:"capture_fps"`
	BoardSize     int     `json:"board_size"`
	DiffThreshold float64 `json:"diff_threshold"`
}

// Region defines a screen capture area
type Region struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

// ModelConfig contains neural network settings
type ModelConfig struct {
	InputSize    int     `json:"input_size"`
	HiddenSize   int     `json:"hidden_size"`
	OutputSize   int     `json:"output_size"`
	LearningRate float64 `json:"learning_rate"`
	BatchSize    int     `json:"batch_size"`
	ModelPath    string  `json:"model_path"`
}

// TrainingConfig contains training and replay buffer settings
type TrainingConfig struct {
	ReplayBufferSize         int    `json:"replay_buffer_size"`
	DBPath                   string `json:"db_path"`
	SaveInterval             int    `json:"save_interval"`
	MinSamplesBeforeTraining int    `json:"min_samples_before_training"`
}

// InterfaceConfig contains UI and logging settings
type InterfaceConfig struct {
	LogLevel            string  `json:"log_level"`
	LogPath             string  `json:"log_path"`
	EnableTTS           bool    `json:"enable_tts"`
	ConfidenceThreshold float64 `json:"confidence_threshold"`
	Quiet               bool    `json:"quiet"`
	TopMoves            int     `json:"top_moves"`
}

// PerformanceConfig holds performance-related settings
type PerformanceConfig struct {
	MaxCPUUsage     float64 `json:"max_cpu_usage"`
	MaxMemoryMB     int     `json:"max_memory_mb"`
	GoroutineLimit  int     `json:"goroutine_limit"`
	EnableProfiling bool    `json:"enable_profiling"`
}

// Load reads and parses the configuration file
func Load(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}

	var cfg Config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return nil, err
	}

	return &cfg, nil
}

// Save writes the configuration to a file
func (c *Config) Save(path string) error {
	// Create directory if it doesn't exist
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create config directory: %w", err)
	}

	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// DefaultConfig returns a default configuration
func DefaultConfig() *Config {
	homeDir, _ := os.UserHomeDir()
	dataDir := filepath.Join(homeDir, ".partner", "data")

	return &Config{
		AppName: "P.A.R.T.N.E.R",
		Version: "1.0.0",
		Vision: VisionConfig{
			ScreenRegion: Region{
				X:      100,
				Y:      100,
				Width:  800,
				Height: 800,
			},
			CaptureFPS:    10,
			BoardSize:     8,
			DiffThreshold: 0.05,
		},
		Model: ModelConfig{
			InputSize:    768, // 12 * 8 * 8
			HiddenSize:   256,
			OutputSize:   4096, // 64 * 64 possible moves
			LearningRate: 0.001,
			BatchSize:    32,
			ModelPath:    filepath.Join(dataDir, "models", "chess_cnn.gob"),
		},
		Training: TrainingConfig{
			ReplayBufferSize:         1000,
			DBPath:                   filepath.Join(dataDir, "positions.db"),
			SaveInterval:             100,
			MinSamplesBeforeTraining: 100,
		},
		Interface: InterfaceConfig{
			LogLevel:            "info",
			LogPath:             filepath.Join(homeDir, ".partner", "logs", "partner.log"),
			EnableTTS:           false,
			ConfidenceThreshold: 0.1,
			Quiet:               false,
			TopMoves:            5,
		},
		Performance: PerformanceConfig{
			MaxCPUUsage:     80.0,
			MaxMemoryMB:     2048,
			GoroutineLimit:  100,
			EnableProfiling: false,
		},
	}
}

// LoadOrDefault loads configuration from file, or returns default if not found
func LoadOrDefault(path string) *Config {
	cfg, err := Load(path)
	if err != nil {
		return DefaultConfig()
	}
	return cfg
}

// Validate validates the configuration
func (c *Config) Validate() error {
	if c.Vision.CaptureFPS <= 0 || c.Vision.CaptureFPS > 60 {
		return fmt.Errorf("invalid capture_fps: %d (must be 1-60)", c.Vision.CaptureFPS)
	}

	if c.Model.BatchSize <= 0 {
		return fmt.Errorf("invalid batch_size: %d", c.Model.BatchSize)
	}

	if c.Model.LearningRate <= 0 || c.Model.LearningRate >= 1 {
		return fmt.Errorf("invalid learning_rate: %f", c.Model.LearningRate)
	}

	if c.Performance.MaxCPUUsage <= 0 || c.Performance.MaxCPUUsage > 100 {
		return fmt.Errorf("invalid max_cpu_usage: %f", c.Performance.MaxCPUUsage)
	}

	return nil
}

// EnsureDirectories creates all necessary directories
func (c *Config) EnsureDirectories() error {
	dirs := []string{
		filepath.Dir(c.Interface.LogPath),
		filepath.Dir(c.Model.ModelPath),
		filepath.Dir(c.Training.DBPath),
	}

	for _, dir := range dirs {
		if dir == "" || dir == "." {
			continue
		}
		if err := os.MkdirAll(dir, 0755); err != nil {
			return fmt.Errorf("failed to create directory %s: %w", dir, err)
		}
	}

	return nil
}
