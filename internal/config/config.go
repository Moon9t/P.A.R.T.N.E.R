package config

import (
	"encoding/json"
	"os"
)

// Config represents the application configuration
type Config struct {
	Vision    VisionConfig    `json:"vision"`
	Model     ModelConfig     `json:"model"`
	Training  TrainingConfig  `json:"training"`
	Interface InterfaceConfig `json:"interface"`
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
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}
