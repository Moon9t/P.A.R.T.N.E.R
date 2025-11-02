package vision

import (
	"encoding/json"
	"fmt"
	"image"
	"os"
)

// Config holds vision system configuration
type Config struct {
	// Screen capture settings
	CaptureRegion CaptureRegion `json:"capture_region"`

	// Board detection settings
	BoardSize     int  `json:"board_size"`      // Number of squares (8 for chess)
	SquareSize    int  `json:"square_size"`     // Pixels per square
	UseGrayscale  bool `json:"use_grayscale"`   // Use grayscale processing
	ConfidenceMin float64 `json:"confidence_min"` // Minimum confidence threshold

	// Change detection settings
	DiffThreshold float64 `json:"diff_threshold"` // Threshold for detecting frame changes
	FPS           int     `json:"fps"`            // Capture frames per second

	// Color thresholds (optional, uses defaults if not set)
	ColorThresholds *ColorThresholds `json:"color_thresholds,omitempty"`
}

// CaptureRegion defines the screen area to capture
type CaptureRegion struct {
	X      int `json:"x"`
	Y      int `json:"y"`
	Width  int `json:"width"`
	Height int `json:"height"`
}

// ToRectangle converts CaptureRegion to image.Rectangle
func (cr CaptureRegion) ToRectangle() image.Rectangle {
	return image.Rect(cr.X, cr.Y, cr.X+cr.Width, cr.Y+cr.Height)
}

// DefaultConfig returns default vision configuration
func DefaultConfig() *Config {
	return &Config{
		CaptureRegion: CaptureRegion{
			X:      100,
			Y:      100,
			Width:  800,
			Height: 800,
		},
		BoardSize:     8,
		SquareSize:    100,
		UseGrayscale:  true,
		ConfidenceMin: 0.5,
		DiffThreshold: 10.0,
		FPS:           2, // 2 frames per second for chess
	}
}

// LoadConfig loads configuration from a JSON file
func LoadConfig(path string) (*Config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("failed to read config file: %w", err)
	}

	var config Config
	if err := json.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("failed to parse config: %w", err)
	}

	// Validate config
	if err := config.Validate(); err != nil {
		return nil, fmt.Errorf("invalid config: %w", err)
	}

	return &config, nil
}

// SaveConfig saves configuration to a JSON file
func (c *Config) SaveConfig(path string) error {
	data, err := json.MarshalIndent(c, "", "  ")
	if err != nil {
		return fmt.Errorf("failed to marshal config: %w", err)
	}

	if err := os.WriteFile(path, data, 0644); err != nil {
		return fmt.Errorf("failed to write config file: %w", err)
	}

	return nil
}

// Validate checks if the configuration is valid
func (c *Config) Validate() error {
	if c.CaptureRegion.Width <= 0 || c.CaptureRegion.Height <= 0 {
		return fmt.Errorf("invalid capture region dimensions")
	}

	if c.BoardSize < 1 || c.BoardSize > 16 {
		return fmt.Errorf("invalid board size: %d (must be 1-16)", c.BoardSize)
	}

	if c.SquareSize < 10 || c.SquareSize > 500 {
		return fmt.Errorf("invalid square size: %d (must be 10-500)", c.SquareSize)
	}

	if c.ConfidenceMin < 0 || c.ConfidenceMin > 1 {
		return fmt.Errorf("invalid confidence minimum: %f (must be 0-1)", c.ConfidenceMin)
	}

	if c.DiffThreshold < 0 || c.DiffThreshold > 255 {
		return fmt.Errorf("invalid diff threshold: %f (must be 0-255)", c.DiffThreshold)
	}

	if c.FPS < 1 || c.FPS > 60 {
		return fmt.Errorf("invalid FPS: %d (must be 1-60)", c.FPS)
	}

	return nil
}

// String returns a string representation of the config
func (c *Config) String() string {
	return fmt.Sprintf(
		"Vision Config:\n"+
			"  Capture Region: (%d,%d) %dx%d\n"+
			"  Board Size: %dx%d\n"+
			"  Square Size: %dpx\n"+
			"  Grayscale: %v\n"+
			"  Confidence Min: %.2f\n"+
			"  Diff Threshold: %.1f\n"+
			"  FPS: %d\n",
		c.CaptureRegion.X, c.CaptureRegion.Y,
		c.CaptureRegion.Width, c.CaptureRegion.Height,
		c.BoardSize, c.BoardSize,
		c.SquareSize,
		c.UseGrayscale,
		c.ConfidenceMin,
		c.DiffThreshold,
		c.FPS,
	)
}
