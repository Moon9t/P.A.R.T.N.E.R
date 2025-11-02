package vision

import (
	"errors"
	"fmt"
	"image"
	"image/color"
	"sync"
	"time"

	"github.com/kbinani/screenshot"
	"gocv.io/x/gocv"
)

// Capturer handles screen capture and frame processing
type Capturer struct {
	region        image.Rectangle
	boardSize     int
	diffThreshold float64
	lastFrame     *gocv.Mat
	mu            sync.Mutex
}

// NewCapturer creates a new vision capturer
func NewCapturer(x, y, width, height, boardSize int, diffThreshold float64) *Capturer {
	return &Capturer{
		region:        image.Rect(x, y, x+width, y+height),
		boardSize:     boardSize,
		diffThreshold: diffThreshold,
	}
}

// CaptureFrame captures the current screen region and returns a processed mat
func (c *Capturer) CaptureFrame() (*gocv.Mat, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	// Capture screen region
	img, err := screenshot.CaptureRect(c.region)
	if err != nil {
		return nil, fmt.Errorf("failed to capture screen: %w", err)
	}

	// Convert image.Image to gocv.Mat
	mat, err := c.imageToMat(img)
	if err != nil {
		return nil, fmt.Errorf("failed to convert image to mat: %w", err)
	}

	return mat, nil
}

// ProcessFrame converts a frame to normalized board tensor
func (c *Capturer) ProcessFrame(frame *gocv.Mat) ([]float64, error) {
	if frame.Empty() {
		return nil, errors.New("empty frame")
	}

	// Convert to grayscale
	gray := gocv.NewMat()
	defer gray.Close()
	gocv.CvtColor(*frame, &gray, gocv.ColorBGRAToGray)

	// Resize to board size
	resized := gocv.NewMat()
	defer resized.Close()
	gocv.Resize(gray, &resized, image.Pt(c.boardSize, c.boardSize), 0, 0, gocv.InterpolationLinear)

	// Normalize to [0, 1]
	normalized := gocv.NewMat()
	defer normalized.Close()
	resized.ConvertTo(&normalized, gocv.MatTypeCV32F)
	normalized.DivideFloat(255.0)

	// Convert to float64 slice
	data, err := c.matToFloat64Slice(&normalized)
	if err != nil {
		return nil, fmt.Errorf("failed to convert mat to slice: %w", err)
	}

	return data, nil
}

// DetectChange checks if the frame has changed significantly from the last frame
func (c *Capturer) DetectChange(frame *gocv.Mat) (bool, float64, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.lastFrame == nil {
		// First frame, save it
		c.lastFrame = &gocv.Mat{}
		frame.CopyTo(c.lastFrame)
		return true, 100.0, nil
	}

	// Convert both frames to grayscale
	gray1 := gocv.NewMat()
	defer gray1.Close()
	gray2 := gocv.NewMat()
	defer gray2.Close()

	gocv.CvtColor(*c.lastFrame, &gray1, gocv.ColorBGRAToGray)
	gocv.CvtColor(*frame, &gray2, gocv.ColorBGRAToGray)

	// Compute absolute difference
	diff := gocv.NewMat()
	defer diff.Close()
	gocv.AbsDiff(gray1, gray2, &diff)

	// Calculate mean difference
	mean := diff.Mean()
	meanVal := mean.Val1

	changed := meanVal > c.diffThreshold

	if changed {
		// Update last frame
		frame.CopyTo(c.lastFrame)
	}

	return changed, meanVal, nil
}

// Close releases resources
func (c *Capturer) Close() {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.lastFrame != nil {
		c.lastFrame.Close()
		c.lastFrame = nil
	}
}

// imageToMat converts image.Image to gocv.Mat
func (c *Capturer) imageToMat(img image.Image) (*gocv.Mat, error) {
	bounds := img.Bounds()
	width := bounds.Dx()
	height := bounds.Dy()

	// Create a new mat
	mat := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC4)

	// Copy pixel data
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			// Convert from uint32 (0-65535) to uint8 (0-255)
			mat.SetUCharAt(y, x*4+0, uint8(b>>8))
			mat.SetUCharAt(y, x*4+1, uint8(g>>8))
			mat.SetUCharAt(y, x*4+2, uint8(r>>8))
			mat.SetUCharAt(y, x*4+3, uint8(a>>8))
		}
	}

	return &mat, nil
}

// matToFloat64Slice converts a CV32F mat to float64 slice
func (c *Capturer) matToFloat64Slice(mat *gocv.Mat) ([]float64, error) {
	if mat.Empty() {
		return nil, errors.New("empty mat")
	}

	rows := mat.Rows()
	cols := mat.Cols()
	data := make([]float64, rows*cols)

	idx := 0
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			data[idx] = float64(mat.GetFloatAt(i, j))
			idx++
		}
	}

	return data, nil
}

// CaptureLoop continuously captures frames at the specified FPS
func (c *Capturer) CaptureLoop(fps int, frameChan chan<- *gocv.Mat, stopChan <-chan struct{}) {
	ticker := time.NewTicker(time.Second / time.Duration(fps))
	defer ticker.Stop()

	for {
		select {
		case <-stopChan:
			return
		case <-ticker.C:
			frame, err := c.CaptureFrame()
			if err != nil {
				// Log error but continue
				continue
			}

			// Try to send frame, skip if channel is full
			select {
			case frameChan <- frame:
			default:
				frame.Close()
			}
		}
	}
}

// BoardState represents the current state of the board
type BoardState struct {
	Grid      []float64
	Timestamp time.Time
	Changed   bool
	DiffScore float64
}

// ExtractBoardState captures and processes the current board state
func (c *Capturer) ExtractBoardState() (*BoardState, error) {
	frame, err := c.CaptureFrame()
	if err != nil {
		return nil, err
	}
	defer frame.Close()

	// Check for changes
	changed, diffScore, err := c.DetectChange(frame)
	if err != nil {
		return nil, err
	}

	// Process frame to normalized grid
	grid, err := c.ProcessFrame(frame)
	if err != nil {
		return nil, err
	}

	return &BoardState{
		Grid:      grid,
		Timestamp: time.Now(),
		Changed:   changed,
		DiffScore: diffScore,
	}, nil
}

// ValidateCapture checks if the capture system is working
func (c *Capturer) ValidateCapture() error {
	state, err := c.ExtractBoardState()
	if err != nil {
		return fmt.Errorf("capture validation failed: %w", err)
	}

	if len(state.Grid) != c.boardSize*c.boardSize {
		return fmt.Errorf("invalid grid size: expected %d, got %d",
			c.boardSize*c.boardSize, len(state.Grid))
	}

	return nil
}

// VisualizeBoard creates a simple ASCII representation of the board
func VisualizeBoard(grid []float64, size int) string {
	if len(grid) != size*size {
		return "Invalid grid size"
	}

	result := "\n"
	for i := 0; i < size; i++ {
		for j := 0; j < size; j++ {
			val := grid[i*size+j]
			if val < 0.3 {
				result += "█ "
			} else if val < 0.7 {
				result += "▒ "
			} else {
				result += "░ "
			}
		}
		result += "\n"
	}
	return result
}

// Screenshot utility for debugging
type Screenshot struct {
	Image     image.Image
	Timestamp time.Time
}

// SaveScreenshot captures and returns a screenshot
func SaveScreenshot(bounds image.Rectangle) (*Screenshot, error) {
	img, err := screenshot.CaptureRect(bounds)
	if err != nil {
		return nil, err
	}

	return &Screenshot{
		Image:     img,
		Timestamp: time.Now(),
	}, nil
}

// ConvertRGBAToGray converts RGBA image to grayscale
func ConvertRGBAToGray(img image.Image) *image.Gray {
	bounds := img.Bounds()
	gray := image.NewGray(bounds)

	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			originalColor := img.At(x, y)
			grayColor := color.GrayModel.Convert(originalColor)
			gray.Set(x, y, grayColor)
		}
	}

	return gray
}
