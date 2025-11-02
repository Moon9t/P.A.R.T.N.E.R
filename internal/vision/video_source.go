package vision

import (
	"fmt"
	"time"

	"gocv.io/x/gocv"
)

// VideoSource provides frames from a video file for replay/testing
type VideoSource struct {
	video      *gocv.VideoCapture
	fps        float64
	frameCount int
	currentFrame int
}

// NewVideoSource opens a video file for playback
func NewVideoSource(videoPath string) (*VideoSource, error) {
	video, err := gocv.VideoCaptureFile(videoPath)
	if err != nil {
		return nil, fmt.Errorf("failed to open video file: %w", err)
	}

	if !video.IsOpened() {
		return nil, fmt.Errorf("video file not opened")
	}

	fps := video.Get(gocv.VideoCaptureFPS)
	frameCount := int(video.Get(gocv.VideoCaptureFrameCount))

	return &VideoSource{
		video:      video,
		fps:        fps,
		frameCount: frameCount,
		currentFrame: 0,
	}, nil
}

// ReadFrame reads the next frame from the video
func (vs *VideoSource) ReadFrame() (*gocv.Mat, error) {
	if vs.video == nil {
		return nil, fmt.Errorf("video source not initialized")
	}

	mat := gocv.NewMat()
	if !vs.video.Read(&mat) {
		mat.Close()
		return nil, fmt.Errorf("failed to read frame or end of video")
	}

	if mat.Empty() {
		mat.Close()
		return nil, fmt.Errorf("empty frame")
	}

	vs.currentFrame++
	return &mat, nil
}

// GetFPS returns the video's frames per second
func (vs *VideoSource) GetFPS() float64 {
	return vs.fps
}

// GetFrameCount returns total number of frames
func (vs *VideoSource) GetFrameCount() int {
	return vs.frameCount
}

// GetCurrentFrame returns the current frame number
func (vs *VideoSource) GetCurrentFrame() int {
	return vs.currentFrame
}

// GetProgress returns playback progress (0-1)
func (vs *VideoSource) GetProgress() float64 {
	if vs.frameCount == 0 {
		return 0
	}
	return float64(vs.currentFrame) / float64(vs.frameCount)
}

// Seek jumps to a specific frame number
func (vs *VideoSource) Seek(frameNum int) error {
	if vs.video == nil {
		return fmt.Errorf("video source not initialized")
	}

	vs.video.Set(gocv.VideoCapturePosFrames, float64(frameNum))
	vs.currentFrame = frameNum
	return nil
}

// Reset restarts the video from the beginning
func (vs *VideoSource) Reset() error {
	return vs.Seek(0)
}

// Close releases video resources
func (vs *VideoSource) Close() error {
	if vs.video != nil {
		err := vs.video.Close()
		vs.video = nil
		return err
	}
	return nil
}

// PlaybackLoop continuously reads frames and sends them to a channel
func (vs *VideoSource) PlaybackLoop(frameChan chan<- *gocv.Mat, stopChan <-chan struct{}, realtime bool) error {
	if vs.video == nil {
		return fmt.Errorf("video source not initialized")
	}

	var frameDuration time.Duration
	if realtime && vs.fps > 0 {
		frameDuration = time.Duration(float64(time.Second) / vs.fps)
	}

	ticker := time.NewTicker(frameDuration)
	if frameDuration == 0 {
		ticker.Stop() // No delay if not realtime
	}
	defer ticker.Stop()

	for {
		select {
		case <-stopChan:
			return nil
		default:
			// Read frame
			frame, err := vs.ReadFrame()
			if err != nil {
				return err // End of video or error
			}

			// Wait for next frame time if realtime
			if realtime && frameDuration > 0 {
				<-ticker.C
			}

			// Send frame
			select {
			case frameChan <- frame:
			case <-stopChan:
				frame.Close()
				return nil
			}
		}
	}
}

// FrameSource interface for different frame sources (screen capture, video file, etc.)
type FrameSource interface {
	ReadFrame() (*gocv.Mat, error)
	Close() error
}

// LiveSource wraps Capturer to implement FrameSource
type LiveSource struct {
	capturer *Capturer
}

// NewLiveSource creates a frame source from screen capture
func NewLiveSource(capturer *Capturer) *LiveSource {
	return &LiveSource{capturer: capturer}
}

// ReadFrame captures a frame from the screen
func (ls *LiveSource) ReadFrame() (*gocv.Mat, error) {
	return ls.capturer.CaptureFrame()
}

// Close releases resources
func (ls *LiveSource) Close() error {
	ls.capturer.Close()
	return nil
}

// VideoInfo holds metadata about a video
type VideoInfo struct {
	FPS        float64
	FrameCount int
	Width      int
	Height     int
	Duration   time.Duration
}

// GetVideoInfo extracts metadata from a video file
func GetVideoInfo(videoPath string) (*VideoInfo, error) {
	video, err := gocv.VideoCaptureFile(videoPath)
	if err != nil {
		return nil, err
	}
	defer video.Close()

	if !video.IsOpened() {
		return nil, fmt.Errorf("failed to open video")
	}

	fps := video.Get(gocv.VideoCaptureFPS)
	frameCount := int(video.Get(gocv.VideoCaptureFrameCount))
	width := int(video.Get(gocv.VideoCaptureFrameWidth))
	height := int(video.Get(gocv.VideoCaptureFrameHeight))

	var duration time.Duration
	if fps > 0 {
		duration = time.Duration(float64(frameCount)/fps) * time.Second
	}

	return &VideoInfo{
		FPS:        fps,
		FrameCount: frameCount,
		Width:      width,
		Height:     height,
		Duration:   duration,
	}, nil
}

// String returns a formatted string of video info
func (vi *VideoInfo) String() string {
	return fmt.Sprintf(
		"Video Info:\n"+
			"  Resolution: %dx%d\n"+
			"  FPS: %.2f\n"+
			"  Frames: %d\n"+
			"  Duration: %v\n",
		vi.Width, vi.Height,
		vi.FPS,
		vi.FrameCount,
		vi.Duration,
	)
}
