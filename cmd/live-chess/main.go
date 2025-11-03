package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/thyrook/partner/internal/model"
	"github.com/thyrook/partner/internal/vision"
)

func main() {
	modelPath := flag.String("model", "data/models/chess_cnn.bin", "CNN model path")
	x := flag.Int("x", 100, "Capture X")
	y := flag.Int("y", 100, "Capture Y")
	width := flag.Int("width", 800, "Capture width")
	height := flag.Int("height", 800, "Capture height")
	fps := flag.Int("fps", 2, "Frames per second")
	topK := flag.Int("top", 5, "Top moves to show")
	flag.Parse()

	fmt.Println("╔═══════════════════════════════════════════════════════════╗")
	fmt.Println("║  P.A.R.T.N.E.R Live Chess Analysis                        ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════╝")
	fmt.Println()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	config := &vision.Config{
		CaptureRegion: vision.CaptureRegion{X: *x, Y: *y, Width: *width, Height: *height},
		BoardSize:     8,
		SquareSize:    100,
		FPS:           *fps,
		DiffThreshold: 0.05,
		UseGrayscale:  false,
		ConfidenceMin: 0.5,
	}

	fmt.Printf("Vision: %dx%d at (%d,%d), FPS=%d\n", *width, *height, *x, *y, *fps)

	fmt.Printf("Loading model: %s\n", *modelPath)
	cnn, err := loadModel(*modelPath)
	if err != nil {
		log.Fatalf("Model load failed: %v", err)
	}
	defer cnn.Close()
	fmt.Println("✅ Model loaded")
	fmt.Println()

	tensorChan := make(chan vision.BoardStateTensor, 10)

	pipeline, err := vision.NewPipeline(config, tensorChan)
	if err != nil {
		log.Fatalf("Pipeline failed: %v", err)
	}

	if err := pipeline.Start(); err != nil {
		log.Fatalf("Start failed: %v", err)
	}
	defer pipeline.Stop()

	fmt.Println("🎥 Monitoring board (Ctrl+C to stop)")
	fmt.Println()

	run(pipeline, cnn, tensorChan, *topK, sig)
}

func loadModel(path string) (*model.ChessCNN, error) {
	if _, err := os.Stat(path); err == nil {
		cnn, err := model.NewChessCNN()
		if err != nil {
			return nil, err
		}
		if err := cnn.LoadModel(path); err != nil {
			cnn.Close()
			return nil, err
		}
		return cnn, nil
	}
	log.Println("Warning: No trained model, creating new")
	return model.NewChessCNN()
}

func run(p *vision.Pipeline, cnn *model.ChessCNN, ch <-chan vision.BoardStateTensor, topK int, sig chan os.Signal) {
	boards := 0
	lastTime := time.Now()

	for {
		select {
		case <-sig:
			fmt.Println("\n🛑 Stopping...")
			stats := p.GetStats()
			fmt.Printf("Frames: %d | Boards: %d | Changes: %d\n",
				stats.FramesProcessed, boards, stats.ChangesDetected)
			return

		case t, ok := <-ch:
			if !ok {
				return
			}

			boards++

			if time.Since(lastTime) < 2*time.Second {
				continue
			}
			lastTime = time.Now()

			if err := vision.ValidateBoardTensor(t.Tensor); err != nil {
				log.Printf("⚠️  Invalid: %v", err)
				continue
			}

			preds, err := cnn.Predict(t.Tensor, topK)
			if err != nil {
				log.Printf("Prediction failed: %v", err)
				continue
			}

			fmt.Printf("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
			fmt.Printf("📋 Board #%d at %s\n", boards, time.Unix(t.Timestamp, 0).Format("15:04:05"))
			fmt.Println(vision.PrintBoardTensor(t.Tensor))

			fmt.Printf("\n🎯 Top %d Moves:\n", len(preds))
			for i, p := range preds {
				from := sq2alg(p.FromSquare)
				to := sq2alg(p.ToSquare)
				bar := makeBar(p.Probability, 15)
				fmt.Printf("  %d. %s→%s %s %.1f%%\n", i+1, from, to, bar, p.Probability*100)
			}
			fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
		}
	}
}

func sq2alg(sq int) string {
	if sq < 0 || sq >= 64 {
		return "??"
	}
	return fmt.Sprintf("%c%d", 'a'+(sq%8), 8-(sq/8))
}

func makeBar(prob float64, w int) string {
	filled := int(prob * float64(w))
	if filled > w {
		filled = w
	}
	bar := "["
	for i := 0; i < w; i++ {
		if i < filled {
			bar += "█"
		} else {
			bar += "░"
		}
	}
	return bar + "]"
}
