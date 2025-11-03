package main

import (
"flag"
"fmt"
"log"
"os"
"os/signal"
"strings"
"syscall"
"time"

"github.com/thyrook/partner/internal/data"
"github.com/thyrook/partner/internal/model"
"github.com/thyrook/partner/internal/training"
)

func main() {
	modelPath := flag.String("model", "data/models/chess_cnn.bin", "CNN model path")
	datasetPath := flag.String("dataset", "data/positions.db", "Dataset path")
	replayDB := flag.String("replay-db", "data/self_improvement.db", "Replay DB")
	maxObs := flag.Int("observations", 100, "Max observations")
	flag.Parse()

	fmt.Println("╔═══════════════════════════════════════════════════════════╗")
	fmt.Println("║  P.A.R.T.N.E.R Self-Improvement System (REAL)             ║")
	fmt.Println("╚═══════════════════════════════════════════════════════════╝")
	fmt.Println()

	sig := make(chan os.Signal, 1)
	signal.Notify(sig, syscall.SIGINT, syscall.SIGTERM)

	log.Println("Loading model...")
	cnn, err := loadModel(*modelPath)
	if err != nil {
		log.Fatalf("Model load failed: %v", err)
	}
	defer cnn.Close()

	log.Println("Initializing self-improver...")
	cfg := training.ImproverConfig{
		BufferSize:         1000,
		MinSamplesForTrain: 50,
		BatchSize:          16,
		LearningRate:       0.0001,
		TrainIntervalSec:   60,
		UseRewardWeighting: true,
		DBPath:             *replayDB,
		JSONLDir:           "data/replays",
		AutoSave:           true,
	}

	improver, err := training.NewSelfImprover(cnn, cfg)
	if err != nil {
		log.Fatalf("Improver init failed: %v", err)
	}
	defer improver.Close()

	log.Printf("Loading dataset: %s", *datasetPath)
	ds, err := data.NewDataset(*datasetPath)
	if err != nil {
		log.Fatalf("Dataset load failed: %v", err)
	}
	defer ds.Close()

	count, _ := ds.Count()
	fmt.Printf("\n✓ Ready: %d positions loaded\n\n", count)

	run(improver, cnn, ds, *maxObs, sig)
}

func loadModel(path string) (*model.ChessCNN, error) {
	if _, err := os.Stat(path); err == nil {
		cnn, err := model.NewChessCNN()
		if err == nil {
			err = cnn.LoadModel(path)
		}
		if err != nil {
			return nil, err
		}
		return cnn, nil
	}
	return model.NewChessCNN()
}

func run(imp *training.SelfImprover, cnn *model.ChessCNN, ds *data.Dataset, maxObs int, sig chan os.Signal) {
	fmt.Println("🚀 Self-improvement active...")
	fmt.Println()
	obs := 0
	total, _ := ds.Count()

	for obs < maxObs {
		select {
		case <-sig:
			fmt.Println("\n🛑 Interrupted")
			showStats(imp)
			return
		default:
		}

		entries, err := ds.LoadBatch((obs/5)%((total+4)/5)*5, 5)
		if err != nil || len(entries) == 0 {
			continue
		}

		for _, e := range entries {
			board, _ := data.FlatArrayToTensor(e.StateTensor)
			preds, _ := cnn.Predict(board, 3)
			
			if len(preds) == 0 {
				continue
			}

			pred := preds[0]
			actual := training.Move{
				Index:      e.FromSquare*64 + e.ToSquare,
				Notation:   fmt.Sprintf("%d→%d", e.FromSquare, e.ToSquare),
				FromSquare: fmt.Sprintf("%d", e.FromSquare),
				ToSquare:   fmt.Sprintf("%d", e.ToSquare),
				Confidence: 1.0,
			}

			topK := make([]training.Move, len(preds))
			for i, p := range preds {
				topK[i] = training.Move{
					Index:      p.MoveIndex,
					Notation:   fmt.Sprintf("%d→%d", p.FromSquare, p.ToSquare),
					FromSquare: fmt.Sprintf("%d", p.FromSquare),
					ToSquare:   fmt.Sprintf("%d", p.ToSquare),
					Confidence: p.Probability,
				}
			}

			imp.ObservePrediction(board, topK[0], actual, topK, pred.Probability)
			obs++

			if obs%10 == 0 {
				fmt.Printf("📊 Observations: %d\n", obs)
				showStats(imp)
			}
		}

		time.Sleep(500 * time.Millisecond)
	}

	fmt.Printf("\n✅ Completed %d observations\n", obs)
	showStats(imp)
}

func showStats(imp *training.SelfImprover) {
	s := imp.GetStats()
	b := imp.GetBufferStats()
	m := imp.CalculateImprovement()

	fmt.Println(strings.Repeat("─", 50))
	fmt.Printf("Cycles: %d | Buffer: %d\n", s.TotalCycles, b.TotalEntries)
	fmt.Printf("Accuracy: %.1f%% → %.1f%% (Δ%+.1f%%)\n",
s.BaselineAccuracy*100, s.CurrentAccuracy*100, m.AbsoluteImprovement*100)
	fmt.Println(strings.Repeat("─", 50))
}
