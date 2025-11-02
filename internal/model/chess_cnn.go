package model

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ChessCNN represents a CNN-based chess move prediction model
type ChessCNN struct {
	// Graph
	g *gorgonia.ExprGraph

	// Input: [batch, 12, 8, 8]
	input *gorgonia.Node

	// Convolutional layers
	conv1W *gorgonia.Node // [32, 12, 3, 3]
	conv1B *gorgonia.Node // [32]
	conv2W *gorgonia.Node // [64, 32, 3, 3]
	conv2B *gorgonia.Node // [64]

	// Dense layers
	fc1W *gorgonia.Node // [flatSize, 512]
	fc1B *gorgonia.Node // [512]
	fc2W *gorgonia.Node // [512, 128]
	fc2B *gorgonia.Node // [128]
	fc3W *gorgonia.Node // [128, 4096]
	fc3B *gorgonia.Node // [4096]

	// Output (logits and probabilities)
	logits *gorgonia.Node
	output *gorgonia.Node

	// VM for execution
	vm gorgonia.VM

	// Training flag
	isTraining bool
}

// NewChessCNN creates a new CNN model for chess move prediction
// The model supports variable batch sizes by using batch dimension in input shape
func NewChessCNN() (*ChessCNN, error) {
	return NewChessCNNWithBatchSize(1)
}

// NewChessCNNForInference creates a model for inference and loads weights from a checkpoint
// This allows using a model trained with a different batch size
func NewChessCNNForInference(checkpointPath string) (*ChessCNN, error) {
	// Always create inference model with batch size 1
	model, err := NewChessCNNWithBatchSize(1)
	if err != nil {
		return nil, fmt.Errorf("failed to create inference model: %w", err)
	}
	
	// Load weights from checkpoint
	if err := model.LoadModel(checkpointPath); err != nil {
		model.Close()
		return nil, fmt.Errorf("failed to load checkpoint: %w", err)
	}
	
	return model, nil
}

// NewChessCNNWithBatchSize creates a CNN model with specified batch size
func NewChessCNNWithBatchSize(batchSize int) (*ChessCNN, error) {
	g := gorgonia.NewGraph()

	// Input: [batch, 12, 8, 8]
	input := gorgonia.NewTensor(g, tensor.Float64, 4,
		gorgonia.WithShape(batchSize, 12, 8, 8),
		gorgonia.WithName("input"))

	// Conv1: 12 -> 32 channels, 3x3 kernel
	conv1W := gorgonia.NewTensor(g, tensor.Float64, 4,
		gorgonia.WithShape(32, 12, 3, 3),
		gorgonia.WithName("conv1_w"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	conv1B := gorgonia.NewTensor(g, tensor.Float64, 1,
		gorgonia.WithShape(32),
		gorgonia.WithName("conv1_b"),
		gorgonia.WithInit(gorgonia.Zeroes()))

	// Conv2: 32 -> 64 channels, 3x3 kernel
	conv2W := gorgonia.NewTensor(g, tensor.Float64, 4,
		gorgonia.WithShape(64, 32, 3, 3),
		gorgonia.WithName("conv2_w"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	conv2B := gorgonia.NewTensor(g, tensor.Float64, 1,
		gorgonia.WithShape(64),
		gorgonia.WithName("conv2_b"),
		gorgonia.WithInit(gorgonia.Zeroes()))

	// Build forward pass
	// Conv1: [1, 12, 8, 8] -> [1, 32, 8, 8] (with padding=1)
	conv1, err := gorgonia.Conv2d(input, conv1W, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return nil, fmt.Errorf("conv1 failed: %w", err)
	}
	conv1 = gorgonia.Must(gorgonia.BroadcastAdd(conv1, conv1B, nil, []byte{0, 2, 3}))
	conv1 = gorgonia.Must(gorgonia.Rectify(conv1))

	// Conv2: [1, 32, 8, 8] -> [1, 64, 8, 8] (with padding=1)
	conv2, err := gorgonia.Conv2d(conv1, conv2W, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return nil, fmt.Errorf("conv2 failed: %w", err)
	}
	conv2 = gorgonia.Must(gorgonia.BroadcastAdd(conv2, conv2B, nil, []byte{0, 2, 3}))
	conv2 = gorgonia.Must(gorgonia.Rectify(conv2))

	// Flatten: [batch, 64, 8, 8] -> [batch, 4096]
	flat := gorgonia.Must(gorgonia.Reshape(conv2, tensor.Shape{batchSize, 64 * 8 * 8}))
	flatSize := 64 * 8 * 8 // 4096

	// FC1: 4096 -> 512
	fc1W := gorgonia.NewMatrix(g, tensor.Float64,
		gorgonia.WithShape(flatSize, 512),
		gorgonia.WithName("fc1_w"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	fc1B := gorgonia.NewVector(g, tensor.Float64,
		gorgonia.WithShape(512),
		gorgonia.WithName("fc1_b"),
		gorgonia.WithInit(gorgonia.Zeroes()))

	fc1 := gorgonia.Must(gorgonia.Mul(flat, fc1W))
	fc1 = gorgonia.Must(gorgonia.BroadcastAdd(fc1, fc1B, nil, []byte{0}))
	fc1 = gorgonia.Must(gorgonia.Rectify(fc1))

	// FC2: 512 -> 128
	fc2W := gorgonia.NewMatrix(g, tensor.Float64,
		gorgonia.WithShape(512, 128),
		gorgonia.WithName("fc2_w"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	fc2B := gorgonia.NewVector(g, tensor.Float64,
		gorgonia.WithShape(128),
		gorgonia.WithName("fc2_b"),
		gorgonia.WithInit(gorgonia.Zeroes()))

	fc2 := gorgonia.Must(gorgonia.Mul(fc1, fc2W))
	fc2 = gorgonia.Must(gorgonia.BroadcastAdd(fc2, fc2B, nil, []byte{0}))
	fc2 = gorgonia.Must(gorgonia.Rectify(fc2))

	// FC3: 128 -> 4096 (64x64 from-to pairs)
	fc3W := gorgonia.NewMatrix(g, tensor.Float64,
		gorgonia.WithShape(128, 4096),
		gorgonia.WithName("fc3_w"),
		gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	fc3B := gorgonia.NewVector(g, tensor.Float64,
		gorgonia.WithShape(4096),
		gorgonia.WithName("fc3_b"),
		gorgonia.WithInit(gorgonia.Zeroes()))

	logits := gorgonia.Must(gorgonia.Mul(fc2, fc3W))
	logits = gorgonia.Must(gorgonia.BroadcastAdd(logits, fc3B, nil, []byte{0}))

	// Softmax for probabilities
	output := gorgonia.Must(gorgonia.SoftMax(logits))

	// Create VM
	vm := gorgonia.NewTapeMachine(g)

	return &ChessCNN{
		g:      g,
		input:  input,
		conv1W: conv1W,
		conv1B: conv1B,
		conv2W: conv2W,
		conv2B: conv2B,
		fc1W:   fc1W,
		fc1B:   fc1B,
		fc2W:   fc2W,
		fc2B:   fc2B,
		fc3W:   fc3W,
		fc3B:   fc3B,
		logits: logits,
		output: output,
		vm:     vm,
	}, nil
}

// Predict performs inference and returns top K moves with probabilities
func (cnn *ChessCNN) Predict(boardTensor [12][8][8]float32, topK int) ([]MovePrediction, error) {
	// Convert float32 to float64 and flatten
	inputData := make([]float64, 12*8*8)
	idx := 0
	for c := 0; c < 12; c++ {
		for r := 0; r < 8; r++ {
			for f := 0; f < 8; f++ {
				inputData[idx] = float64(boardTensor[c][r][f])
				idx++
			}
		}
	}

	// Create tensor
	inputTensor := tensor.New(
		tensor.WithShape(1, 12, 8, 8),
		tensor.WithBacking(inputData),
	)

	// Set input
	if err := gorgonia.Let(cnn.input, inputTensor); err != nil {
		return nil, fmt.Errorf("failed to set input: %w", err)
	}

	// Run forward pass
	if err := cnn.vm.RunAll(); err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	// Get output probabilities
	outputValue := cnn.output.Value()
	if outputValue == nil {
		return nil, fmt.Errorf("output is nil")
	}

	probs := outputValue.Data().([]float64)

	// Reset VM for next run
	cnn.vm.Reset()

	// Get top K moves
	if topK <= 0 || topK > len(probs) {
		topK = 3
	}

	return getTopKPredictions(probs, topK), nil
}

// MovePrediction represents a predicted move with probability
type MovePrediction struct {
	FromSquare  int     // 0-63
	ToSquare    int     // 0-63
	Probability float64
	MoveIndex   int // Combined index (from*64 + to)
}

// getTopKPredictions returns top K moves sorted by probability
func getTopKPredictions(probs []float64, k int) []MovePrediction {
	predictions := make([]MovePrediction, len(probs))
	for i, p := range probs {
		fromSquare := i / 64
		toSquare := i % 64
		predictions[i] = MovePrediction{
			FromSquare:  fromSquare,
			ToSquare:    toSquare,
			Probability: p,
			MoveIndex:   i,
		}
	}

	// Selection sort for top K
	for i := 0; i < k && i < len(predictions); i++ {
		maxIdx := i
		for j := i + 1; j < len(predictions); j++ {
			if predictions[j].Probability > predictions[maxIdx].Probability {
				maxIdx = j
			}
		}
		predictions[i], predictions[maxIdx] = predictions[maxIdx], predictions[i]
	}

	return predictions[:k]
}

// Learnables returns all trainable parameters
func (cnn *ChessCNN) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{
		cnn.conv1W, cnn.conv1B,
		cnn.conv2W, cnn.conv2B,
		cnn.fc1W, cnn.fc1B,
		cnn.fc2W, cnn.fc2B,
		cnn.fc3W, cnn.fc3B,
	}
}

// SaveModel saves model weights to file
func (cnn *ChessCNN) SaveModel(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer f.Close()

	encoder := gob.NewEncoder(f)

	// Save metadata
	metadata := ModelMetadata{
		Version:     "1.0",
		ModelType:   "ChessCNN",
		InputShape:  []int{12, 8, 8},
		OutputShape: []int{4096},
	}
	if err := encoder.Encode(metadata); err != nil {
		return fmt.Errorf("failed to encode metadata: %w", err)
	}

	// Save weights
	weights := []*gorgonia.Node{
		cnn.conv1W, cnn.conv1B,
		cnn.conv2W, cnn.conv2B,
		cnn.fc1W, cnn.fc1B,
		cnn.fc2W, cnn.fc2B,
		cnn.fc3W, cnn.fc3B,
	}

	for i, w := range weights {
		val := w.Value()
		if val == nil {
			return fmt.Errorf("weight %d has nil value", i)
		}

		data := val.Data().([]float64)
		shape := val.Shape()

		if err := encoder.Encode(shape); err != nil {
			return fmt.Errorf("failed to encode weight %d shape: %w", i, err)
		}
		if err := encoder.Encode(data); err != nil {
			return fmt.Errorf("failed to encode weight %d data: %w", i, err)
		}
	}

	return nil
}

// LoadModel loads model weights from file
func (cnn *ChessCNN) LoadModel(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	decoder := gob.NewDecoder(f)

	// Load metadata
	var metadata ModelMetadata
	if err := decoder.Decode(&metadata); err != nil {
		return fmt.Errorf("failed to decode metadata: %w", err)
	}

	// Validate metadata
	if metadata.ModelType != "ChessCNN" {
		return fmt.Errorf("invalid model type: %s", metadata.ModelType)
	}

	// Load weights
	weights := []*gorgonia.Node{
		cnn.conv1W, cnn.conv1B,
		cnn.conv2W, cnn.conv2B,
		cnn.fc1W, cnn.fc1B,
		cnn.fc2W, cnn.fc2B,
		cnn.fc3W, cnn.fc3B,
	}

	for i, w := range weights {
		var shape tensor.Shape
		var data []float64

		if err := decoder.Decode(&shape); err != nil {
			return fmt.Errorf("failed to decode weight %d shape: %w", i, err)
		}
		if err := decoder.Decode(&data); err != nil {
			return fmt.Errorf("failed to decode weight %d data: %w", i, err)
		}

		t := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data))
		if err := gorgonia.Let(w, t); err != nil {
			return fmt.Errorf("failed to set weight %d: %w", i, err)
		}
	}

	return nil
}

// Close cleans up resources
func (cnn *ChessCNN) Close() error {
	if cnn.vm != nil {
		cnn.vm.Close()
	}
	return nil
}

// ModelMetadata stores model information
type ModelMetadata struct {
	Version     string
	ModelType   string
	InputShape  []int
	OutputShape []int
}

// ComputeLoss computes categorical cross-entropy loss
func (cnn *ChessCNN) ComputeLoss(target *gorgonia.Node) (*gorgonia.Node, error) {
	// Cross-entropy loss: -sum(target * log(output))
	logProbs := gorgonia.Must(gorgonia.Log(cnn.output))
	loss := gorgonia.Must(gorgonia.HadamardProd(target, logProbs))
	loss = gorgonia.Must(gorgonia.Sum(loss))
	loss = gorgonia.Must(gorgonia.Neg(loss))

	return loss, nil
}

// ClipGradients applies gradient clipping for stability
func ClipGradients(nodes gorgonia.Nodes, maxNorm float64) error {
	for _, n := range nodes {
		grad, err := n.Grad()
		if err != nil {
			continue // Skip if no gradient
		}

		if grad == nil {
			continue
		}

		gradData := grad.Data().([]float64)

		// Compute L2 norm
		var norm float64
		for _, v := range gradData {
			norm += v * v
		}
		norm = math.Sqrt(norm)

		// Clip if necessary
		if norm > maxNorm {
			scale := maxNorm / norm
			for i := range gradData {
				gradData[i] *= scale
			}
		}
	}

	return nil
}

// ConvertMoveToTarget creates one-hot target vector
func ConvertMoveToTarget(fromSquare, toSquare int) ([]float64, error) {
	if fromSquare < 0 || fromSquare >= 64 || toSquare < 0 || toSquare >= 64 {
		return nil, fmt.Errorf("invalid squares: from=%d, to=%d", fromSquare, toSquare)
	}

	moveIndex := fromSquare*64 + toSquare
	target := make([]float64, 4096)
	target[moveIndex] = 1.0

	return target, nil
}
