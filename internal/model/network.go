package model

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

// ChessNet represents the neural network for chess move prediction
type ChessNet struct {
	// Graph
	g *gorgonia.ExprGraph

	// Input
	input *gorgonia.Node

	// Convolutional layers
	conv1W *gorgonia.Node
	conv1B *gorgonia.Node
	conv2W *gorgonia.Node
	conv2B *gorgonia.Node

	// Dense layers
	fc1W *gorgonia.Node
	fc1B *gorgonia.Node
	fc2W *gorgonia.Node
	fc2B *gorgonia.Node
	fc3W *gorgonia.Node
	fc3B *gorgonia.Node

	// Output
	output *gorgonia.Node

	// VM for execution
	vm gorgonia.VM

	// Configuration
	inputSize  int
	hiddenSize int
	outputSize int
}

// NewChessNet creates a new chess neural network
func NewChessNet(inputSize, hiddenSize, outputSize int) (*ChessNet, error) {
	g := gorgonia.NewGraph()

	// Input: batch x 1 x 8 x 8 (assuming 8x8 board)
	boardSize := int(math.Sqrt(float64(inputSize)))
	input := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(1, 1, boardSize, boardSize), gorgonia.WithName("input"))

	// Conv1: 1 -> 16 channels, 3x3 kernel
	conv1W := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(16, 1, 3, 3), gorgonia.WithName("conv1_w"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	conv1B := gorgonia.NewTensor(g, tensor.Float64, 1, gorgonia.WithShape(16), gorgonia.WithName("conv1_b"), gorgonia.WithInit(gorgonia.Zeroes()))

	// Conv2: 16 -> 32 channels, 3x3 kernel
	conv2W := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(32, 16, 3, 3), gorgonia.WithName("conv2_w"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	conv2B := gorgonia.NewTensor(g, tensor.Float64, 1, gorgonia.WithShape(32), gorgonia.WithName("conv2_b"), gorgonia.WithInit(gorgonia.Zeroes()))

	// Build forward pass
	// Conv1 + ReLU + MaxPool
	conv1, err := gorgonia.Conv2d(input, conv1W, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return nil, fmt.Errorf("conv1 failed: %w", err)
	}
	conv1 = gorgonia.Must(gorgonia.BroadcastAdd(conv1, conv1B, nil, []byte{0, 2, 3}))
	conv1 = gorgonia.Must(gorgonia.Rectify(conv1))
	pool1, err := gorgonia.MaxPool2D(conv1, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return nil, fmt.Errorf("pool1 failed: %w", err)
	}

	// Conv2 + ReLU + MaxPool
	conv2, err := gorgonia.Conv2d(pool1, conv2W, tensor.Shape{3, 3}, []int{1, 1}, []int{1, 1}, []int{1, 1})
	if err != nil {
		return nil, fmt.Errorf("conv2 failed: %w", err)
	}
	conv2 = gorgonia.Must(gorgonia.BroadcastAdd(conv2, conv2B, nil, []byte{0, 2, 3}))
	conv2 = gorgonia.Must(gorgonia.Rectify(conv2))
	pool2, err := gorgonia.MaxPool2D(conv2, tensor.Shape{2, 2}, []int{0, 0}, []int{2, 2})
	if err != nil {
		return nil, fmt.Errorf("pool2 failed: %w", err)
	}

	// Flatten
	batchSize := 1
	flat := gorgonia.Must(gorgonia.Reshape(pool2, tensor.Shape{batchSize, -1}))

	// Calculate flattened size: after 2x MaxPool(2x2) on 8x8, we get 2x2 spatial
	// With 32 channels: 32 * 2 * 2 = 128
	flatSize := 32 * 2 * 2 // After two 2x2 pooling operations

	// FC1: flattened -> hiddenSize
	fc1W := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(flatSize, hiddenSize), gorgonia.WithName("fc1_w"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	fc1B := gorgonia.NewVector(g, tensor.Float64, gorgonia.WithShape(hiddenSize), gorgonia.WithName("fc1_b"), gorgonia.WithInit(gorgonia.Zeroes()))

	fc1 := gorgonia.Must(gorgonia.Mul(flat, fc1W))
	fc1 = gorgonia.Must(gorgonia.BroadcastAdd(fc1, fc1B, nil, []byte{0}))
	fc1 = gorgonia.Must(gorgonia.Rectify(fc1))

	// FC2: hiddenSize -> hiddenSize
	fc2W := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(hiddenSize, hiddenSize), gorgonia.WithName("fc2_w"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	fc2B := gorgonia.NewVector(g, tensor.Float64, gorgonia.WithShape(hiddenSize), gorgonia.WithName("fc2_b"), gorgonia.WithInit(gorgonia.Zeroes()))

	fc2 := gorgonia.Must(gorgonia.Mul(fc1, fc2W))
	fc2 = gorgonia.Must(gorgonia.BroadcastAdd(fc2, fc2B, nil, []byte{0}))
	fc2 = gorgonia.Must(gorgonia.Rectify(fc2))

	// FC3: hiddenSize -> outputSize
	fc3W := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(hiddenSize, outputSize), gorgonia.WithName("fc3_w"), gorgonia.WithInit(gorgonia.GlorotU(1.0)))
	fc3B := gorgonia.NewVector(g, tensor.Float64, gorgonia.WithShape(outputSize), gorgonia.WithName("fc3_b"), gorgonia.WithInit(gorgonia.Zeroes()))

	fc3 := gorgonia.Must(gorgonia.Mul(fc2, fc3W))
	output := gorgonia.Must(gorgonia.BroadcastAdd(fc3, fc3B, nil, []byte{0}))

	// Softmax for probabilities
	output = gorgonia.Must(gorgonia.SoftMax(output))

	// Create VM
	vm := gorgonia.NewTapeMachine(g)

	return &ChessNet{
		g:          g,
		input:      input,
		conv1W:     conv1W,
		conv1B:     conv1B,
		conv2W:     conv2W,
		conv2B:     conv2B,
		fc1W:       fc1W,
		fc1B:       fc1B,
		fc2W:       fc2W,
		fc2B:       fc2B,
		fc3W:       fc3W,
		fc3B:       fc3B,
		output:     output,
		vm:         vm,
		inputSize:  inputSize,
		hiddenSize: hiddenSize,
		outputSize: outputSize,
	}, nil
}

// Predict performs inference on the input board state
func (cn *ChessNet) Predict(boardState []float64) ([]float64, error) {
	if len(boardState) != cn.inputSize {
		return nil, fmt.Errorf("invalid input size: expected %d, got %d", cn.inputSize, len(boardState))
	}

	// Reshape to 4D tensor (batch, channels, height, width)
	boardSize := int(math.Sqrt(float64(cn.inputSize)))
	inputTensor := tensor.New(
		tensor.WithShape(1, 1, boardSize, boardSize),
		tensor.WithBacking(boardState),
	)

	// Set input
	if err := gorgonia.Let(cn.input, inputTensor); err != nil {
		return nil, fmt.Errorf("failed to set input: %w", err)
	}

	// Run forward pass
	if err := cn.vm.RunAll(); err != nil {
		return nil, fmt.Errorf("failed to run inference: %w", err)
	}

	// Get output
	outputValue := cn.output.Value()
	if outputValue == nil {
		return nil, fmt.Errorf("output is nil")
	}

	outputData := outputValue.Data().([]float64)

	// Reset VM for next run
	cn.vm.Reset()

	return outputData, nil
}

// GetLearnables returns all learnable parameters
func (cn *ChessNet) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{
		cn.conv1W, cn.conv1B,
		cn.conv2W, cn.conv2B,
		cn.fc1W, cn.fc1B,
		cn.fc2W, cn.fc2B,
		cn.fc3W, cn.fc3B,
	}
}

// Save saves the model weights to a file
func (cn *ChessNet) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer f.Close()

	encoder := gob.NewEncoder(f)

	// Save each weight matrix
	weights := []struct {
		name string
		node *gorgonia.Node
	}{
		{"conv1W", cn.conv1W},
		{"conv1B", cn.conv1B},
		{"conv2W", cn.conv2W},
		{"conv2B", cn.conv2B},
		{"fc1W", cn.fc1W},
		{"fc1B", cn.fc1B},
		{"fc2W", cn.fc2W},
		{"fc2B", cn.fc2B},
		{"fc3W", cn.fc3W},
		{"fc3B", cn.fc3B},
	}

	for _, w := range weights {
		val := w.node.Value()
		if val == nil {
			continue
		}

		data := val.Data().([]float64)
		shape := val.Shape()

		if err := encoder.Encode(shape); err != nil {
			return fmt.Errorf("failed to encode %s shape: %w", w.name, err)
		}
		if err := encoder.Encode(data); err != nil {
			return fmt.Errorf("failed to encode %s data: %w", w.name, err)
		}
	}

	return nil
}

// Load loads the model weights from a file
func (cn *ChessNet) Load(path string) error {
	f, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer f.Close()

	decoder := gob.NewDecoder(f)

	weights := []*gorgonia.Node{
		cn.conv1W, cn.conv1B,
		cn.conv2W, cn.conv2B,
		cn.fc1W, cn.fc1B,
		cn.fc2W, cn.fc2B,
		cn.fc3W, cn.fc3B,
	}

	for _, w := range weights {
		var shape tensor.Shape
		var data []float64

		if err := decoder.Decode(&shape); err != nil {
			return fmt.Errorf("failed to decode shape: %w", err)
		}
		if err := decoder.Decode(&data); err != nil {
			return fmt.Errorf("failed to decode data: %w", err)
		}

		t := tensor.New(tensor.WithShape(shape...), tensor.WithBacking(data))
		if err := gorgonia.Let(w, t); err != nil {
			return fmt.Errorf("failed to set weight: %w", err)
		}
	}

	return nil
}

// Close cleans up resources
func (cn *ChessNet) Close() error {
	cn.vm.Close()
	return nil
}

// ModelExists checks if a model file exists
func ModelExists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}

// GetTopKMoves returns the top K moves with their probabilities
func GetTopKMoves(predictions []float64, k int) []MoveScore {
	if k > len(predictions) {
		k = len(predictions)
	}

	scores := make([]MoveScore, len(predictions))
	for i, p := range predictions {
		scores[i] = MoveScore{
			MoveIndex: i,
			Score:     p,
		}
	}

	// Sort by score descending
	for i := 0; i < k; i++ {
		maxIdx := i
		for j := i + 1; j < len(scores); j++ {
			if scores[j].Score > scores[maxIdx].Score {
				maxIdx = j
			}
		}
		scores[i], scores[maxIdx] = scores[maxIdx], scores[i]
	}

	return scores[:k]
}

// MoveScore represents a move with its confidence score
type MoveScore struct {
	MoveIndex int
	Score     float64
}

// DecodeMove converts a move index to chess notation (simplified)
func DecodeMove(moveIndex int) string {
	// For 8x8 board, 4096 possible moves (64 * 64)
	from := moveIndex / 64
	to := moveIndex % 64

	fromFile := rune('a' + (from % 8))
	fromRank := (from / 8) + 1
	toFile := rune('a' + (to % 8))
	toRank := (to / 8) + 1

	return fmt.Sprintf("%c%d%c%d", fromFile, fromRank, toFile, toRank)
}

// EncodeMove converts chess notation to move index
func EncodeMove(move string) (int, error) {
	if len(move) != 4 {
		return -1, fmt.Errorf("invalid move format: %s", move)
	}

	fromFile := int(move[0] - 'a')
	fromRank := int(move[1] - '1')
	toFile := int(move[2] - 'a')
	toRank := int(move[3] - '1')

	if fromFile < 0 || fromFile > 7 || fromRank < 0 || fromRank > 7 ||
		toFile < 0 || toFile > 7 || toRank < 0 || toRank > 7 {
		return -1, fmt.Errorf("invalid move coordinates: %s", move)
	}

	from := fromRank*8 + fromFile
	to := toRank*8 + toFile

	return from*64 + to, nil
}
