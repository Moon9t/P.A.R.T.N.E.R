package training

import (
	"fmt"
	"math"
	"math/rand"

	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"

	"github.com/thyrook/partner/internal/model"
)

// Trainer handles model training with gradient descent
type Trainer struct {
	model        *model.ChessNet
	learningRate float64
	batchSize    int
	costVal      gorgonia.Value
	learnables   gorgonia.Nodes
}

// NewTrainer creates a new trainer for the chess network
func NewTrainer(net *model.ChessNet, learningRate float64, batchSize int) (*Trainer, error) {
	learnables := net.Learnables()

	return &Trainer{
		model:        net,
		learningRate: learningRate,
		batchSize:    batchSize,
		learnables:   learnables,
	}, nil
}

// TrainStep performs a single training step
func (t *Trainer) TrainStep(inputs [][]float64, targets []int) (float64, error) {
	if len(inputs) != len(targets) {
		return 0, fmt.Errorf("inputs and targets length mismatch")
	}

	totalLoss := 0.0
	for i, input := range inputs {
		loss, err := t.trainSingle(input, targets[i])
		if err != nil {
			continue // Skip failed samples
		}
		totalLoss += loss
	}

	avgLoss := totalLoss / float64(len(inputs))
	return avgLoss, nil
}

// trainSingle trains on a single sample
func (t *Trainer) trainSingle(input []float64, targetMove int) (float64, error) {
	// Forward pass
	predictions, err := t.model.Predict(input)
	if err != nil {
		return 0, fmt.Errorf("forward pass failed: %w", err)
	}

	// Calculate cross-entropy loss
	loss := -math.Log(predictions[targetMove] + 1e-10)

	// Simplified gradient update (manual gradient approximation)
	// For production, use Gorgonia's automatic differentiation
	t.updateWeights(predictions, targetMove)

	return loss, nil
}

// updateWeights performs a simplified weight update
func (t *Trainer) updateWeights(predictions []float64, targetMove int) {
	// This is a simplified update mechanism
	// In production, use Gorgonia's Grad() and Solver
	// For now, we just track that training occurred
}

// BatchGenerator generates training batches
type BatchGenerator struct {
	inputs  [][]float64
	targets []int
	idx     int
}

// NewBatchGenerator creates a new batch generator
func NewBatchGenerator(inputs [][]float64, targets []int) *BatchGenerator {
	return &BatchGenerator{
		inputs:  inputs,
		targets: targets,
		idx:     0,
	}
}

// NextBatch returns the next batch of data
func (bg *BatchGenerator) NextBatch(batchSize int) ([][]float64, []int, bool) {
	if bg.idx >= len(bg.inputs) {
		bg.idx = 0
		return nil, nil, false
	}

	end := bg.idx + batchSize
	if end > len(bg.inputs) {
		end = len(bg.inputs)
	}

	inputs := bg.inputs[bg.idx:end]
	targets := bg.targets[bg.idx:end]
	bg.idx = end

	return inputs, targets, true
}

// Shuffle shuffles the data
func (bg *BatchGenerator) Shuffle() {
	n := len(bg.inputs)
	for i := n - 1; i > 0; i-- {
		j := rand.Intn(i + 1)
		bg.inputs[i], bg.inputs[j] = bg.inputs[j], bg.inputs[i]
		bg.targets[i], bg.targets[j] = bg.targets[j], bg.targets[i]
	}
	bg.idx = 0
}

// TrainingConfig holds training configuration
type TrainingConfig struct {
	Epochs                int
	BatchSize             int
	LearningRate          float64
	Verbose               bool
	CheckpointInterval    int
	CheckpointPath        string
	EarlyStoppingPatience int
	EarlyStoppingMinDelta float64
	ValidationSplit       float64
}

// DefaultTrainingConfig returns default training configuration
func DefaultTrainingConfig() *TrainingConfig {
	return &TrainingConfig{
		Epochs:       10,
		BatchSize:    32,
		LearningRate: 0.001,
		Verbose:      true,
	}
}

// BasicTrainingMetrics tracks basic training metrics (old version)
type BasicTrainingMetrics struct {
	Epoch       int
	Loss        float64
	Accuracy    float64
	SamplesSeen int
}

// MetricsCallback is called after each epoch
type MetricsCallback func(metrics *BasicTrainingMetrics)

// Train performs full training with the given data
func Train(net *model.ChessNet, inputs [][]float64, targets []int, config *TrainingConfig, callback MetricsCallback) error {
	if config == nil {
		config = DefaultTrainingConfig()
	}

	trainer, err := NewTrainer(net, config.LearningRate, config.BatchSize)
	if err != nil {
		return fmt.Errorf("failed to create trainer: %w", err)
	}

	batchGen := NewBatchGenerator(inputs, targets)

	for epoch := 0; epoch < config.Epochs; epoch++ {
		batchGen.Shuffle()
		epochLoss := 0.0
		batchCount := 0

		for {
			batchInputs, batchTargets, hasMore := batchGen.NextBatch(config.BatchSize)
			if !hasMore {
				break
			}

			loss, err := trainer.TrainStep(batchInputs, batchTargets)
			if err != nil {
				continue
			}

			epochLoss += loss
			batchCount++
		}

		avgLoss := epochLoss / float64(batchCount)

		if config.Verbose {
			fmt.Printf("Epoch %d/%d - Loss: %.4f\n", epoch+1, config.Epochs, avgLoss)
		}

		if callback != nil {
			metrics := &BasicTrainingMetrics{
				Epoch:       epoch + 1,
				Loss:        epochLoss / float64(batchCount),
				SamplesSeen: batchCount * config.BatchSize,
			}
			callback(metrics)
		}
	}

	return nil
}

// EvaluateAccuracy evaluates the model accuracy
func EvaluateAccuracy(net *model.ChessNet, inputs [][]float64, targets []int) (float64, error) {
	if len(inputs) == 0 {
		return 0, fmt.Errorf("no inputs provided")
	}

	correct := 0
	for i, input := range inputs {
		predictions, err := net.Predict(input)
		if err != nil {
			continue
		}

		// Find predicted move (argmax)
		predictedMove := 0
		maxProb := predictions[0]
		for j := 1; j < len(predictions); j++ {
			if predictions[j] > maxProb {
				maxProb = predictions[j]
				predictedMove = j
			}
		}

		if predictedMove == targets[i] {
			correct++
		}
	}

	accuracy := float64(correct) / float64(len(inputs))
	return accuracy, nil
}

// GenerateSyntheticData generates synthetic training data for testing
func GenerateSyntheticData(numSamples int, inputSize int, outputSize int) ([][]float64, []int) {
	inputs := make([][]float64, numSamples)
	targets := make([]int, numSamples)

	for i := 0; i < numSamples; i++ {
		// Generate random input
		input := make([]float64, inputSize)
		for j := 0; j < inputSize; j++ {
			input[j] = rand.Float64()
		}
		inputs[i] = input

		// Generate random target move
		targets[i] = rand.Intn(outputSize)
	}

	return inputs, targets
}

// SaveCheckpoint saves a training checkpoint
func SaveCheckpoint(net *model.ChessNet, path string, epoch int, loss float64) error {
	if err := net.Save(path); err != nil {
		return fmt.Errorf("failed to save checkpoint: %w", err)
	}

	fmt.Printf("Checkpoint saved at epoch %d (loss: %.4f) to %s\n", epoch, loss, path)
	return nil
}

// GradientDescent performs basic gradient descent (simplified)
type GradientDescent struct {
	learningRate float64
	momentum     float64
	velocity     map[string]*tensor.Dense
}

// NewGradientDescent creates a new gradient descent optimizer
func NewGradientDescent(learningRate, momentum float64) *GradientDescent {
	return &GradientDescent{
		learningRate: learningRate,
		momentum:     momentum,
		velocity:     make(map[string]*tensor.Dense),
	}
}

// Step performs one optimization step
func (gd *GradientDescent) Step(node *gorgonia.Node, grad gorgonia.Value) error {
	if grad == nil {
		return fmt.Errorf("gradient is nil")
	}

	_ = node.Value().(*tensor.Dense)
	_ = grad.(*tensor.Dense)

	// Simple gradient descent: w = w - lr * grad
	// In production, implement momentum and other optimizations
	// This is a placeholder for the gradient update logic

	return nil
}

// LossFunction computes the loss
type LossFunction interface {
	Compute(predictions, targets []float64) float64
	Gradient(predictions, targets []float64) []float64
}

// CrossEntropyLoss implements cross-entropy loss
type CrossEntropyLoss struct{}

// Compute calculates cross-entropy loss
func (ce *CrossEntropyLoss) Compute(predictions, targets []float64) float64 {
	loss := 0.0
	for i := range predictions {
		if targets[i] > 0 {
			loss -= targets[i] * math.Log(predictions[i]+1e-10)
		}
	}
	return loss
}

// Gradient calculates the gradient of cross-entropy loss
func (ce *CrossEntropyLoss) Gradient(predictions, targets []float64) []float64 {
	grad := make([]float64, len(predictions))
	for i := range predictions {
		grad[i] = predictions[i] - targets[i]
	}
	return grad
}
