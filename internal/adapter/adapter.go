package adapter

import (
	"fmt"

	"gorgonia.org/tensor"
)

// GameAdapter defines the interface for game-specific state/action encoding
// This allows P.A.R.T.N.E.R to work with any game without changing core logic
type GameAdapter interface {
	// EncodeState converts game-specific state to neural network tensor
	// frame can be any game-specific representation (FEN string, board array, pixel data, etc.)
	EncodeState(frame interface{}) (tensor.Tensor, error)

	// EncodeAction converts game-specific action to neural network format
	// action can be a move string, button press, steering angle, etc.
	EncodeAction(action interface{}) (tensor.Tensor, error)

	// DecodeAction converts neural network prediction to game-specific action
	// pred is the raw network output that needs to be interpreted
	DecodeAction(pred tensor.Tensor) (interface{}, error)

	// Feedback provides the correct action for learning
	// This is used to update the replay buffer for retraining
	Feedback(correctAction interface{}) error

	// ValidateState checks if a game state is valid
	ValidateState(frame interface{}) error

	// GetStateDimensions returns the tensor dimensions for this game's state
	GetStateDimensions() []int

	// GetActionDimensions returns the tensor dimensions for this game's actions
	GetActionDimensions() []int

	// GetGameName returns the name of the game this adapter handles
	GetGameName() string
}

// AdapterFactory creates game adapters by name
type AdapterFactory struct {
	adapters map[string]func() GameAdapter
}

// NewAdapterFactory creates a new adapter factory
func NewAdapterFactory() *AdapterFactory {
	factory := &AdapterFactory{
		adapters: make(map[string]func() GameAdapter),
	}

	// Register built-in adapters
	factory.Register("chess", func() GameAdapter {
		return NewChessAdapter()
	})

	return factory
}

// Register adds a new adapter to the factory
func (f *AdapterFactory) Register(name string, constructor func() GameAdapter) {
	f.adapters[name] = constructor
}

// Create instantiates an adapter by name
func (f *AdapterFactory) Create(name string) (GameAdapter, error) {
	constructor, exists := f.adapters[name]
	if !exists {
		return nil, fmt.Errorf("unknown adapter: %s", name)
	}
	return constructor(), nil
}

// ListAdapters returns all available adapter names
func (f *AdapterFactory) ListAdapters() []string {
	names := make([]string, 0, len(f.adapters))
	for name := range f.adapters {
		names = append(names, name)
	}
	return names
}

// BaseAdapter provides common functionality for all adapters
type BaseAdapter struct {
	gameName         string
	stateDimensions  []int
	actionDimensions []int
	replayBuffer     []Experience
	maxBufferSize    int
}

// Experience represents a single game experience for learning
type Experience struct {
	State     tensor.Tensor
	Action    tensor.Tensor
	Reward    float64
	NextState tensor.Tensor
	Done      bool
	Metadata  map[string]interface{}
}

// NewBaseAdapter creates a base adapter with common functionality
func NewBaseAdapter(gameName string, stateDims, actionDims []int) *BaseAdapter {
	return &BaseAdapter{
		gameName:         gameName,
		stateDimensions:  stateDims,
		actionDimensions: actionDims,
		replayBuffer:     make([]Experience, 0),
		maxBufferSize:    10000,
	}
}

// GetGameName returns the game name
func (b *BaseAdapter) GetGameName() string {
	return b.gameName
}

// GetStateDimensions returns state tensor dimensions
func (b *BaseAdapter) GetStateDimensions() []int {
	return b.stateDimensions
}

// GetActionDimensions returns action tensor dimensions
func (b *BaseAdapter) GetActionDimensions() []int {
	return b.actionDimensions
}

// AddExperience adds an experience to the replay buffer
func (b *BaseAdapter) AddExperience(exp Experience) {
	b.replayBuffer = append(b.replayBuffer, exp)

	// Trim buffer if it exceeds max size
	if len(b.replayBuffer) > b.maxBufferSize {
		// Remove oldest experiences
		b.replayBuffer = b.replayBuffer[len(b.replayBuffer)-b.maxBufferSize:]
	}
}

// GetReplayBuffer returns the replay buffer
func (b *BaseAdapter) GetReplayBuffer() []Experience {
	return b.replayBuffer
}

// ClearReplayBuffer clears the replay buffer
func (b *BaseAdapter) ClearReplayBuffer() {
	b.replayBuffer = make([]Experience, 0)
}

// GetBufferSize returns current buffer size
func (b *BaseAdapter) GetBufferSize() int {
	return len(b.replayBuffer)
}

// SetMaxBufferSize sets the maximum replay buffer size
func (b *BaseAdapter) SetMaxBufferSize(size int) {
	b.maxBufferSize = size
}
