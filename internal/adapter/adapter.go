package adapter

import (
	"fmt"

	"gorgonia.org/tensor"
)

// GameAdapter defines the interface for game-specific state/action encoding.
// This allows P.A.R.T.N.E.R to work with any game without changing core logic.
//
// # Purpose
//
// The adapter pattern decouples game-specific logic from the neural network,
// enabling P.A.R.T.N.E.R to support chess, racing games, robotics, and more
// without modifying the core training and inference pipeline.
//
// # Typical Usage
//
//	// Create an adapter for your game
//	adapter := NewChessAdapter()
//
//	// Encode game state to neural network input
//	state := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1" // FEN
//	tensor, err := adapter.EncodeState(state)
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Run neural network inference
//	prediction := model.Forward(tensor)
//
//	// Decode network output to game action
//	action, err := adapter.DecodeAction(prediction)
//	// action will be: {"move": "e2e4", "from_square": 12, "to_square": 28, "probability": 0.85}
//
// # Creating Custom Adapters
//
// To support a new game, implement the GameAdapter interface:
//
//	type MyGameAdapter struct {
//	    *BaseAdapter
//	    // game-specific fields
//	}
//
//	func NewMyGameAdapter() *MyGameAdapter {
//	    return &MyGameAdapter{
//	        BaseAdapter: NewBaseAdapter("mygame", []int{channels, height, width}, []int{numActions}),
//	    }
//	}
//
//	func (m *MyGameAdapter) EncodeState(frame interface{}) (tensor.Tensor, error) {
//	    // Convert your game state to tensor
//	    // Example: image pixels, sensor readings, board state
//	    return tensor.New(...), nil
//	}
//
//	func (m *MyGameAdapter) DecodeAction(pred tensor.Tensor) (interface{}, error) {
//	    // Convert network output to game action
//	    // Example: button press, move coordinates, steering angle
//	    return myAction, nil
//	}
//
// # Supported Games
//
// - Chess: FEN strings, 12-channel board representation (piece planes)
// - Racing: Speed, position, sensor data → steering/throttle/brake
// - Custom: Implement your own adapter for any game or robotic task
//
// # Best Practices
//
// 1. Normalize inputs: Scale state values to [0, 1] or [-1, 1]
// 2. Validate inputs: Check state validity before encoding
// 3. Use replay buffer: Store experiences for continuous learning
// 4. Handle errors: Provide clear error messages for invalid states
// 5. Document dimensions: Clearly specify tensor shapes in comments
//
// # Example: Racing Game Adapter
//
//	type RacingAdapter struct {
//	    *BaseAdapter
//	}
//
//	func (r *RacingAdapter) EncodeState(frame interface{}) (tensor.Tensor, error) {
//	    // frame contains: speed, position, track sensors
//	    state := frame.(map[string]interface{})
//	    speed := state["speed"].(float64) / 300.0    // Normalize to [0, 1]
//	    sensors := state["sensors"].([]float64)       // 8 track sensors
//
//	    data := append([]float64{speed}, sensors...)
//	    return tensor.New(tensor.WithBacking(data), tensor.WithShape(9)), nil
//	}
//
//	func (r *RacingAdapter) DecodeAction(pred tensor.Tensor) (interface{}, error) {
//	    // pred is [3] vector: [steering, throttle, brake]
//	    data := pred.Data().([]float64)
//	    return map[string]float64{
//	        "steering": data[0], // [-1, 1]
//	        "throttle": data[1], // [0, 1]
//	        "brake":    data[2], // [0, 1]
//	    }, nil
//	}
type GameAdapter interface {
	// EncodeState converts game-specific state to neural network tensor.
	//
	// Input formats supported (game-specific):
	// - Chess: FEN string, [12][8][8]float32 piece planes, or map representation
	// - Racing: map with speed, position, sensor readings
	// - Custom: any game-specific state representation
	//
	// Output: tensor.Tensor with shape matching GetStateDimensions()
	//
	// Example:
	//	state := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	//	tensor, err := adapter.EncodeState(state)
	EncodeState(frame interface{}) (tensor.Tensor, error)

	// EncodeAction converts game-specific action to neural network format.
	//
	// Input formats supported (game-specific):
	// - Chess: move string ("e2e4"), or struct with from/to squares
	// - Racing: map with steering, throttle, brake
	// - Custom: any game-specific action representation
	//
	// Output: tensor.Tensor, often one-hot encoded
	//
	// Example:
	//	action := "e2e4"
	//	tensor, err := adapter.EncodeAction(action)
	EncodeAction(action interface{}) (tensor.Tensor, error)

	// DecodeAction converts neural network prediction to game-specific action.
	//
	// Input: tensor.Tensor from model output (usually probability distribution)
	// Output: game-specific action (move string, control values, etc.)
	//
	// Example:
	//	prediction := model.Forward(stateTensor)
	//	action, err := adapter.DecodeAction(prediction)
	//	// For chess: {"move": "e2e4", "probability": 0.85}
	//	// For racing: {"steering": 0.3, "throttle": 0.8}
	DecodeAction(pred tensor.Tensor) (interface{}, error)

	// Feedback provides the correct action for learning.
	// Used to update the replay buffer for retraining.
	//
	// When observing human play or ground truth:
	//	correctMove := "e2e4"
	//	err := adapter.Feedback(correctMove)
	//	// This stores the experience for future training
	Feedback(correctAction interface{}) error

	// ValidateState checks if a game state is valid.
	//
	// Example checks:
	// - Chess: both kings present, valid piece positions
	// - Racing: speed in valid range, sensors not NaN
	//
	// Returns error if state is invalid.
	ValidateState(frame interface{}) error

	// GetStateDimensions returns the tensor dimensions for this game's state.
	//
	// Examples:
	// - Chess: [12, 8, 8] (12 piece planes, 8x8 board)
	// - Racing: [9] (speed + 8 sensors)
	// - Images: [3, 224, 224] (RGB channels)
	GetStateDimensions() []int

	// GetActionDimensions returns the tensor dimensions for this game's actions.
	//
	// Examples:
	// - Chess: [4096] (64 from × 64 to squares)
	// - Racing: [3] (steering, throttle, brake)
	// - Discrete: [10] (10 possible actions)
	GetActionDimensions() []int

	// GetGameName returns the name of the game this adapter handles.
	//
	// Used for logging, configuration, and adapter factory registration.
	GetGameName() string
}

// AdapterFactory creates game adapters by name using the factory pattern.
//
// The factory allows dynamic adapter selection at runtime, making it easy to
// switch between games or add new game support without modifying core code.
//
// # Usage Example
//
//	// Create factory with built-in adapters
//	factory := NewAdapterFactory()
//
//	// List available adapters
//	fmt.Println("Available:", factory.ListAdapters()) // ["chess", "racing"]
//
//	// Create an adapter by name
//	adapter, err := factory.Create("chess")
//	if err != nil {
//	    log.Fatal(err)
//	}
//
//	// Register custom adapter
//	factory.Register("mygame", func() GameAdapter {
//	    return NewMyGameAdapter()
//	})
//
// # Adding Custom Adapters
//
//	factory := NewAdapterFactory()
//	factory.Register("poker", func() GameAdapter {
//	    return &PokerAdapter{
//	        BaseAdapter: NewBaseAdapter("poker", []int{52, 5}, []int{4}),
//	    }
//	})
//
//	pokerAdapter, _ := factory.Create("poker")
type AdapterFactory struct {
	adapters map[string]func() GameAdapter
}

// NewAdapterFactory creates a new adapter factory with built-in game adapters.
//
// Built-in adapters:
// - "chess": Chess game with FEN support and 12-channel piece planes
// - Additional adapters can be registered using Register()
//
// Returns a factory ready to create adapters by name.
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

// Register adds a new adapter to the factory.
//
// Allows extending P.A.R.T.N.E.R with custom game support.
//
// Parameters:
// - name: Unique identifier for the adapter (e.g., "chess", "racing", "poker")
// - constructor: Function that creates and returns a new adapter instance
//
// Example:
//
//	factory.Register("custom", func() GameAdapter {
//	    return &CustomAdapter{...}
//	})
func (f *AdapterFactory) Register(name string, constructor func() GameAdapter) {
	f.adapters[name] = constructor
}

// Create instantiates an adapter by name.
//
// Parameters:
// - name: The registered adapter name
//
// Returns:
// - GameAdapter: The created adapter instance
// - error: Non-nil if adapter name is not registered
//
// Example:
//
//	adapter, err := factory.Create("chess")
//	if err != nil {
//	    log.Fatalf("Unknown adapter: %v", err)
//	}
func (f *AdapterFactory) Create(name string) (GameAdapter, error) {
	constructor, exists := f.adapters[name]
	if !exists {
		available := f.ListAdapters()
		return nil, fmt.Errorf("unknown adapter '%s'. Available adapters: %v", name, available)
	}
	return constructor(), nil
}

// ListAdapters returns all available adapter names.
//
// Useful for displaying available games to users or validating configuration.
//
// Example:
//
//	adapters := factory.ListAdapters()
//	fmt.Println("Supported games:", adapters) // ["chess", "racing", "custom"]
func (f *AdapterFactory) ListAdapters() []string {
	names := make([]string, 0, len(f.adapters))
	for name := range f.adapters {
		names = append(names, name)
	}
	return names
}

// BaseAdapter provides common functionality for all game adapters.
//
// Includes replay buffer management, basic statistics, and helper methods
// that are useful across all games.
//
// # Usage in Custom Adapters
//
//	type MyAdapter struct {
//	    *BaseAdapter  // Embed to inherit replay buffer, stats, etc.
//	    mySpecificField string
//	}
//
//	func NewMyAdapter() *MyAdapter {
//	    return &MyAdapter{
//	        BaseAdapter: NewBaseAdapter("mygame", []int{10, 10}, []int{4}),
//	    }
//	}
//
// # Replay Buffer
//
// The base adapter maintains a replay buffer for experience storage,
// which is essential for reinforcement learning and continuous improvement.
//
//	// Add experience after each game step
//	adapter.AddExperience(Experience{
//	    State:  stateTensor,
//	    Action: actionTensor,
//	    Reward: 1.0,  // Positive reward for good moves
//	    Done:   false,
//	})
//
//	// Retrieve for training
//	experiences := adapter.GetReplayBuffer()
//	// Train model on experiences...
type BaseAdapter struct {
	gameName         string
	stateDimensions  []int
	actionDimensions []int
	replayBuffer     []Experience
	maxBufferSize    int
}

// Experience represents a single game experience for learning.
//
// Used in reinforcement learning and self-play scenarios to store
// state-action-reward transitions.
//
// # Fields
//
// - State: The game state at this moment (as tensor)
// - Action: The action taken (as tensor)
// - Reward: Reward received (positive for good, negative for bad)
// - NextState: The resulting state after action (nil if terminal)
// - Done: Whether this ended the episode/game
// - Metadata: Additional context (timestamps, player info, etc.)
//
// # Example
//
//	exp := Experience{
//	    State:     encodedBoard,
//	    Action:    encodedMove,
//	    Reward:    1.0,  // Correct move
//	    NextState: encodedNextBoard,
//	    Done:      false,
//	    Metadata: map[string]interface{}{
//	        "move":      "e2e4",
//	        "timestamp": time.Now(),
//	        "player":    "human",
//	    },
//	}
type Experience struct {
	State     tensor.Tensor
	Action    tensor.Tensor
	Reward    float64
	NextState tensor.Tensor
	Done      bool
	Metadata  map[string]interface{}
}

// NewBaseAdapter creates a base adapter with common functionality.
//
// Parameters:
// - gameName: Human-readable game name (e.g., "chess", "racing")
// - stateDims: Dimensions of state tensor (e.g., [12, 8, 8] for chess)
// - actionDims: Dimensions of action tensor (e.g., [4096] for chess moves)
//
// Returns initialized adapter with empty replay buffer (max 10,000 experiences).
//
// Example:
//
//	base := NewBaseAdapter("chess", []int{12, 8, 8}, []int{4096})
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

// GetBufferStats returns statistics about the replay buffer
func (b *BaseAdapter) GetBufferStats() BufferStats {
	if len(b.replayBuffer) == 0 {
		return BufferStats{}
	}

	totalReward := 0.0
	positiveRewards := 0
	negativeRewards := 0
	episodes := 0

	for _, exp := range b.replayBuffer {
		totalReward += exp.Reward
		if exp.Reward > 0 {
			positiveRewards++
		} else if exp.Reward < 0 {
			negativeRewards++
		}
		if exp.Done {
			episodes++
		}
	}

	return BufferStats{
		TotalExperiences: len(b.replayBuffer),
		AverageReward:    totalReward / float64(len(b.replayBuffer)),
		TotalReward:      totalReward,
		PositiveRewards:  positiveRewards,
		NegativeRewards:  negativeRewards,
		TotalEpisodes:    episodes,
	}
}

// GetRecentExperiences returns the most recent N experiences
func (b *BaseAdapter) GetRecentExperiences(n int) []Experience {
	if n >= len(b.replayBuffer) {
		return b.replayBuffer
	}
	start := len(b.replayBuffer) - n
	return b.replayBuffer[start:]
}

// SampleExperiences randomly samples N experiences from the buffer
func (b *BaseAdapter) SampleExperiences(n int) []Experience {
	if n >= len(b.replayBuffer) {
		return b.replayBuffer
	}

	// Simple random sampling (for production, use weighted sampling)
	sampled := make([]Experience, n)
	for i := 0; i < n; i++ {
		idx := i * len(b.replayBuffer) / n // Evenly spaced sampling
		sampled[i] = b.replayBuffer[idx]
	}
	return sampled
}

// BufferStats contains replay buffer statistics
type BufferStats struct {
	TotalExperiences int     `json:"total_experiences"`
	AverageReward    float64 `json:"average_reward"`
	TotalReward      float64 `json:"total_reward"`
	PositiveRewards  int     `json:"positive_rewards"`
	NegativeRewards  int     `json:"negative_rewards"`
	TotalEpisodes    int     `json:"total_episodes"`
}

// String returns a human-readable representation of buffer stats
func (bs BufferStats) String() string {
	return fmt.Sprintf(
		"Buffer{Experiences: %d, Episodes: %d, AvgReward: %.3f, Positive: %d, Negative: %d}",
		bs.TotalExperiences,
		bs.TotalEpisodes,
		bs.AverageReward,
		bs.PositiveRewards,
		bs.NegativeRewards,
	)
}
