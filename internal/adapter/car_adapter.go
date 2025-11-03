package adapter

import (
	"fmt"
	"math"

	"gorgonia.org/tensor"
)

// RacingAdapter implements GameAdapter for racing/driving games.
//
// State representation:
// - Speed (normalized 0-1)
// - Position on track (x, y, heading)
// - Track sensors (8 distance readings)
// - Previous control inputs (for temporal consistency)
//
// Action representation:
// - Steering: continuous [-1, 1] (left to right)
// - Throttle: continuous [0, 1] (0 to full acceleration)
// - Brake: continuous [0, 1] (0 to full braking)
//
// # Example Usage
//
//	adapter := NewRacingAdapter()
//
//	// Encode car state
//	state := RacingState{
//	    Speed:       180.0, // km/h
//	    Position:    Position{X: 100.5, Y: 250.3, Heading: 1.57},
//	    TrackSensors: []float64{10.0, 12.5, 15.0, 20.0, 18.0, 14.0, 11.0, 9.5},
//	    LastSteering: 0.2,
//	    LastThrottle: 0.8,
//	}
//	tensor, _ := adapter.EncodeState(state)
//
//	// Decode network output to controls
//	prediction := model.Forward(tensor)
//	controls, _ := adapter.DecodeAction(prediction)
//	// controls: {"steering": 0.15, "throttle": 0.9, "brake": 0.0}
type RacingAdapter struct {
	*BaseAdapter

	// Configuration
	maxSpeed    float64 // Maximum speed for normalization (km/h)
	maxDistance float64 // Maximum sensor distance (meters)
	numSensors  int     // Number of track distance sensors

	// State tracking
	lastState   RacingState
	lastControl RacingControl

	// Statistics
	totalDistance float64
	crashCount    int
	bestLapTime   float64
}

// RacingState represents the complete state of a racing car
type RacingState struct {
	// Velocity
	Speed  float64 `json:"speed"`   // Current speed (km/h)
	SpeedX float64 `json:"speed_x"` // Velocity components
	SpeedY float64 `json:"speed_y"`

	// Position
	Position Position `json:"position"` // Track position

	// Sensors
	TrackSensors []float64 `json:"track_sensors"` // Distance to track edge (8 directions)

	// Previous controls (for temporal consistency)
	LastSteering float64 `json:"last_steering"` // Previous steering input
	LastThrottle float64 `json:"last_throttle"` // Previous throttle input
	LastBrake    float64 `json:"last_brake"`    // Previous brake input

	// Track info
	DistanceRaced float64 `json:"distance_raced"` // Total distance (meters)
	LapTime       float64 `json:"lap_time"`       // Current lap time (seconds)
	OnTrack       bool    `json:"on_track"`       // Whether car is on track

	// Metadata
	Timestamp int64 `json:"timestamp"`
}

// Position represents 2D position and orientation
type Position struct {
	X       float64 `json:"x"`       // X coordinate (meters)
	Y       float64 `json:"y"`       // Y coordinate (meters)
	Heading float64 `json:"heading"` // Orientation in radians [0, 2π)
}

// RacingControl represents the control inputs for the car
type RacingControl struct {
	Steering float64 `json:"steering"` // [-1, 1] left to right
	Throttle float64 `json:"throttle"` // [0, 1] acceleration
	Brake    float64 `json:"brake"`    // [0, 1] braking

	// Advanced controls (optional)
	Gear   int     `json:"gear"`   // Current gear (0=reverse, 1-6)
	Clutch float64 `json:"clutch"` // [0, 1] clutch engagement
}

// NewRacingAdapter creates a new racing game adapter
func NewRacingAdapter() *RacingAdapter {
	return NewRacingAdapterWithConfig(RacingConfig{
		MaxSpeed:    300.0, // km/h
		MaxDistance: 200.0, // meters
		NumSensors:  8,     // 8 directional sensors
	})
}

// RacingConfig holds racing adapter configuration
type RacingConfig struct {
	MaxSpeed    float64 `json:"max_speed"`
	MaxDistance float64 `json:"max_distance"`
	NumSensors  int     `json:"num_sensors"`
}

// NewRacingAdapterWithConfig creates a racing adapter with custom configuration
func NewRacingAdapterWithConfig(config RacingConfig) *RacingAdapter {
	// State: speed(3) + position(3) + sensors(8) + prev_controls(3) = 17 dimensions
	stateDims := []int{3 + 3 + config.NumSensors + 3}

	// Action: steering(1) + throttle(1) + brake(1) = 3 dimensions
	actionDims := []int{3}

	return &RacingAdapter{
		BaseAdapter: NewBaseAdapter("racing", stateDims, actionDims),
		maxSpeed:    config.MaxSpeed,
		maxDistance: config.MaxDistance,
		numSensors:  config.NumSensors,
		bestLapTime: math.MaxFloat64,
	}
}

// EncodeState converts racing state to neural network tensor
func (r *RacingAdapter) EncodeState(frame interface{}) (tensor.Tensor, error) {
	var state RacingState

	switch s := frame.(type) {
	case RacingState:
		state = s

	case map[string]interface{}:
		// Parse from generic map
		var err error
		state, err = r.parseStateFromMap(s)
		if err != nil {
			return nil, fmt.Errorf("failed to parse state map: %w", err)
		}

	default:
		return nil, fmt.Errorf("unsupported state type: %T", frame)
	}

	// Normalize and build tensor
	data := make([]float64, 0, r.stateDimensions[0])

	// 1. Speed components (normalized to [0, 1])
	data = append(data, state.Speed/r.maxSpeed)
	data = append(data, state.SpeedX/r.maxSpeed)
	data = append(data, state.SpeedY/r.maxSpeed)

	// 2. Position (normalized, heading as sin/cos)
	data = append(data, math.Cos(state.Position.Heading))      // Cos(heading)
	data = append(data, math.Sin(state.Position.Heading))      // Sin(heading)
	data = append(data, clamp(state.Position.X/1000.0, -1, 1)) // Normalized X

	// 3. Track sensors (normalized distances)
	for i := 0; i < r.numSensors; i++ {
		if i < len(state.TrackSensors) {
			normalized := state.TrackSensors[i] / r.maxDistance
			data = append(data, clamp(normalized, 0, 1))
		} else {
			data = append(data, 1.0) // Max distance if sensor not available
		}
	}

	// 4. Previous controls (for temporal consistency)
	data = append(data, clamp(state.LastSteering, -1, 1))
	data = append(data, clamp(state.LastThrottle, 0, 1))
	data = append(data, clamp(state.LastBrake, 0, 1))

	// Store for feedback
	r.lastState = state

	return tensor.New(
		tensor.WithShape(r.stateDimensions...),
		tensor.WithBacking(data),
	), nil
}

// EncodeAction converts racing control to tensor
func (r *RacingAdapter) EncodeAction(action interface{}) (tensor.Tensor, error) {
	var control RacingControl

	switch a := action.(type) {
	case RacingControl:
		control = a

	case map[string]interface{}:
		// Parse from map
		if steering, ok := a["steering"].(float64); ok {
			control.Steering = steering
		}
		if throttle, ok := a["throttle"].(float64); ok {
			control.Throttle = throttle
		}
		if brake, ok := a["brake"].(float64); ok {
			control.Brake = brake
		}

	case []float64:
		// Direct array
		if len(a) >= 3 {
			control.Steering = a[0]
			control.Throttle = a[1]
			control.Brake = a[2]
		}

	default:
		return nil, fmt.Errorf("unsupported action type: %T", action)
	}

	// Validate and clamp
	control.Steering = clamp(control.Steering, -1, 1)
	control.Throttle = clamp(control.Throttle, 0, 1)
	control.Brake = clamp(control.Brake, 0, 1)

	data := []float64{control.Steering, control.Throttle, control.Brake}

	return tensor.New(
		tensor.WithShape(3),
		tensor.WithBacking(data),
	), nil
}

// DecodeAction converts network prediction to racing controls
func (r *RacingAdapter) DecodeAction(pred tensor.Tensor) (interface{}, error) {
	shape := pred.Shape()
	if len(shape) != 1 || shape[0] != 3 {
		return nil, fmt.Errorf("invalid prediction shape: %v (expected [3])", shape)
	}

	data := pred.Data().([]float64)

	// Decode and clamp controls
	control := RacingControl{
		Steering: clamp(data[0], -1, 1),
		Throttle: clamp(data[1], 0, 1),
		Brake:    clamp(data[2], 0, 1),
	}

	// Mutual exclusivity: if braking, reduce throttle
	if control.Brake > 0.1 {
		control.Throttle *= (1.0 - control.Brake)
	}

	// Store for next iteration
	r.lastControl = control

	return map[string]interface{}{
		"steering": control.Steering,
		"throttle": control.Throttle,
		"brake":    control.Brake,
		"formatted": fmt.Sprintf("Steer: %+.2f, Throttle: %.2f, Brake: %.2f",
			control.Steering, control.Throttle, control.Brake),
	}, nil
}

// Feedback provides the correct action for learning
func (r *RacingAdapter) Feedback(correctAction interface{}) error {
	actionTensor, err := r.EncodeAction(correctAction)
	if err != nil {
		return fmt.Errorf("failed to encode correct action: %w", err)
	}

	stateTensor, err := r.EncodeState(r.lastState)
	if err != nil {
		return fmt.Errorf("failed to encode state: %w", err)
	}

	// Calculate reward based on performance
	reward := r.calculateReward(r.lastState, correctAction)

	exp := Experience{
		State:  stateTensor,
		Action: actionTensor,
		Reward: reward,
		Done:   !r.lastState.OnTrack, // Episode ends on crash
		Metadata: map[string]interface{}{
			"speed":    r.lastState.Speed,
			"on_track": r.lastState.OnTrack,
			"lap_time": r.lastState.LapTime,
			"distance": r.lastState.DistanceRaced,
		},
	}

	r.AddExperience(exp)

	// Update statistics
	if !r.lastState.OnTrack {
		r.crashCount++
	}

	return nil
}

// ValidateState checks if racing state is valid
func (r *RacingAdapter) ValidateState(frame interface{}) error {
	state, ok := frame.(RacingState)
	if !ok {
		return fmt.Errorf("state must be RacingState, got %T", frame)
	}

	// Check speed bounds
	if state.Speed < 0 || state.Speed > r.maxSpeed*1.5 {
		return fmt.Errorf("speed out of bounds: %.2f (max: %.2f)", state.Speed, r.maxSpeed)
	}

	// Check sensor count
	if len(state.TrackSensors) != r.numSensors {
		return fmt.Errorf("expected %d sensors, got %d", r.numSensors, len(state.TrackSensors))
	}

	// Check for NaN/Inf values
	if math.IsNaN(state.Speed) || math.IsInf(state.Speed, 0) {
		return fmt.Errorf("invalid speed value: %v", state.Speed)
	}

	for i, sensor := range state.TrackSensors {
		if math.IsNaN(sensor) || math.IsInf(sensor, 0) {
			return fmt.Errorf("invalid sensor %d value: %v", i, sensor)
		}
	}

	return nil
}

// calculateReward computes reward for reinforcement learning
func (r *RacingAdapter) calculateReward(state RacingState, action interface{}) float64 {
	reward := 0.0

	// Base reward: staying on track
	if state.OnTrack {
		reward += 1.0
	} else {
		reward -= 10.0 // Heavy penalty for crashes
	}

	// Speed reward (encourage high speed when safe)
	minSensorDist := math.MaxFloat64
	for _, dist := range state.TrackSensors {
		if dist < minSensorDist {
			minSensorDist = dist
		}
	}

	// Reward speed when there's clearance
	if minSensorDist > 10.0 {
		reward += (state.Speed / r.maxSpeed) * 0.5
	}

	// Distance reward
	reward += state.DistanceRaced * 0.01

	// Smooth driving reward (penalize jerky controls)
	if control, ok := action.(RacingControl); ok {
		steeringChange := math.Abs(control.Steering - r.lastControl.Steering)
		if steeringChange < 0.1 {
			reward += 0.1 // Reward smooth steering
		} else if steeringChange > 0.5 {
			reward -= 0.2 // Penalize jerky steering
		}
	}

	return reward
}

// GetPerformanceStats returns driving performance statistics
func (r *RacingAdapter) GetPerformanceStats() PerformanceStats {
	bufferStats := r.GetBufferStats()

	avgSpeed := 0.0
	if bufferStats.TotalExperiences > 0 {
		for _, exp := range r.GetReplayBuffer() {
			if meta, ok := exp.Metadata["speed"].(float64); ok {
				avgSpeed += meta
			}
		}
		avgSpeed /= float64(bufferStats.TotalExperiences)
	}

	return PerformanceStats{
		TotalDistance:    r.totalDistance,
		AverageSpeed:     avgSpeed,
		CrashCount:       r.crashCount,
		BestLapTime:      r.bestLapTime,
		TotalExperiences: bufferStats.TotalExperiences,
	}
}

// PerformanceStats contains racing performance metrics
type PerformanceStats struct {
	TotalDistance    float64 `json:"total_distance"`
	AverageSpeed     float64 `json:"average_speed"`
	CrashCount       int     `json:"crash_count"`
	BestLapTime      float64 `json:"best_lap_time"`
	TotalExperiences int     `json:"total_experiences"`
}

// String returns human-readable performance stats
func (ps PerformanceStats) String() string {
	return fmt.Sprintf(
		"Racing{Distance: %.1fm, AvgSpeed: %.1fkm/h, Crashes: %d, BestLap: %.2fs}",
		ps.TotalDistance,
		ps.AverageSpeed,
		ps.CrashCount,
		ps.BestLapTime,
	)
}

// parseStateFromMap converts generic map to RacingState
func (r *RacingAdapter) parseStateFromMap(data map[string]interface{}) (RacingState, error) {
	state := RacingState{
		TrackSensors: make([]float64, r.numSensors),
	}

	// Parse speed
	if speed, ok := data["speed"].(float64); ok {
		state.Speed = speed
	}

	// Parse position
	if pos, ok := data["position"].(map[string]interface{}); ok {
		if x, ok := pos["x"].(float64); ok {
			state.Position.X = x
		}
		if y, ok := pos["y"].(float64); ok {
			state.Position.Y = y
		}
		if heading, ok := pos["heading"].(float64); ok {
			state.Position.Heading = heading
		}
	}

	// Parse sensors
	if sensors, ok := data["sensors"].([]interface{}); ok {
		for i, s := range sensors {
			if i >= r.numSensors {
				break
			}
			if val, ok := s.(float64); ok {
				state.TrackSensors[i] = val
			}
		}
	} else if sensors, ok := data["sensors"].([]float64); ok {
		copy(state.TrackSensors, sensors)
	}

	// Parse on_track
	if onTrack, ok := data["on_track"].(bool); ok {
		state.OnTrack = onTrack
	} else {
		state.OnTrack = true // Default to on track
	}

	return state, nil
}

// Helper functions

func clamp(value, min, max float64) float64 {
	if value < min {
		return min
	}
	if value > max {
		return max
	}
	return value
}
