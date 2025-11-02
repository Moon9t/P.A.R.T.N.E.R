package model

import (
	"math"
)

// LRScheduler defines the interface for learning rate scheduling
type LRScheduler interface {
	GetLR(step int) float64
	Step()
	GetCurrentLR() float64
}

// CosineAnnealingScheduler implements cosine annealing with warmup
type CosineAnnealingScheduler struct {
	baseLR      float64
	minLR       float64
	warmupSteps int
	totalSteps  int
	currentStep int
	currentLR   float64
}

// NewCosineAnnealingScheduler creates a new cosine annealing scheduler
func NewCosineAnnealingScheduler(baseLR, minLR float64, warmupSteps, totalSteps int) *CosineAnnealingScheduler {
	return &CosineAnnealingScheduler{
		baseLR:      baseLR,
		minLR:       minLR,
		warmupSteps: warmupSteps,
		totalSteps:  totalSteps,
		currentStep: 0,
		currentLR:   baseLR,
	}
}

// GetLR returns the learning rate for a given step
func (s *CosineAnnealingScheduler) GetLR(step int) float64 {
	if step < s.warmupSteps {
		// Linear warmup
		return s.baseLR * float64(step+1) / float64(s.warmupSteps)
	}

	// Cosine annealing
	progress := float64(step-s.warmupSteps) / float64(s.totalSteps-s.warmupSteps)
	if progress > 1.0 {
		progress = 1.0
	}
	cosine := 0.5 * (1.0 + math.Cos(math.Pi*progress))
	return s.minLR + (s.baseLR-s.minLR)*cosine
}

// Step advances the scheduler by one step
func (s *CosineAnnealingScheduler) Step() {
	s.currentStep++
	s.currentLR = s.GetLR(s.currentStep)
}

// GetCurrentLR returns the current learning rate
func (s *CosineAnnealingScheduler) GetCurrentLR() float64 {
	return s.currentLR
}

// StepLRScheduler implements step-based learning rate decay
type StepLRScheduler struct {
	baseLR      float64
	decayRate   float64
	decaySteps  int
	currentStep int
	currentLR   float64
}

// NewStepLRScheduler creates a new step-based scheduler
func NewStepLRScheduler(baseLR, decayRate float64, decaySteps int) *StepLRScheduler {
	return &StepLRScheduler{
		baseLR:      baseLR,
		decayRate:   decayRate,
		decaySteps:  decaySteps,
		currentStep: 0,
		currentLR:   baseLR,
	}
}

// GetLR returns the learning rate for a given step
func (s *StepLRScheduler) GetLR(step int) float64 {
	numDecays := step / s.decaySteps
	return s.baseLR * math.Pow(s.decayRate, float64(numDecays))
}

// Step advances the scheduler by one step
func (s *StepLRScheduler) Step() {
	s.currentStep++
	s.currentLR = s.GetLR(s.currentStep)
}

// GetCurrentLR returns the current learning rate
func (s *StepLRScheduler) GetCurrentLR() float64 {
	return s.currentLR
}

// ExponentialLRScheduler implements exponential decay
type ExponentialLRScheduler struct {
	baseLR      float64
	decayRate   float64
	currentStep int
	currentLR   float64
}

// NewExponentialLRScheduler creates a new exponential scheduler
func NewExponentialLRScheduler(baseLR, decayRate float64) *ExponentialLRScheduler {
	return &ExponentialLRScheduler{
		baseLR:      baseLR,
		decayRate:   decayRate,
		currentStep: 0,
		currentLR:   baseLR,
	}
}

// GetLR returns the learning rate for a given step
func (s *ExponentialLRScheduler) GetLR(step int) float64 {
	return s.baseLR * math.Pow(s.decayRate, float64(step))
}

// Step advances the scheduler by one step
func (s *ExponentialLRScheduler) Step() {
	s.currentStep++
	s.currentLR = s.GetLR(s.currentStep)
}

// GetCurrentLR returns the current learning rate
func (s *ExponentialLRScheduler) GetCurrentLR() float64 {
	return s.currentLR
}
