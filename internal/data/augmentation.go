package data

import (
	"math/rand"
)

// AugmentationConfig controls data augmentation behavior
type AugmentationConfig struct {
	HorizontalFlipProb float64 // Probability of horizontal flip (0.0-1.0)
	VerticalFlipProb   float64 // Probability of vertical flip (0.0-1.0)
	ColorInvertProb    float64 // Probability of color inversion (0.0-1.0)
	Rotation90Prob     float64 // Probability of 90° rotation (0.0-1.0)
	Enabled            bool    // Master switch for augmentation
}

// DefaultAugmentationConfig returns sensible defaults
func DefaultAugmentationConfig() AugmentationConfig {
	return AugmentationConfig{
		HorizontalFlipProb: 0.5,
		VerticalFlipProb:   0.0, // Chess boards are rarely flipped vertically
		ColorInvertProb:    0.25,
		Rotation90Prob:     0.0, // 90° rotation changes the game orientation
		Enabled:            true,
	}
}

// FlipHorizontal flips the board left-to-right
func FlipHorizontal(tensor [12][8][8]float32, fromSquare, toSquare int) ([12][8][8]float32, int, int) {
	var flipped [12][8][8]float32

	for c := 0; c < 12; c++ {
		for r := 0; r < 8; r++ {
			for f := 0; f < 8; f++ {
				flipped[c][r][7-f] = tensor[c][r][f]
			}
		}
	}

	// Adjust move coordinates
	// fromSquare = rank*8 + file, so file = fromSquare % 8
	fromRank := fromSquare / 8
	fromFile := fromSquare % 8
	toRank := toSquare / 8
	toFile := toSquare % 8

	newFromSquare := fromRank*8 + (7 - fromFile)
	newToSquare := toRank*8 + (7 - toFile)

	return flipped, newFromSquare, newToSquare
}

// InvertColors swaps white and black pieces and flips ranks
func InvertColors(tensor [12][8][8]float32, fromSquare, toSquare int) ([12][8][8]float32, int, int) {
	var inverted [12][8][8]float32

	// Swap white pieces (channels 0-5) with black pieces (channels 6-11)
	// Also flip ranks (7-r instead of r)
	for c := 0; c < 6; c++ {
		for r := 0; r < 8; r++ {
			for f := 0; f < 8; f++ {
				// White (c) becomes Black (c+6) at flipped rank
				inverted[c+6][7-r][f] = tensor[c][r][f]
				// Black (c+6) becomes White (c) at flipped rank
				inverted[c][7-r][f] = tensor[c+6][r][f]
			}
		}
	}

	// Adjust move coordinates (flip ranks)
	fromRank := fromSquare / 8
	fromFile := fromSquare % 8
	toRank := toSquare / 8
	toFile := toSquare % 8

	newFromSquare := (7-fromRank)*8 + fromFile
	newToSquare := (7-toRank)*8 + toFile

	return inverted, newFromSquare, newToSquare
}

// AugmentEntry applies random augmentations to a single data entry
func AugmentEntry(entry *DataEntry, config AugmentationConfig) *DataEntry {
	if !config.Enabled {
		return entry
	}

	// Convert flat array to tensor
	tensor, err := FlatArrayToTensor(entry.StateTensor)
	if err != nil {
		return entry // Return original on error
	}

	fromSquare := entry.FromSquare
	toSquare := entry.ToSquare

	// Apply horizontal flip
	if rand.Float64() < config.HorizontalFlipProb {
		tensor, fromSquare, toSquare = FlipHorizontal(tensor, fromSquare, toSquare)
	}

	// Apply color inversion
	if rand.Float64() < config.ColorInvertProb {
		tensor, fromSquare, toSquare = InvertColors(tensor, fromSquare, toSquare)
	}

	// Convert back to flat array
	flatTensor := TensorToFlatArray(tensor)

	return &DataEntry{
		StateTensor: flatTensor,
		FromSquare:  fromSquare,
		ToSquare:    toSquare,
		GameID:      entry.GameID,
		MoveNumber:  entry.MoveNumber,
	}
}

// AugmentBatch applies augmentation to a batch of entries
// Returns original entries plus augmented versions
func AugmentBatch(entries []*DataEntry, config AugmentationConfig) []*DataEntry {
	if !config.Enabled {
		return entries
	}

	augmented := make([]*DataEntry, 0, len(entries)*2)

	for _, entry := range entries {
		// Always include original
		augmented = append(augmented, entry)

		// Add augmented version
		augmentedEntry := AugmentEntry(entry, config)
		if augmentedEntry != entry { // Only add if actually augmented
			augmented = append(augmented, augmentedEntry)
		}
	}

	return augmented
}
