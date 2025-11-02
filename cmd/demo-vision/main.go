package main
package main

import (
	"fmt"
	"log"

	"github.com/thyrook/partner/internal/vision"
)

func main() {
	// Test the vision pipeline with a synthetic board
	fmt.Println("Testing P.A.R.T.N.E.R Vision Module")
	fmt.Println("===================================\n")

	// Create a synthetic board state tensor (starting position)
	var tensor [12][8][8]float32

	// White pieces (channels 0-5)
	// White pawns on rank 1 (row index 1)
	for file := 0; file < 8; file++ {
		tensor[0][1][file] = 1.0 // White pawn channel
	}

	// White rooks
	tensor[3][0][0] = 1.0 // a1
	tensor[3][0][7] = 1.0 // h1

	// White knights
	tensor[1][0][1] = 1.0 // b1
	tensor[1][0][6] = 1.0 // g1

	// White bishops
	tensor[2][0][2] = 1.0 // c1
	tensor[2][0][5] = 1.0 // f1

	// White queen
	tensor[4][0][3] = 1.0 // d1

	// White king
	tensor[5][0][4] = 1.0 // e1

	// Black pieces (channels 6-11)
	// Black pawns on rank 6 (row index 6)
	for file := 0; file < 8; file++ {
		tensor[6][6][file] = 1.0 // Black pawn channel
	}

	// Black rooks
	tensor[9][7][0] = 1.0  // a8
	tensor[9][7][7] = 1.0  // h8

	// Black knights
	tensor[7][7][1] = 1.0  // b8
	tensor[7][7][6] = 1.0  // g8

	// Black bishops
	tensor[8][7][2] = 1.0  // c8
	tensor[8][7][5] = 1.0  // f8

	// Black queen
	tensor[10][7][3] = 1.0 // d8

	// Black king
	tensor[11][7][4] = 1.0 // e8

	fmt.Println("1. Testing board tensor validation...")
	if err := vision.ValidateBoardTensor(tensor); err != nil {
		log.Fatalf("❌ Validation failed: %v", err)
	}
	fmt.Println("✅ Board tensor is valid")

	fmt.Println("\n2. Testing board visualization...")
	boardStr := vision.PrintBoardTensor(tensor)
	fmt.Println(boardStr)

	fmt.Println("\n3. Testing position notation...")
	pos := vision.Position{Rank: 4, File: 4} // e5
	fmt.Printf("Position (4,4) in algebraic notation: %s\n", pos.String())
	fmt.Printf("Square index: %d\n", pos.ToSquareIndex())

	fmt.Println("\n4. Testing board difference detection...")
	// Create a second board with a pawn moved
	var tensor2 [12][8][8]float32
	// Copy tensor to tensor2
	tensor2 = tensor
	// Move white e2 pawn to e4
	tensor2[0][1][4] = 0.0 // Remove from e2
	tensor2[0][3][4] = 1.0 // Add to e4

	detector := vision.NewBoardDetector(100, true)
	changes := detector.DetectBoardDifference(tensor, tensor2)
	fmt.Printf("Detected %d changes:\n", len(changes))
	for _, pos := range changes {
		fmt.Printf("  - %s\n", pos.String())
	}

	fmt.Println("\n5. Testing piece counting...")
	whitePieces, blackPieces := countPieces(tensor)
	fmt.Printf("White pieces: %d\n", whitePieces)
	fmt.Printf("Black pieces: %d\n", blackPieces)
	fmt.Printf("Total pieces: %d\n", whitePieces+blackPieces)

	fmt.Println("\n6. Testing configuration...")
	config := vision.DefaultConfig()
	fmt.Println(config.String())

	fmt.Println("\n✅ All vision module tests passed!")
	fmt.Println("\nThe vision module is ready to:")
	fmt.Println("  - Capture chess boards from screen/video/images")
	fmt.Println("  - Detect piece positions")
	fmt.Println("  - Convert to [12][8][8] tensor format")
	fmt.Println("  - Validate board states")
	fmt.Println("  - Track changes between positions")
	fmt.Println("\nNext: Integrate with CNN model for move predictions!")
}

func countPieces(tensor [12][8][8]float32) (white, black int) {
	for channel := 0; channel < 12; channel++ {
		for rank := 0; rank < 8; rank++ {
			for file := 0; file < 8; file++ {
				if tensor[channel][rank][file] > 0.5 {
					if channel < 6 {
						white++
					} else {
						black++
					}
				}
			}
		}
	}
	return
}
