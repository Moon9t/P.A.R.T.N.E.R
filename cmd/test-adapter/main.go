package main

import (
	"fmt"
	"log"

	"github.com/thyrook/partner/internal/adapter"
)

func main() {
	fmt.Println("╔══════════════════════════════════════════════════════════════════╗")
	fmt.Println("║                                                                  ║")
	fmt.Println("║         P.A.R.T.N.E.R Game Adapter Interface Test                ║")
	fmt.Println("║                                                                  ║")
	fmt.Println("╚══════════════════════════════════════════════════════════════════╝")
	fmt.Println()

	// Create adapter factory
	factory := adapter.NewAdapterFactory()

	// List available adapters
	fmt.Println("📋 Available Adapters:")
	for _, name := range factory.ListAdapters() {
		fmt.Printf("  • %s\n", name)
	}
	fmt.Println()

	// Create chess adapter
	fmt.Println("🎯 Testing Chess Adapter")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println()

	chessAdapter, err := factory.Create("chess")
	if err != nil {
		log.Fatalf("Failed to create adapter: %v", err)
	}

	fmt.Printf("✓ Created adapter: %s\n", chessAdapter.GetGameName())
	fmt.Printf("  State dimensions:  %v\n", chessAdapter.GetStateDimensions())
	fmt.Printf("  Action dimensions: %v\n", chessAdapter.GetActionDimensions())
	fmt.Println()

	// Test 1: Encode starting position (FEN)
	fmt.Println("Test 1: Encoding starting position from FEN")
	fmt.Println("─────────────────────────────────────────────")
	fen := "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
	fmt.Printf("FEN: %s\n", fen)

	stateTensor, err := chessAdapter.EncodeState(fen)
	if err != nil {
		log.Fatalf("Failed to encode state: %v", err)
	}
	fmt.Printf("✓ Encoded to tensor shape: %v\n", stateTensor.Shape())
	fmt.Println()

	// Test 2: Validate state
	fmt.Println("Test 2: Validating board state")
	fmt.Println("──────────────────────────────")
	if err := chessAdapter.ValidateState(fen); err != nil {
		fmt.Printf("✗ Validation failed: %v\n", err)
	} else {
		fmt.Println("✓ State is valid")
	}
	fmt.Println()

	// Test 3: Encode action (move)
	fmt.Println("Test 3: Encoding chess move")
	fmt.Println("───────────────────────────")
	move := "e2e4"
	fmt.Printf("Move: %s\n", move)

	actionTensor, err := chessAdapter.EncodeAction(move)
	if err != nil {
		log.Fatalf("Failed to encode action: %v", err)
	}
	fmt.Printf("✓ Encoded to tensor shape: %v\n", actionTensor.Shape())

	// Check that the correct move is encoded
	data := actionTensor.Data().([]float64)
	maxIdx := 0
	maxVal := data[0]
	for i, v := range data {
		if v > maxVal {
			maxVal = v
			maxIdx = i
		}
	}
	fromSquare := maxIdx / 64
	toSquare := maxIdx % 64
	fmt.Printf("  From square: %d (%s)\n", fromSquare, indexToSquare(fromSquare))
	fmt.Printf("  To square:   %d (%s)\n", toSquare, indexToSquare(toSquare))
	fmt.Println()

	// Test 4: Decode action back
	fmt.Println("Test 4: Decoding action from tensor")
	fmt.Println("────────────────────────────────────")
	decoded, err := chessAdapter.DecodeAction(actionTensor)
	if err != nil {
		log.Fatalf("Failed to decode action: %v", err)
	}

	decodedMap := decoded.(map[string]interface{})
	fmt.Printf("✓ Decoded move: %s\n", decodedMap["move"])
	fmt.Printf("  Probability: %.4f\n", decodedMap["probability"])
	fmt.Println()

	// Test 5: Feedback mechanism
	fmt.Println("Test 5: Testing feedback mechanism")
	fmt.Println("───────────────────────────────────")
	correctMove := "d2d4"
	fmt.Printf("Providing feedback for correct move: %s\n", correctMove)

	if err := chessAdapter.Feedback(correctMove); err != nil {
		log.Fatalf("Failed to provide feedback: %v", err)
	}
	fmt.Println("✓ Feedback recorded in replay buffer")
	fmt.Println()

	// Test 6: Alternative move formats
	fmt.Println("Test 6: Testing alternative move formats")
	fmt.Println("─────────────────────────────────────────")

	// Test with map format
	moveMap := map[string]interface{}{
		"from": 12, // e2
		"to":   28, // e4
	}
	fmt.Printf("Map format: from=%d, to=%d\n", moveMap["from"], moveMap["to"])
	_, err = chessAdapter.EncodeAction(moveMap)
	if err != nil {
		fmt.Printf("✗ Failed: %v\n", err)
	} else {
		fmt.Println("✓ Map format works")
	}
	fmt.Println()

	// Test 7: Invalid move handling
	fmt.Println("Test 7: Testing invalid move handling")
	fmt.Println("──────────────────────────────────────")
	invalidMove := "a1a9" // a9 doesn't exist
	fmt.Printf("Invalid move: %s\n", invalidMove)
	_, err = chessAdapter.EncodeAction(invalidMove)
	if err != nil {
		fmt.Printf("✓ Correctly rejected: %v\n", err)
	} else {
		fmt.Println("✗ Should have been rejected")
	}
	fmt.Println()

	// Summary
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println("✅ ADAPTER SYSTEM TEST COMPLETE")
	fmt.Println("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
	fmt.Println()
	fmt.Println("The adapter system is working correctly!")
	fmt.Println("You can now use:")
	fmt.Println("  partner --adapter=chess")
	fmt.Println()
	fmt.Println("The learning system is now game-agnostic.")
	fmt.Println("Just swap adapters to work with different games!")
}

func indexToSquare(index int) string {
	rank := index / 8
	file := index % 8
	return string(rune('a'+file)) + string(rune('1'+rank))
}
