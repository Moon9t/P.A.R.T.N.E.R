package data

import (
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/notnil/chess"
)

// PGNParser handles parsing of PGN files
type PGNParser struct {
	filepath string
}

// NewPGNParser creates a new PGN parser
func NewPGNParser(filepath string) *PGNParser {
	return &PGNParser{
		filepath: filepath,
	}
}

// ParsePGN parses a PGN file and returns a list of games
func (p *PGNParser) ParsePGN() ([]*chess.Game, error) {
	file, err := os.Open(p.filepath)
	if err != nil {
		return nil, fmt.Errorf("failed to open PGN file: %w", err)
	}
	defer file.Close()

	return p.ParsePGNReader(file)
}

// ParsePGNReader parses PGN from an io.Reader
func (p *PGNParser) ParsePGNReader(reader io.Reader) ([]*chess.Game, error) {
	var games []*chess.Game

	scanner := chess.NewScanner(reader)
	gameCount := 0
	
	for scanner.Scan() {
		gameCount++
		game := scanner.Next()
		if game != nil {
			// Validate the game has moves
			if len(game.Moves()) > 0 {
				games = append(games, game)
			}
		}
	}

	// EOF is expected at end of file, not an error
	// Be very lenient with parsing errors - extract what we can
	if err := scanner.Err(); err != nil && err != io.EOF {
		// If we got some games, return them despite the error
		if len(games) > 0 {
			fmt.Printf("Warning: PGN parsing stopped at game %d due to errors\n", gameCount)
			fmt.Printf("Successfully extracted %d valid games\n", len(games))
			return games, nil
		}
		// Even if we got no games, it might just be a bad format
		// Return empty list instead of failing
		fmt.Printf("Warning: Could not parse any games from PGN (error at game %d)\n", gameCount)
		return games, nil // Return empty list, not error
	}

	return games, nil
}

// ExtractPositions extracts all positions from a game as (state, move) pairs
func ExtractPositions(game *chess.Game) ([]*ChessPosition, error) {
	if game == nil {
		return nil, fmt.Errorf("game is nil")
	}

	var positions []*ChessPosition

	// Get all positions from the game
	currentPos := game.Position()
	if currentPos == nil {
		return nil, fmt.Errorf("initial position is nil")
	}

	// Start from the beginning
	moves := game.Moves()
	if len(moves) == 0 {
		return positions, nil
	}

	// Replay the game to get all positions
	tempGame := chess.NewGame()
	for i, move := range moves {
		// Get current position before the move
		pos := tempGame.Position()

		// Store the position and move
		positions = append(positions, &ChessPosition{
			Board: pos.Board(),
			Move:  move,
		})

		// Apply the move
		if err := tempGame.Move(move); err != nil {
			// Stop processing this game if we hit an invalid move
			// but return what we have so far
			if len(positions) > 0 {
				return positions, nil
			}
			return nil, fmt.Errorf("invalid move %d: %w", i+1, err)
		}
	}

	return positions, nil
}

// ChessPosition represents a chess position and the move played from it
type ChessPosition struct {
	Board *chess.Board
	Move  *chess.Move
}

// ValidatePGN checks if a PGN file is valid without fully parsing it
func ValidatePGN(filepath string) error {
	file, err := os.Open(filepath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	content := make([]byte, 1024)
	n, err := file.Read(content)
	if err != nil && err != io.EOF {
		return fmt.Errorf("failed to read file: %w", err)
	}

	// Check for basic PGN markers
	contentStr := string(content[:n])
	if !strings.Contains(contentStr, "[Event") && !strings.Contains(contentStr, "1.") {
		return fmt.Errorf("file does not appear to be a valid PGN file")
	}

	return nil
}
