package main

import (
	"fmt"
	"image"
	"image/color"
	"os"

	"gocv.io/x/gocv"
)

func main() {
	// Create an 800x800 image (8x8 board with 100px squares)
	width, height := 800, 800
	img := gocv.NewMatWithSize(height, width, gocv.MatTypeCV8UC3)
	defer img.Close()

	// Fill with a checkerboard pattern
	squareSize := 100
	lightSquare := color.RGBA{240, 217, 181, 255} // Light brown
	darkSquare := color.RGBA{181, 136, 99, 255}   // Dark brown

	for row := 0; row < 8; row++ {
		for col := 0; col < 8; col++ {
			// Determine square color (checkerboard pattern)
			var squareColor color.RGBA
			if (row+col)%2 == 0 {
				squareColor = lightSquare
			} else {
				squareColor = darkSquare
			}

			// Fill the square
			for y := row * squareSize; y < (row+1)*squareSize; y++ {
				for x := col * squareSize; x < (col+1)*squareSize; x++ {
					img.SetUCharAt3(y, x, 0, squareColor.B)
					img.SetUCharAt3(y, x, 1, squareColor.G)
					img.SetUCharAt3(y, x, 2, squareColor.R)
				}
			}
		}
	}

	// Draw some simple "pieces" (circles)
	white := color.RGBA{255, 255, 255, 255}
	black := color.RGBA{50, 50, 50, 255}

	// Starting position: white pieces on rows 0-1, black pieces on rows 6-7
	// Row 1: white pawns
	for col := 0; col < 8; col++ {
		x := col*squareSize + squareSize/2
		y := 1*squareSize + squareSize/2
		gocv.Circle(&img, image.Pt(x, y), 30, white, -1)
	}

	// Row 6: black pawns
	for col := 0; col < 8; col++ {
		x := col*squareSize + squareSize/2
		y := 6*squareSize + squareSize/2
		gocv.Circle(&img, image.Pt(x, y), 30, black, -1)
	}

	// Row 0: white back rank (simplified - just rooks and king)
	// Rook a1
	gocv.Circle(&img, image.Pt(squareSize/2, squareSize/2), 35, white, -1)
	// King e1
	gocv.Circle(&img, image.Pt(4*squareSize+squareSize/2, squareSize/2), 40, white, -1)
	// Rook h1
	gocv.Circle(&img, image.Pt(7*squareSize+squareSize/2, squareSize/2), 35, white, -1)

	// Row 7: black back rank (simplified - just rooks and king)
	// Rook a8
	gocv.Circle(&img, image.Pt(squareSize/2, 7*squareSize+squareSize/2), 35, black, -1)
	// King e8
	gocv.Circle(&img, image.Pt(4*squareSize+squareSize/2, 7*squareSize+squareSize/2), 40, black, -1)
	// Rook h8
	gocv.Circle(&img, image.Pt(7*squareSize+squareSize/2, 7*squareSize+squareSize/2), 35, black, -1)

	// Save as PNG
	outFile := "testdata/chess_board.png"
	if ok := gocv.IMWrite(outFile, img); !ok {
		fmt.Printf("Failed to save image to %s\n", outFile)
		os.Exit(1)
	}

	fmt.Printf("Created test chess board image: %s\n", outFile)
	fmt.Printf("Image size: %dx%d\n", width, height)
	fmt.Println("Board layout: Starting position (simplified)")
}
