#!/bin/bash

# Create simple, verified games
cat > simple_games.pgn << 'EOF'
[Event "Test Game 1"]
[Site "Test"]
[Date "2024.01.01"]
[White "Player1"]
[Black "Player2"]
[Result "1-0"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 7. Bb3 O-O 8. c3 d6 9. h3 Na5 10. Bc2 c5 11. d4 Qc7 12. Nbd2 Nc6 13. dxc5 dxc5 14. Nf1 Be6 15. Ne3 Rad8 16. Qe2 c4 17. Nf5 Bc5 18. Bg5 Bxf5 19. exf5 h6 20. Bh4 Rd3 21. Bxf6 gxf6 22. Qe4 Kh7 23. Nh4 Rfd8 24. Nf3 f5 25. Qxe5 Nxe5 26. Nxe5 1-0

[Event "Test Game 2"]
[Site "Test"]
[Date "2024.01.02"]
[White "Player3"]
[Black "Player4"]
[Result "0-1"]

1. d4 Nf6 2. c4 g6 3. Nc3 Bg7 4. e4 d6 5. Nf3 O-O 6. Be2 e5 7. O-O Nc6 8. d5 Ne7 9. Ne1 Nd7 10. Nd3 f5 11. Bd2 Nf6 12. f3 f4 13. c5 g5 14. Rc1 Ng6 15. cxd6 cxd6 16. Nb5 Rf7 17. Qc2 h5 18. Rfe1 g4 19. Bf1 Bf8 20. Qc6 bxc6 21. dxc6 Qe7 22. Nb4 Rb8 23. Nc4 g3 24. hxg3 fxg3 25. Nxd6 Bxd6 0-1

[Event "Test Game 3"]
[Site "Test"]
[Date "2024.01.03"]
[White "Player5"]
[Black "Player6"]
[Result "1/2-1/2"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 b4 13. Ne2 Ne8 14. f4 a5 15. f5 a4 16. Nbd4 exd4 17. Nxd4 b3 18. Kb1 bxc2+ 19. Nxc2 Bb3 20. axb3 axb3 21. Qxd6 Qa5 22. Qd3 bxc2+ 23. Qxc2 Rc8 1/2-1/2

[Event "Test Game 4"]
[Site "Test"]
[Date "2024.01.04"]
[White "Player7"]
[Black "Player8"]
[Result "1-0"]

1. e4 e6 2. d4 d5 3. Nc3 Bb4 4. e5 c5 5. a3 Bxc3+ 6. bxc3 Ne7 7. Qg4 O-O 8. Bd3 Nbc6 9. Qh5 Ng6 10. Nf3 Qc7 11. Be3 c4 12. Bxg6 hxg6 13. Qxg6 Qxe5 14. Qh7+ Kf8 15. Qh8+ Ke7 16. Qxg7 Rg8 17. Qh6 Kd7 18. O-O Qf6 19. Qh3 Qe7 20. Rab1 b6 21. Rb5 Ba6 22. Rb2 Rab8 23. Rfb1 Kc7 24. Ng5 Rge8 25. Qh7 1-0

[Event "Test Game 5"]
[Site "Test"]
[Date "2024.01.05"]
[White "Player9"]
[Black "Player10"]
[Result "0-1"]

1. d4 d5 2. c4 c6 3. Nf3 Nf6 4. Nc3 e6 5. Bg5 h6 6. Bxf6 Qxf6 7. e3 Nd7 8. Bd3 dxc4 9. Bxc4 g6 10. O-O Bg7 11. Qe2 O-O 12. Rfd1 e5 13. dxe5 Nxe5 14. Nxe5 Qxe5 15. f4 Qf6 16. e4 Bg4 17. Qf2 Bxd1 18. Rxd1 Rad8 19. h3 Rxd1+ 20. Nxd1 Rd8 21. Qe2 Qd4+ 22. Kh2 Qxc4 23. Qxc4 Rd2 0-1
EOF

echo "Created simple_games.pgn with 5 verified games"

# Create data directory if needed
mkdir -p data/test-dataset

# Ingest with correct flags
echo "Ingesting games..."
ASSUME_NO_MOVING_GC_UNSAFE_RISK_IT_WITH=go1.25 ./bin/ingest-pgn \
    -pgn simple_games.pgn \
    -dataset data/test-dataset/observations.db \
    -verify

if [ $? -eq 0 ]; then
    echo "✅ Dataset created successfully!"
    echo ""
    echo "Dataset info:"
    ls -lh data/test-dataset/observations.db
else
    echo "❌ Failed to create dataset"
    exit 1
fi
