"""
Simple test script to validate the chess analysis pipeline
Creates a small sample PGN and tests the analyzer
"""

import chess.pgn
from io import StringIO
from chess_analyzer import ChessGameAnalyzer, PlaystyleLabeler

# Sample PGN game
SAMPLE_GAME = """
[Event "Rated Blitz game"]
[Site "https://lichess.org/abc123"]
[Date "2024.01.15"]
[White "AggressivePlayer"]
[Black "PositionalPlayer"]
[Result "1-0"]
[WhiteElo "2200"]
[BlackElo "2180"]
[Opening "Sicilian Defense"]

1. e4 c5 2. Nf3 d6 3. d4 cxd4 4. Nxd4 Nf6 5. Nc3 a6 6. Be3 e5 7. Nb3 Be6 
8. f3 Be7 9. Qd2 O-O 10. O-O-O Nbd7 11. g4 b5 12. g5 b4 13. Ne2 Ne8 
14. f4 a5 15. f5 Bc4 16. Nbd4 exd4 17. Nxd4 a4 18. Kb1 Qa5 19. Qf2 Rc8 
20. Bd3 Bxd3 21. Rxd3 Nc5 22. Rg3 Qc7 23. Rf1 Kh8 24. f6 gxf6 25. gxf6 
Bg5 26. Bxg5 Rg8 27. Qh4 1-0
"""


def test_analyzer(stockfish_path: str):
    """Test the analyzer with a sample game"""
    print("="*50)
    print("TESTING CHESS ANALYZER PIPELINE")
    print("="*50)
    
    # Parse sample game
    pgn = StringIO(SAMPLE_GAME)
    game = chess.pgn.read_game(pgn)
    
    print("\nGame Info:")
    print(f"White: {game.headers['White']} ({game.headers['WhiteElo']})")
    print(f"Black: {game.headers['Black']} ({game.headers['BlackElo']})")
    print(f"Opening: {game.headers['Opening']}")
    print(f"Result: {game.headers['Result']}")
    
    # Initialize analyzer
    print("\nInitializing Stockfish...")
    try:
        analyzer = ChessGameAnalyzer(stockfish_path, depth=15)
        print("Stockfish loaded successfully")
    except Exception as e:
        print(f"Error loading Stockfish: {e}")
        print("\nMake sure to provide correct Stockfish path:")
        print("python test_pipeline.py /path/to/stockfish")
        return
    
    # Analyze White's play
    print("\n" + "-"*50)
    print("Analyzing White's playstyle...")
    print("-"*50)
    
    try:
        white_features = analyzer.analyze_game(game, chess.WHITE)
        white_label = PlaystyleLabeler.label_game(white_features)
        white_summary = PlaystyleLabeler.get_feature_summary(white_features)
        
        print(f"\nWhite's Playstyle: {white_label.upper()}")
        print("\nKey Metrics:")
        for key, value in white_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error analyzing White: {e}")
        return
    
    # Analyze Black's play
    print("\n" + "-"*50)
    print("Analyzing Black's playstyle...")
    print("-"*50)
    
    try:
        black_features = analyzer.analyze_game(game, chess.BLACK)
        black_label = PlaystyleLabeler.label_game(black_features)
        black_summary = PlaystyleLabeler.get_feature_summary(black_features)
        
        print(f"\nBlack's Playstyle: {black_label.upper()}")
        print("\nKey Metrics:")
        for key, value in black_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")
        
    except Exception as e:
        print(f"Error analyzing Black: {e}")
        return
    
    print("\n" + "="*50)
    print("TEST COMPLETED SUCCESSFULLY!")
    print("="*50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python test_pipeline.py /path/to/stockfish")
        sys.exit(1)
    
    stockfish_path = sys.argv[1]
    test_analyzer(stockfish_path)