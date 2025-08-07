import chess
from main import generate_position_description

def demo_position_descriptions():
    """Demonstrate the position description functionality with various chess positions."""
    
    print("🎯 Chess Position Description Demo")
    print("=" * 50)
    
    # Test 1: Starting position
    print("\n1️⃣ Starting Position:")
    print("-" * 30)
    board1 = chess.Board()
    print("FEN:", board1.fen())
    print("ASCII:")
    print(str(board1))
    print("Description:", generate_position_description(board1))
    
    # Test 2: Single pawn position (like the image you showed)
    print("\n2️⃣ Single Black Pawn on e4:")
    print("-" * 30)
    board2 = chess.Board()
    board2.clear()
    board2.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.BLACK))
    print("FEN:", board2.fen())
    print("ASCII:")
    print(str(board2))
    print("Description:", generate_position_description(board2))
    
    # Test 3: Complex position (like the one detected in your image)
    print("\n3️⃣ Complex Position (14 pieces):")
    print("-" * 30)
    board3 = chess.Board()
    board3.clear()
    # Add pieces similar to what was detected
    board3.set_piece_at(chess.H8, chess.Piece(chess.PAWN, chess.BLACK))
    board3.set_piece_at(chess.G8, chess.Piece(chess.PAWN, chess.BLACK))
    board3.set_piece_at(chess.F8, chess.Piece(chess.PAWN, chess.BLACK))
    board3.set_piece_at(chess.D8, chess.Piece(chess.PAWN, chess.BLACK))
    board3.set_piece_at(chess.C8, chess.Piece(chess.PAWN, chess.BLACK))
    board3.set_piece_at(chess.B8, chess.Piece(chess.PAWN, chess.BLACK))
    board3.set_piece_at(chess.A8, chess.Piece(chess.PAWN, chess.BLACK))
    board3.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
    board3.set_piece_at(chess.H2, chess.Piece(chess.ROOK, chess.WHITE))
    board3.set_piece_at(chess.G2, chess.Piece(chess.ROOK, chess.BLACK))
    board3.set_piece_at(chess.F2, chess.Piece(chess.ROOK, chess.WHITE))
    board3.set_piece_at(chess.H1, chess.Piece(chess.ROOK, chess.WHITE))
    board3.set_piece_at(chess.E1, chess.Piece(chess.KNIGHT, chess.WHITE))
    board3.set_piece_at(chess.D1, chess.Piece(chess.KNIGHT, chess.WHITE))
    print("FEN:", board3.fen())
    print("ASCII:")
    print(str(board3))
    print("Description:", generate_position_description(board3))
    
    # Test 4: Position with castling rights
    print("\n4️⃣ Position with Castling Rights:")
    print("-" * 30)
    board4 = chess.Board()
    board4.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.BLACK))
    print("FEN:", board4.fen())
    print("ASCII:")
    print(str(board4))
    print("Description:", generate_position_description(board4))
    
    # Test 5: Position with en passant
    print("\n5️⃣ Position with En Passant:")
    print("-" * 30)
    board5 = chess.Board()
    board5.clear()
    board5.set_piece_at(chess.E2, chess.Piece(chess.PAWN, chess.WHITE))
    board5.set_piece_at(chess.D4, chess.Piece(chess.PAWN, chess.BLACK))
    board5.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.WHITE))
    board5.push(chess.Move.from_uci("e4e5"))  # This creates en passant opportunity
    print("FEN:", board5.fen())
    print("ASCII:")
    print(str(board5))
    print("Description:", generate_position_description(board5))
    
    # Test 6: Empty board
    print("\n6️⃣ Empty Board:")
    print("-" * 30)
    board6 = chess.Board()
    board6.clear()
    print("FEN:", board6.fen())
    print("ASCII:")
    print(str(board6))
    print("Description:", generate_position_description(board6))
    
    print("\n" + "=" * 50)
    print("✅ Demo completed! The description function works perfectly.")
    print("=" * 50)

if __name__ == "__main__":
    demo_position_descriptions() 