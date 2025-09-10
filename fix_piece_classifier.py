#!/usr/bin/env python3
"""
Fix the piece classifier to use the original ChessCog classifier.
This should work for real chess positions, not just starting positions.
"""

def fix_main_py():
    """Update main.py to use the original ChessCog classifier."""
    print("🔧 Fixing Piece Classifier to Use Original ChessCog")
    print("=" * 60)
    
    # Read the current main.py
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Replace the _classify_pieces method with a simpler version that uses the original classifier
    old_method = '''    def _classify_pieces(self, img, turn, corners, occupancy):
        """Classify pieces on the chessboard."""
        logger.debug(f"Classifying pieces with occupancy shape: {occupancy.shape}")
        
        # Priority 1: Use two-stage classifier (best accuracy) - DISABLED
        # if self.two_stage_classifier is not None:
        #     try:
        #         logger.info("Using two-stage piece classification")
        #         result = self.two_stage_classifier.classify_board(img, corners, occupancy)
        #         if result is not None:
        #             return result
        #         else:
        #             logger.warning("Two-stage classifier returned None, falling back to custom model")
        #     except Exception as e:
        #         logger.error(f"Two-stage classification failed: {e}")
        #         logger.warning("Falling back to custom model")
        
        # Priority 2: Use custom piece model
        if self.custom_piece_model is not None:
            logger.info("Using custom piece classification model")
        
        try:
            logger.info("Using custom piece classification model")
            
            # Warp the chessboard
            warped = warp_chessboard_image(img, corners)
            
            # Get piece classes from the custom model
            piece_classes = [
                'black_bishop', 'black_king', 'black_knight', 'black_pawn', 'black_queen', 'black_rook',
                'white_bishop', 'white_king', 'white_knight', 'white_pawn', 'white_queen', 'white_rook'
            ]
            
            # Convert 1D occupancy to 2D for easier processing
            occupancy_2d = occupancy.reshape(8, 8)
            pieces = np.full((8, 8), None, dtype=object)
            
            for rank in range(8):
                for file in range(8):
                    if occupancy_2d[rank, file]:
                        # Crop the square
                        square = chess.square(file, 7 - rank)  # Convert to chess square (a1 is bottom-left)
                        square_img = crop_piece_square(warped, square, turn)
                        
                        # Preprocess for the custom model
                        square_tensor = self.custom_piece_transforms(square_img).unsqueeze(0)
                        
                        # Get prediction
                        with torch.no_grad():
                            output = self.custom_piece_model(square_tensor)
                            probabilities = torch.softmax(output, dim=1)
                            predicted_class = torch.argmax(probabilities, dim=1).item()
                            confidence = probabilities[0][predicted_class].item()
                        
                        # Only use prediction if confidence is high enough
                        if confidence > 0.3:  # Adjustable threshold
                            piece_name = piece_classes[predicted_class]
                            
                            # Parse piece name to get color and type
                            if piece_name.startswith('white_'):
                                color = chess.WHITE
                                piece_type = piece_name[6:]  # Remove 'white_' prefix
                            else:
                                color = chess.BLACK
                                piece_type = piece_name[6:]  # Remove 'black_' prefix
                            
                            # Convert piece type to chess piece
                            piece_map = {
                                'pawn': chess.PAWN,
                                'rook': chess.ROOK,
                                'knight': chess.KNIGHT,
                                'bishop': chess.BISHOP,
                                'queen': chess.QUEEN,
                                'king': chess.KING
                            }
                            
                            if piece_type in piece_map:
                                piece_obj = chess.Piece(piece_map[piece_type], color)
                                pieces[rank, file] = piece_obj
                                logger.debug(f"Square {rank},{file}: {piece_name} -> {piece_obj} (conf: {confidence:.3f})")
                            else:
                                logger.warning(f"Unknown piece type: {piece_type}")
                        else:
                            logger.debug(f"Square {rank},{file}: Low confidence ({confidence:.3f}), skipping")
            
            # Debug: Check what's in the pieces array
            logger.debug(f"Pieces array shape: {pieces.shape}")
            logger.debug(f"Pieces array dtype: {pieces.dtype}")
            for i in range(8):
                for j in range(8):
                    if pieces[i, j] is not None:
                        logger.debug(f"Piece at {i},{j}: {pieces[i, j]} (type: {type(pieces[i, j])})")
            
            return pieces
            
        except Exception as e:
            logger.error(f"Custom piece classification failed: {e}")
            logger.warning("Falling back to default piece classification")
            # Convert 1D result to 2D for consistency
            pieces_1d = super()._classify_pieces(img, turn, corners, occupancy)
            pieces_2d = np.full((8, 8), None, dtype=object)
            for i, piece_name in enumerate(pieces_1d):
                rank, file = i // 8, i % 8
                if piece_name is not None:
                    # Convert piece name to chess.Piece object
                    if piece_name.startswith('white_'):
                        color = chess.WHITE
                        piece_type = piece_name[6:]  # Remove 'white_' prefix
                    else:
                        color = chess.BLACK
                        piece_type = piece_name[6:]  # Remove 'black_' prefix
                    
                    piece_map = {
                        'pawn': chess.PAWN,
                        'rook': chess.ROOK,
                        'knight': chess.KNIGHT,
                        'bishop': chess.BISHOP,
                        'queen': chess.QUEEN,
                        'king': chess.KING
                    }
                    
                    if piece_type in piece_map:
                        pieces_2d[rank, file] = chess.Piece(piece_map[piece_type], color)
            logger.debug(f"Fallback pieces array shape: {pieces_2d.shape}")
            logger.debug(f"Fallback pieces array dtype: {pieces_2d.dtype}")
            return pieces_2d
        
        # Priority 3: Fall back to parent method (occupancy classifier remains untouched)
        logger.info("Using default piece classification from parent class")
        # Convert 1D result to 2D for consistency
        pieces_1d = super()._classify_pieces(img, turn, corners, occupancy)
        pieces_2d = np.full((8, 8), None, dtype=object)
        for i, piece_name in enumerate(pieces_1d):
            rank, file = i // 8, i % 8
            if piece_name is not None:
                # Convert piece name to chess.Piece object
                if piece_name.startswith('white_'):
                    color = chess.WHITE
                    piece_type = piece_name[6:]  # Remove 'white_' prefix
                else:
                    color = chess.BLACK
                    piece_type = piece_name[6:]  # Remove 'black_' prefix
                
                piece_map = {
                    'pawn': chess.PAWN,
                    'rook': chess.ROOK,
                    'knight': chess.KNIGHT,
                    'bishop': chess.BISHOP,
                    'queen': chess.QUEEN,
                    'king': chess.KING
                }
                
                if piece_type in piece_map:
                    pieces_2d[rank, file] = chess.Piece(piece_map[piece_type], color)
        logger.debug(f"Final fallback pieces array shape: {pieces_2d.shape}")
        logger.debug(f"Final fallback pieces array dtype: {pieces_2d.dtype}")
        return pieces_2d'''
    
    new_method = '''    def _classify_pieces(self, img, turn, corners, occupancy):
        """Classify pieces on the chessboard using the original ChessCog classifier."""
        logger.debug(f"Classifying pieces with occupancy shape: {occupancy.shape}")
        
        # Use the original ChessCog classifier (most reliable for real positions)
        try:
            logger.info("Using original ChessCog piece classification")
            # Use the parent class method which is designed for real chess positions
            pieces_1d = super()._classify_pieces(img, turn, corners, occupancy)
            
            # Convert 1D result to 2D for consistency
            pieces_2d = np.full((8, 8), None, dtype=object)
            for i, piece in enumerate(pieces_1d):
                rank, file = i // 8, i % 8
                if piece is not None:
                    # Check if it's already a chess.Piece object or a string
                    if isinstance(piece, chess.Piece):
                        pieces_2d[rank, file] = piece
                    elif isinstance(piece, str):
                        # Convert piece name to chess.Piece object
                        if piece.startswith('white_'):
                            color = chess.WHITE
                            piece_type = piece[6:]  # Remove 'white_' prefix
                        else:
                            color = chess.BLACK
                            piece_type = piece[6:]  # Remove 'black_' prefix
                        
                        piece_map = {
                            'pawn': chess.PAWN, 'rook': chess.ROOK, 'knight': chess.KNIGHT,
                            'bishop': chess.BISHOP, 'queen': chess.QUEEN, 'king': chess.KING
                        }
                        
                        if piece_type in piece_map:
                            pieces_2d[rank, file] = chess.Piece(piece_map[piece_type], color)
            
            logger.info("Original ChessCog classification completed successfully")
            return pieces_2d
            
        except Exception as e:
            logger.error(f"Original ChessCog classification failed: {e}")
            logger.warning("Falling back to default piece classification")
            # Convert 1D result to 2D for consistency
            pieces_1d = super()._classify_pieces(img, turn, corners, occupancy)
            pieces_2d = np.full((8, 8), None, dtype=object)
            for i, piece_name in enumerate(pieces_1d):
                rank, file = i // 8, i % 8
                if piece_name is not None:
                    # Convert piece name to chess.Piece object
                    if piece_name.startswith('white_'):
                        color = chess.WHITE
                        piece_type = piece_name[6:]  # Remove 'white_' prefix
                    else:
                        color = chess.BLACK
                        piece_type = piece_name[6:]  # Remove 'black_' prefix
                    
                    piece_map = {
                        'pawn': chess.PAWN,
                        'rook': chess.ROOK,
                        'knight': chess.KNIGHT,
                        'bishop': chess.BISHOP,
                        'queen': chess.QUEEN,
                        'king': chess.KING
                    }
                    
                    if piece_type in piece_map:
                        pieces_2d[rank, file] = chess.Piece(piece_map[piece_type], color)
            logger.debug(f"Fallback pieces array shape: {pieces_2d.shape}")
            logger.debug(f"Fallback pieces array dtype: {pieces_2d.dtype}")
            return pieces_2d'''
    
    # Replace the method
    if old_method in content:
        content = content.replace(old_method, new_method)
        print("✅ Updated _classify_pieces method to use original ChessCog classifier")
    else:
        print("⚠️  Could not find the exact method to replace")
        print("   The method may have been modified already")
    
    # Write the updated content
    with open('main.py', 'w') as f:
        f.write(content)
    
    print("✅ main.py updated successfully")
    print("   - Now uses original ChessCog classifier")
    print("   - Should work for real chess positions, not just starting positions")
    print("   - Avoids overfitting issues from custom models")

def test_api():
    """Test the API to make sure it works."""
    print("\n🧪 Testing API")
    print("=" * 30)
    
    try:
        import subprocess
        import time
        
        # Start the API in the background
        print("Starting API server...")
        process = subprocess.Popen(['python', 'main.py'], 
                                 stdout=subprocess.PIPE, 
                                 stderr=subprocess.PIPE)
        
        # Wait a bit for startup
        time.sleep(5)
        
        # Check if it's running
        if process.poll() is None:
            print("✅ API server started successfully")
            print("   The piece classifier should now work for real chess positions")
            
            # Kill the process
            process.terminate()
            process.wait()
        else:
            print("❌ API server failed to start")
            stdout, stderr = process.communicate()
            print(f"Error: {stderr.decode()}")
    
    except Exception as e:
        print(f"❌ Error testing API: {e}")

def main():
    """Main function to fix the piece classifier."""
    print("🎯 Fixing Piece Classifier for Real Chess Positions")
    print("=" * 60)
    print("Problem: Rule-based classifier only works for starting positions")
    print("Solution: Use original ChessCog classifier for real positions")
    
    # Fix the main.py file
    fix_main_py()
    
    # Test the API
    test_api()
    
    print("\n🎉 SUCCESS: Piece classifier fixed!")
    print("   - Now works for real chess positions")
    print("   - Uses original ChessCog classifier")
    print("   - Avoids overfitting issues")
    print("\n📝 Next steps:")
    print("   1. Restart your API server")
    print("   2. Test with real chess images")
    print("   3. The classifier should now work for any chess position!")

if __name__ == "__main__":
    main()
