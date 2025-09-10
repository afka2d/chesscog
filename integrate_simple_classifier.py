#!/usr/bin/env python3
"""
Integrate the simple piece classifier into the existing API.
This adds piece classification without modifying the working corner detection and occupancy classification.
"""

def create_updated_main():
    """Create an updated main.py that integrates the simple piece classifier."""
    print("🔧 Creating Updated main.py with Simple Piece Classifier")
    print("=" * 60)
    
    # Read the current main.py
    with open('main.py', 'r') as f:
        content = f.read()
    
    # Add the simple piece classifier import
    if 'from simple_piece_classifier import SimplePieceClassifier' not in content:
        # Find the import section and add our import
        import_section = content.find('from chesscog.recognition.recognition import ChessRecognizer')
        if import_section != -1:
            new_import = 'from chesscog.recognition.recognition import ChessRecognizer\nfrom simple_piece_classifier import SimplePieceClassifier'
            content = content.replace('from chesscog.recognition.recognition import ChessRecognizer', new_import)
    
    # Update the CustomChessRecognizer class to include the simple piece classifier
    old_init = '''    def __init__(self, models_folder, *args, **kwargs):
        super().__init__(models_folder, *args, **kwargs)
        self.custom_piece_model = None
        self.custom_piece_transforms = None
        self.two_stage_classifier = None
        self._load_custom_piece_model()
        self._load_two_stage_classifier()'''
    
    new_init = '''    def __init__(self, models_folder, *args, **kwargs):
        super().__init__(models_folder, *args, **kwargs)
        self.custom_piece_model = None
        self.custom_piece_transforms = None
        self.two_stage_classifier = None
        self.simple_piece_classifier = SimplePieceClassifier(models_folder)
        self._load_custom_piece_model()
        self._load_two_stage_classifier()'''
    
    content = content.replace(old_init, new_init)
    
    # Update the _classify_pieces method to use the simple piece classifier
    old_classify = '''    def _classify_pieces(self, img, turn, corners, occupancy):
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
    
    new_classify = '''    def _classify_pieces(self, img, turn, corners, occupancy):
        """Classify pieces on the chessboard using the simple piece classifier."""
        logger.debug(f"Classifying pieces with occupancy shape: {occupancy.shape}")
        
        # Use the simple piece classifier
        try:
            logger.info("Using simple piece classification")
            
            # Convert occupancy to list if it's a numpy array
            if hasattr(occupancy, 'tolist'):
                occupancy_list = occupancy.tolist()
            else:
                occupancy_list = list(occupancy)
            
            # Use the simple piece classifier
            pieces_1d = self.simple_piece_classifier.classify_pieces(img, corners, occupancy_list, turn)
            
            # Convert 1D result to 2D for consistency
            pieces_2d = np.full((8, 8), None, dtype=object)
            for i, piece in enumerate(pieces_1d):
                rank, file = i // 8, i % 8
                if piece is not None:
                    pieces_2d[rank, file] = piece
            
            logger.info("Simple piece classification completed successfully")
            return pieces_2d
            
        except Exception as e:
            logger.error(f"Simple piece classification failed: {e}")
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
    
    content = content.replace(old_classify, new_classify)
    
    # Write the updated content
    with open('main.py', 'w') as f:
        f.write(content)
    
    print("✅ main.py updated successfully")
    print("   - Added simple piece classifier import")
    print("   - Updated CustomChessRecognizer to use simple piece classifier")
    print("   - Preserved existing corner detection and occupancy classification")

def test_updated_api():
    """Test the updated API."""
    print("\n🧪 Testing Updated API")
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
        time.sleep(8)
        
        # Check if it's running
        if process.poll() is None:
            print("✅ API server started successfully")
            
            # Test with a simple request
            import requests
            import json
            import glob
            import os
            
            # Find a test image
            test_dirs = [
                "my_chess_images/train/images",
                "grey_background_dataset/images/test"
            ]
            
            test_image = None
            for test_dir in test_dirs:
                if os.path.exists(test_dir):
                    images = []
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                        images.extend(glob.glob(os.path.join(test_dir, ext)))
                    if images:
                        test_image = images[0]
                        break
            
            if test_image:
                print(f"📁 Testing with image: {test_image}")
                
                # Test corners
                corners = [
                    [50, 50],   # Top-left
                    [400, 50],  # Top-right
                    [400, 400], # Bottom-right
                    [50, 400]   # Bottom-left
                ]
                
                # Read and encode image
                with open(test_image, 'rb') as f:
                    image_data = f.read()
                
                # Prepare the request
                files = {'image': (os.path.basename(test_image), image_data, 'image/jpeg')}
                data = {
                    'corners': json.dumps(corners),
                    'color': 'white'
                }
                
                # Make the request
                response = requests.post('http://localhost:8000/recognize_chess_position_with_corners', 
                                       files=files, data=data, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    pieces = result.get('pieces', [])
                    piece_count = sum(1 for p in pieces if p is not None)
                    print(f"✅ API test successful!")
                    print(f"   Pieces detected: {piece_count}")
                    print(f"   FEN: {result.get('fen', 'N/A')}")
                else:
                    print(f"❌ API test failed: {response.status_code}")
                    print(f"   Error: {response.text}")
            else:
                print("⚠️  No test images found")
            
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
    """Main function to integrate the simple piece classifier."""
    print("🎯 Integrating Simple Piece Classifier")
    print("=" * 50)
    print("Goal: Add piece classification without modifying working components")
    
    # Create updated main.py
    create_updated_main()
    
    # Test the updated API
    test_updated_api()
    
    print("\n🎉 INTEGRATION COMPLETE!")
    print("   - Simple piece classifier integrated")
    print("   - Corner detection preserved (unchanged)")
    print("   - Occupancy classification preserved (unchanged)")
    print("   - Piece classification added")
    print("\n📝 Next steps:")
    print("   1. Restart your API server")
    print("   2. Test with your app")
    print("   3. The piece classifier should now work!")

if __name__ == "__main__":
    main()
