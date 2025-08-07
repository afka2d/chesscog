import requests
import json
import base64
import os

def test_position_description_api():
    """Test the new position description API endpoint."""
    
    # API endpoint
    API_URL = "http://localhost:8001/recognize_chess_position_with_description"
    
    # Test image path
    IMAGE_PATH = "IMG_4540.jpeg"
    
    print("Testing Chess Position Description API")
    print("=" * 50)
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image file {IMAGE_PATH} not found")
        return
    
    try:
        # Prepare the request
        with open(IMAGE_PATH, "rb") as f:
            files = {"image": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
            data = {
                "color": "white",
                "debug_image_width": 800,
                "debug_image_height": 600
            }
            
            print(f"Uploading image: {IMAGE_PATH}")
            response = requests.post(API_URL, files=files, data=data)
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            print("\n✅ API Response Success!")
            print("=" * 50)
            
            # Display main results
            print(f"FEN: {result['fen']}")
            print(f"Legal Position: {result['legal_position']}")
            print(f"Lichess URL: {result['lichess_url']}")
            
            # Display the new position description
            print("\n📝 POSITION DESCRIPTION:")
            print("-" * 30)
            print(result['position_description'])
            
            # Display ASCII board
            print("\n🎯 ASCII BOARD:")
            print("-" * 30)
            print(result['ascii'])
            
            # Display piece count
            piece_count = len([c for c in result['fen'].split()[0] if c.isalpha()])
            print(f"\n📊 Total pieces detected: {piece_count}")
            
            # Display processing info
            print(f"\n⏱️ Processing time: {result['processing_time']}")
            
            # Display debug images info
            debug_images = result.get('debug_images', {})
            print(f"\n🖼️ Debug images generated: {len(debug_images)}")
            for key in debug_images.keys():
                print(f"  - {key}")
            
            # Display image info
            image_info = result.get('image_info', {})
            print(f"\n📸 Image info:")
            print(f"  - Filename: {image_info.get('filename', 'N/A')}")
            print(f"  - Size: {image_info.get('size_bytes', 0)} bytes")
            print(f"  - Shape: {image_info.get('shape', 'N/A')}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Error message: {response.text}")
            
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()

def test_description_function():
    """Test the description generation function directly."""
    
    print("\n" + "=" * 50)
    print("Testing Description Generation Function")
    print("=" * 50)
    
    try:
        # Import the function
        import sys
        sys.path.append('.')
        
        # We'll need to import the function from main.py
        # For now, let's test with a simple chess board
        import chess
        
        # Create a test board with some pieces
        board = chess.Board()
        board.clear()  # Clear the board
        
        # Add some pieces to test the description
        board.set_piece_at(chess.E4, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.E2, chess.Piece(chess.QUEEN, chess.WHITE))
        board.set_piece_at(chess.F6, chess.Piece(chess.PAWN, chess.WHITE))
        board.set_piece_at(chess.A3, chess.Piece(chess.PAWN, chess.BLACK))
        board.set_piece_at(chess.C6, chess.Piece(chess.PAWN, chess.BLACK))
        
        print("Test Board FEN:", board.fen())
        print("Test Board ASCII:")
        print(str(board))
        
        # Test the description function
        from main import generate_position_description
        description = generate_position_description(board, "white")
        
        print("\n📝 Generated Description:")
        print("-" * 30)
        print(description)
        
    except Exception as e:
        print(f"❌ Function test failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the API endpoint
    test_position_description_api()
    
    # Test the description function
    test_description_function() 