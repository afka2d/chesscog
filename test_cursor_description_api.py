import requests
import json
import os

def test_cursor_description_api():
    """Test the new Cursor description-based API endpoint."""
    
    # API endpoint
    API_URL = "http://localhost:8001/recognize_chess_position_with_cursor_description"
    
    # Test image path
    IMAGE_PATH = "IMG_4540.jpeg"
    
    # Example Cursor description (this would come from Cursor's image analysis)
    CURSOR_DESCRIPTION = """
    This image displays a chess board with several pieces, viewed from a slightly elevated angle, resting on a light-colored wooden surface.
    
    Here's a detailed breakdown of the image:
    
    Chess Board:
    - It is a standard 8x8 grid with alternating dark green and off-white (or cream) squares.
    - The board is oriented for White's perspective, with the files 'a' through 'h' labeled along the bottom edge (from left to right) and the ranks '1' through '8' labeled along the left edge (from bottom to top).
    - The "US CHESS FEDERATION" logo is visible at the bottom center of the board.
    - The board appears to be a flexible mat, possibly made of vinyl or similar material.
    
    Chess Pieces: There are five pieces on the board in total: two white pieces and three black pieces.
    - White Pieces:
      - A white queen is positioned on square e2 (a light-colored square).
      - A white pawn is positioned on square f6 (a dark green square).
    - Black Pieces:
      - A black pawn is positioned on square a3 (a dark green square).
      - A black pawn is positioned on square c6 (a dark green square).
      - A black pawn is positioned on square e4 (a light-colored square).
    - All pieces are standard tournament-style, likely made of plastic, with the white pieces being off-white and the black pieces being solid black.
    
    Background: The chess board is placed on a wooden surface, which has a light brown hue and visible wood grain.
    
    Lighting and Clarity: The image is well-lit, providing clear visibility of the board and all the pieces.
    """
    
    print("🎯 Testing Cursor Description-Based Chess Position API")
    print("=" * 60)
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image file {IMAGE_PATH} not found")
        return
    
    try:
        # Prepare the request
        with open(IMAGE_PATH, "rb") as f:
            files = {"image": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
            data = {
                "cursor_description": CURSOR_DESCRIPTION,
                "color": "white",
                "debug_image_width": 800,
                "debug_image_height": 600
            }
            
            print(f"📤 Uploading image: {IMAGE_PATH}")
            print(f"📝 Cursor description length: {len(CURSOR_DESCRIPTION)} characters")
            print(f"🌐 API URL: {API_URL}")
            print("-" * 60)
            
            # Make the request
            response = requests.post(API_URL, files=files, data=data)
            
            print(f"📥 Response Status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                print("\n✅ SUCCESS! Here's what the API returns:")
                print("=" * 60)
                
                # Pretty print the JSON response
                print(json.dumps(result, indent=2))
                
                # Extract key information
                print("\n🎯 Key Information:")
                print("-" * 30)
                print(f"📝 Position Description: {result.get('position_description', 'N/A')}")
                print(f"♟️ FEN: {result.get('fen', 'N/A')}")
                print(f"✅ Legal Position: {result.get('legal_position', 'N/A')}")
                print(f"🔗 Lichess URL: {result.get('lichess_url', 'N/A')}")
                print(f"⏱️ Processing Time: {result.get('processing_time', 'N/A')}")
                print(f"🔧 Method Used: {result.get('method', 'N/A')}")
                
                # Show ASCII board
                if 'ascii' in result:
                    print(f"\n📋 ASCII Board:\n{result['ascii']}")
                
                # Show debug images info
                if 'debug_images' in result:
                    print(f"\n🖼️ Debug Images: {len(result['debug_images'])} images generated")
                    for key in result['debug_images'].keys():
                        print(f"   - {key}")
                
                # Show cursor description info
                if 'cursor_description' in result:
                    print(f"\n📝 Cursor Description (truncated):")
                    cursor_desc = result['cursor_description']
                    if len(cursor_desc) > 200:
                        print(f"   {cursor_desc[:200]}...")
                    else:
                        print(f"   {cursor_desc}")
                
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the server is running on port 8001")
        print("💡 Start the server with: python main.py")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_parser_function():
    """Test the parser function directly with different descriptions."""
    
    print("\n" + "=" * 60)
    print("Testing Parser Function Directly")
    print("=" * 60)
    
    try:
        from main import parse_cursor_description_to_board
        
        # Test cases
        test_descriptions = [
            "White Queen on e2, Black Pawn on e4",
            "A white queen is positioned on square e2. A black pawn is positioned on square e4.",
            "There is a white queen on e2 and a black pawn on e4",
            "The board shows a white queen at e2 and black pawn at e4",
            "White pieces: White Queen on e2; White Pawn on f6. Black pieces: Black Pawns on c6, e4, a3.",
        ]
        
        for i, description in enumerate(test_descriptions, 1):
            print(f"\n🧪 Test Case {i}:")
            print(f"Description: {description}")
            
            board = parse_cursor_description_to_board(description)
            
            print(f"FEN: {board.fen()}")
            print(f"ASCII Board:")
            print(str(board))
            
            # Count pieces
            piece_count = len([c for c in board.fen().split()[0] if c.isalpha()])
            print(f"Total pieces detected: {piece_count}")
            print("-" * 40)
            
    except Exception as e:
        print(f"❌ Parser test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Test the API endpoint
    test_cursor_description_api()
    
    # Test the parser function
    test_parser_function() 