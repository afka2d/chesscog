import requests
import json
import os

def test_new_description_endpoint():
  the new recognize_chess_position_with_description endpoint."""
    # API endpoint (note: server runs on port801   API_URL = http://localhost:801ize_chess_position_with_description"
    
    # Test image path - using one of the available images
    IMAGE_PATH = "IMG_4540eg    
    print("🎯 Testing New Chess Position Description Endpoint)
    print(= * 60)
    
    # Check if image exists
    if not os.path.exists(IMAGE_PATH):
        print(f"❌ Error: Image file {IMAGE_PATH} not found")
        return
    
    try:
        # Prepare the request
        with open(IMAGE_PATH, "rb") as f:
            files = {"image": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
            data =[object Object]
                color,
                debug_image_width:400   }
            
            print(f"📤 Uploading image: {IMAGE_PATH}")
            print(f"🌐 API URL: {API_URL}")
            print(f"⚙️ Parameters: {data}")
            print("-" *60      
            # Make the request
            response = requests.post(API_URL, files=files, data=data)
            
            print(f"📥 Response Status: {response.status_code}")
            
            if response.status_code == 200            result = response.json()
                
                print(undefinedn✅ SUCCESS! Here's what the API returns:)             print("=" * 60)
                
                # Pretty print the JSON response
                print(json.dumps(result, indent=2))
                
                # Extract key information
                print("\n🎯 Key Information:)             print("-" * 30             print(f"📝 Position Description: {result.get('position_description', 'N/A')})             print(f"♟️ FEN: {result.get('fen', 'N/A')})             print(f✅ Legal Position: {result.get('legal_position', 'N/A')})             print(f🔗 Lichess URL: {result.get('lichess_url', 'N/A')})             print(f"⏱️ Processing Time: {result.get(processing_time', 'N/A')}")
                
                # Show ASCII board
                ifascii' in result:
                    print(f"\n📋 ASCII Board:\n{result['ascii']}")
                
                # Show debug images info
                ifdebug_images' in result:
                    print(f"\n🖼️ Debug Images: {len(result['debug_images'])} images generated")
                    for i, img_name in enumerate(result.get(debug_image_paths', [])):
                        print(f"   {i+1}. {img_name}")
                
            else:
                print(f"❌ Error: {response.status_code})             print(f"Response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the server is running on port 801)
        print("💡 Start the server with: python main.py")
    except Exception as e:
        print(f"❌ Error: {e})if __name__ == "__main__":
    test_new_description_endpoint() 