import requests
import json
import os

def test_new_endpoint():
  the new recognize_chess_position_with_description endpoint."   API_URL = http://localhost:801ize_chess_position_with_description    IMAGE_PATH = "IMG_4540eg    print(Testingnew chess position description endpoint)
    print(= * 50    
    if not os.path.exists(IMAGE_PATH):
        print(fError: Image file {IMAGE_PATH} not found")
        return
    
    try:
        with open(IMAGE_PATH, "rb") as f:
            files = {"image": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
            data =[object Object]
                color,
                debug_image_width:400   }
            
            print(f"Uploading image: {IMAGE_PATH}")
            response = requests.post(API_URL, files=files, data=data)
            
            print(f"Response Status: {response.status_code}")
            
            if response.status_code == 200            result = response.json()
                print(nSUCCESS! Here's what the API returns:)             print("=" * 50             print(json.dumps(result, indent=2))
            else:
                print(f"Error: {response.status_code})             print(f"Response: {response.text}")
                
    except requests.exceptions.ConnectionError:
        print(Connection Error: Make sure the server is running on port8001  except Exception as e:
        print(f"Error: {e})if __name__ == "__main__":
    test_new_endpoint() 