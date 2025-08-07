#!/usr/bin/env python3

import requests
import json
import base64
from PIL import Image
import io

def submit_image_to_api(image_path, api_url="http://localhost:8001/recognize_chess_position_simple"):
    """
    Submit an image to the simplified chess position recognition API
    """
    try:
        # Prepare the multipart form data - only image, no description needed
        files = {'image': (image_path, open(image_path, 'rb'), 'image/jpeg')}
        
        print(f"Submitting image '{image_path}' to {api_url}...")
        response = requests.post(api_url, files=files)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        
        result = response.json()
        print("API Response:")
        print(json.dumps(result, indent=2))
        
        # Print a nice summary
        print(f"\n📊 Summary:")
        print(f"   FEN: {result['fen']}")
        print(f"   Pieces found: {result['pieces_found']}")
        print(f"   Legal position: {result['legal_position']}")
        print(f"   2D Board:")
        for i, row in enumerate(result['board_2d']):
            print(f"   {8-i}: {' '.join(row)}")
        print(f"     a b c d e f g h")
        
    except requests.exceptions.RequestException as e:
        print(f"Error submitting image: {e}")
    except json.JSONDecodeError:
        print(f"Error decoding JSON response: {response.text}")

if __name__ == "__main__":
    # Use the correct filename for the attached image
    image_to_submit = "IMG_4698.JPG"
    submit_image_to_api(image_to_submit) 