#!/usr/bin/env python3
"""
Test YOLO corner detection visualization
"""
import requests
import base64
import os

def test_yolo_visualization():
    api_url = "http://localhost:8002"
    test_image_path = "my_chess_images/train/images/IMG_4698.JPG"
    
    print("🎨 Testing YOLO Corner Visualization")
    print("=" * 50)
    
    try:
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{api_url}/visualize_corners", files=files)
        
        print(f"Visualization response: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Success!")
            print(f"Corners: {data['corners']}")
            print(f"Processing time: {data['processing_time']}s")
            print(f"Model: {data['model']}")
            
            # Save the visualization image
            if 'image' in data:
                img_data = base64.b64decode(data['image'])
                output_path = "yolo_corner_visualization_test.jpg"
                with open(output_path, 'wb') as f:
                    f.write(img_data)
                print(f"📸 Visualization saved to: {output_path}")
                
        else:
            print(f"❌ Error: {response.status_code}")
            try:
                error_data = response.json()
                print(f"Detail: {error_data.get('detail', 'Unknown error')}")
            except:
                print(f"Raw response: {response.text}")
                
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_yolo_visualization()
