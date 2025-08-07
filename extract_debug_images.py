import requests
import base64
import os

API_URL = "http://localhost:8000/recognize_chess_position"
IMAGE_PATH = "IMG_4540.jpeg"  # Change this to your test image path

def main():
    with open(IMAGE_PATH, "rb") as f:
        files = {"image": (os.path.basename(IMAGE_PATH), f, "image/jpeg")}
        response = requests.post(API_URL, files=files)
    
    if response.status_code != 200:
        print("Error:", response.text)
        return

    data = response.json()
    debug_images = data.get("debug_images", {})
    if not debug_images:
        print("No debug images found in response.")
        return

    os.makedirs("debug_outputs", exist_ok=True)
    for name, b64img in debug_images.items():
        img_bytes = base64.b64decode(b64img)
        out_path = os.path.join("debug_outputs", f"{name}.png")
        with open(out_path, "wb") as out_file:
            out_file.write(img_bytes)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main() 