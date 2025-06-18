from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
import base64
import io

import numpy as np
import cv2
import time
import logging

from chesscog.recognition.recognition import ChessRecognizer
from chesscog.corner_detection import find_corners
from chesscog.corner_detection.detect_corners import CN

app = FastAPI()

# Load the default configuration at startup
cfg = CN.load_yaml_with_base("config/corner_detection.yaml")
recognizer = ChessRecognizer(Path("models"))  # models folder path as Path object

def encode_image(img):
    """Convert a numpy array image to base64 string."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

@app.post("/recognize_chess_position")
async def recognize_chess_position(image: UploadFile = File(...), color: str = "white"):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported image type. Use JPEG or PNG.")

    img_bytes = await image.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    try:
        # First get debug images
        _, debug_images = find_corners(cfg, img)
        
        # Convert debug images to base64
        debug_images_base64 = {
            key: encode_image(img) for key, img in debug_images.items()
        }
        
        # Then do the actual recognition
        board = recognizer.predict(img, color)[0]  # Only get the board, ignore other return values
        
        fen = board.fen()
        ascii_board = str(board)
        lichess_url = f"https://lichess.org/editor/{fen}?color={color}"
        legal = board.is_valid()

        return JSONResponse(
            content={
                "fen": fen,
                "ascii": ascii_board,
                "lichess_url": lichess_url,
                "legal_position": legal,
                "debug_images": debug_images_base64
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

@app.post("/detect_corners")
async def detect_corners(image: UploadFile = File(...)):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Unsupported image type. Use JPEG or PNG.")

    img_bytes = await image.read()
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode image")

    try:
        corners, debug_images = find_corners(cfg, img)
        
        # Convert debug images to base64
        debug_images_base64 = {
            key: encode_image(img) for key, img in debug_images.items()
        }
        
        # Convert corners to list format for JSON serialization
        corners_list = corners.tolist()
        
        return JSONResponse(
            content={
                "corners": corners_list,
                "message": "Successfully detected chessboard corners",
                "debug_images": debug_images_base64
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Corner detection failed: {str(e)}")
