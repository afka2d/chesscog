from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path

import numpy as np
import cv2

from chesscog.recognition.recognition import ChessRecognizer

app = FastAPI()

recognizer = ChessRecognizer(Path("models"))  # models folder path as Path object

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
        board, *_ = recognizer.predict(img, color)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")

    fen = board.fen()
    ascii_board = str(board)
    lichess_url = f"https://lichess.org/editor/{fen.replace(' ', '/')}"

    legal = board.is_valid()

    return JSONResponse(
        content={
            "fen": fen,
            "ascii": ascii_board,
            "lichess_url": lichess_url,
            "legal_position": legal,
        }
    )
