"""Script to create the piece classification dataset.

Based on the four chessboard corner points which are supplied as labels, this module is responsible for warping the image and cutting out the squares.
Note that before running this module requires the rendered dataset to be downloaded and split (see the :mod:`chesscog.data_synthesis` module for more information).

Note that script is different to :mod:`chesscog.core.piece_classifier.create_dataset` because the squares are cropped differently.

.. code-block:: console

    $ python -m chesscog.piece_classifier.create_dataet --help p
    usage: create_dataset.py [-h]
    
    Create the dataset for piece classification.
    
    optional arguments:
      -h, --help  show this help message and exit
"""

from pathlib import Path
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageDraw
import json
import numpy as np
import chess
import os
import shutil
from recap import URI
import argparse

from chesscog.core import sort_corner_points
from chesscog.core.dataset import piece_name

RENDERS_DIR = URI("data://render")
OUT_DIR = URI("data://pieces")
SQUARE_SIZE = 50
BOARD_SIZE = 8 * SQUARE_SIZE
IMG_SIZE = BOARD_SIZE * 2
MARGIN = (IMG_SIZE - BOARD_SIZE) / 2
MIN_HEIGHT_INCREASE, MAX_HEIGHT_INCREASE = 1, 3
MIN_WIDTH_INCREASE, MAX_WIDTH_INCREASE = .25, 1
OUT_WIDTH = int((1 + MAX_WIDTH_INCREASE) * SQUARE_SIZE)
OUT_HEIGHT = int((1 + MAX_HEIGHT_INCREASE) * SQUARE_SIZE)


def crop_square(img: np.ndarray, square: chess.Square, turn: chess.Color) -> np.ndarray:
    """Crop a chess square from the warped input image for piece classification.

    Args:
        img (np.ndarray): the warped input image
        square (chess.Square): the square to crop
        turn (chess.Color): the current player

    Returns:
        np.ndarray: the cropped square
    """

    rank = chess.square_rank(square)
    file = chess.square_file(square)
    if turn == chess.WHITE:
        row, col = 7 - rank, file
    else:
        row, col = rank, 7 - file
    height_increase = MIN_HEIGHT_INCREASE + \
        (MAX_HEIGHT_INCREASE - MIN_HEIGHT_INCREASE) * ((7 - row) / 7)
    left_increase = 0 if col >= 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((3 - col) / 3)
    right_increase = 0 if col < 4 else MIN_WIDTH_INCREASE + \
        (MAX_WIDTH_INCREASE - MIN_WIDTH_INCREASE) * ((col - 4) / 3)
    x1 = int(MARGIN + SQUARE_SIZE * (col - left_increase))
    x2 = int(MARGIN + SQUARE_SIZE * (col + 1 + right_increase))
    y1 = int(MARGIN + SQUARE_SIZE * (row - height_increase))
    y2 = int(MARGIN + SQUARE_SIZE * (row + 1))
    width = x2-x1
    height = y2-y1
    cropped_piece = img[y1:y2, x1:x2]
    if col < 4:
        cropped_piece = cv2.flip(cropped_piece, 1)
    result = np.zeros((OUT_HEIGHT, OUT_WIDTH, 3), dtype=cropped_piece.dtype)
    result[OUT_HEIGHT - height:, :width] = cropped_piece
    return result


def warp_chessboard_image(img: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Warp the image of the chessboard onto a regular grid.

    Note: this method is different to :meth:`chesscog.core.piece_classifier.create_dataset.warp_chessboard_image` because the the warped image will be bigger.

    Args:
        img (np.ndarray): the image of the chessboard
        corners (np.ndarray): pixel locations of the four corner points

    Returns:
        np.ndarray: the warped image
    """

    src_points = sort_corner_points(corners)
    dst_points = np.array([[MARGIN, MARGIN],  # top left
                           [BOARD_SIZE + MARGIN, MARGIN],  # top right
                           [BOARD_SIZE + MARGIN, \
                            BOARD_SIZE + MARGIN],  # bottom right
                           [MARGIN, BOARD_SIZE + MARGIN]  # bottom left
                           ], dtype=np.float32)
    transformation_matrix, mask = cv2.findHomography(src_points, dst_points)
    return cv2.warpPerspective(img, transformation_matrix, (IMG_SIZE, IMG_SIZE))


def _extract_squares_from_sample(id: str, subset: str = "", input_dir: Path = RENDERS_DIR, output_dir: Path = OUT_DIR):
    img = cv2.imread(str(input_dir / subset / (id + ".png")))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with (input_dir / subset / (id + ".json")).open("r") as f:
        label = json.load(f)

    corners = np.array(label["corners"], dtype=np.float32)
    unwarped = warp_chessboard_image(img, corners)

    board = chess.Board(label["fen"])

    for square, piece in board.piece_map().items():
        piece_img = crop_square(unwarped, square, label["white_turn"])
        with Image.fromarray(piece_img, "RGB") as piece_img:
            piece_img.save(output_dir / subset / piece_name(piece) /
                           f"{id}_{chess.square_name(square)}.png")


def _create_folders(subset: str, output_dir: Path):
    for piece_type in chess.PIECE_TYPES:
        for color in chess.COLORS:
            piece = chess.Piece(piece_type, color)
            folder = output_dir / subset / piece_name(piece)
            folder.mkdir(parents=True, exist_ok=True)


def create_dataset(input_dir: Path = RENDERS_DIR, output_dir: Path = OUT_DIR):
    """Create the piece classification dataset.

    Args:
        input_dir (Path, optional): the input folder of the rendered images. Defaults to ``data://render``.
        output_dir (Path, optional): the output folder. Defaults to ``data://pieces``.
    """

    for subset in ("train", "val", "test"):
        _create_folders(subset, output_dir)
        # Support .png, .jpg, .jpeg
        samples = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.JPG", "*.JPEG"):
            samples.extend((input_dir / subset).glob(ext))
        for img_path in samples:
            print(f"\n[DEBUG] Processing image: {img_path}")
            ann_path = input_dir.parent / "annotations" / subset / (img_path.stem + ".json")
            if not ann_path.exists():
                print(f"[DEBUG] Annotation file not found: {ann_path}")
                continue
            with open(ann_path, "r") as f:
                ann = json.load(f)
            fen = ann.get("fen", None)
            corners = ann.get("corners", None)
            if not fen:
                print(f"[DEBUG] No FEN in annotation: {ann_path}")
                continue
            if not corners or len(corners) != 4:
                print(f"[DEBUG] Invalid corners in annotation: {corners}")
                continue
            print(f"[DEBUG] FEN: {fen}")
            print(f"[DEBUG] Corners: {corners}")

            # Load the image
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"[DEBUG] Could not load image: {img_path}")
                continue
            # Compute perspective transform to get unwarped board
            try:
                src = np.array(corners, dtype=np.float32)
                board_size = 800  # px
                dst = np.array([[0,0],[board_size-1,0],[board_size-1,board_size-1],[0,board_size-1]], dtype=np.float32)
                M = cv2.getPerspectiveTransform(src, dst)
                unwarped = cv2.warpPerspective(img, M, (board_size, board_size))
            except Exception as e:
                print(f"[DEBUG] Failed to warp image: {e}")
                continue

            board = chess.Board(fen)
            for square, piece in board.piece_map().items():
                try:
                    piece_img = crop_square(unwarped, square, ann["white_turn"])
                    with Image.fromarray(piece_img, "RGB") as piece_img:
                        piece_img.save(output_dir / subset / piece_name(piece) /
                                       f"{img_path.stem}_{chess.square_name(square)}.png")
                    print(f"[DEBUG] Extracted piece: {piece_name(piece)}, {chess.square_name(square)}")
                except Exception as e:
                    print(f"[DEBUG] Skipped piece: {chess.square_name(square)} due to error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create the dataset for piece classification.")
    parser.add_argument("input_dir", nargs="?", default=RENDERS_DIR, help="Input directory (default: data://render)")
    parser.add_argument("output_dir", nargs="?", default=OUT_DIR, help="Output directory (default: data://pieces)")
    args = parser.parse_args()
    create_dataset(Path(args.input_dir), Path(args.output_dir))
