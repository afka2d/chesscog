"""Module that brings together the whole recognition pipeline into a single class so it can be conveniently executed.

This module simultaneously acts as a script to perform a single inference:

.. code-block:: console

    $ python -m chesscog.recognition.recognition --help
    usage: recognition.py [-h] [--white] [--black] file
    
    Run the chess recognition pipeline on an input image
    
    positional arguments:
      file        path to the input image
    
    optional arguments:
      -h, --help  show this help message and exit
      --white     indicate that the image is from the white player's
                  perspective (default)
      --black     indicate that the image is from the black player's
                  perspective
"""

import numpy as np
import chess
from chess import Status
from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import functools
import cv2
import argparse
import typing
from recap import URI, CfgNode as CN

from chesscog.corner_detection import find_corners, resize_image
from chesscog.occupancy_classifier import create_dataset as create_occupancy_dataset
from chesscog.piece_classifier import create_dataset as create_piece_dataset
from chesscog.core import device, DEVICE
from chesscog.core.dataset import build_transforms, Datasets
from chesscog.core.dataset import name_to_piece


class ChessRecognizer:
    """A class implementing the entire chess inference pipeline.

    Once you create an instance of this class, the CNNs are loaded into memory (possibly the GPU if available), so if you want to perform multiple inferences, they should all use one instance of this class for performance purposes.
    """

    _squares = list(chess.SQUARES)

    def __init__(self, classifiers_folder: Path = URI("models://")):
        """Constructor.

        Args:
            classifiers_folder (Path, optional): the path to the classifiers (supplying a different path is especially useful because the transfer learning classifiers are located at ``models://transfer_learning``). Defaults to ``models://``.
        """
        self._corner_detection_cfg = CN.load_yaml_with_base(
            "config://corner_detection.yaml")

        self._occupancy_cfg, self._occupancy_model = self._load_classifier(
            classifiers_folder / "occupancy_classifier")
        self._occupancy_transforms = build_transforms(
            self._occupancy_cfg, mode=Datasets.TEST)
        self._pieces_cfg, self._pieces_model = self._load_classifier(
            classifiers_folder / "piece_classifier")
        self._pieces_transforms = build_transforms(
            self._pieces_cfg, mode=Datasets.TEST)
        self._piece_classes = np.array(list(map(name_to_piece,
                                                self._pieces_cfg.DATASET.CLASSES)))

    @classmethod
    def _load_classifier(cls, path: Path):
        model_file = next(iter(path.glob("*.pt")))
        yaml_file = next(iter(path.glob("*.yaml")))
        cfg = CN.load_yaml_with_base(yaml_file)
        model = torch.load(model_file, map_location=DEVICE, weights_only=False)
        model = device(model)
        model.eval()
        return cfg, model

    def _classify_occupancy(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray) -> np.ndarray:
        warped = create_occupancy_dataset.warp_chessboard_image(
            img, corners)
        square_imgs = map(functools.partial(
            create_occupancy_dataset.crop_square, warped, turn=turn), self._squares)
        square_imgs = map(Image.fromarray, square_imgs)
        square_imgs = map(self._occupancy_transforms, square_imgs)
        square_imgs = list(square_imgs)
        square_imgs = torch.stack(square_imgs)
        square_imgs = device(square_imgs)
        occupancy = self._occupancy_model(square_imgs)
        occupancy = occupancy.argmax(
            axis=-1) == self._occupancy_cfg.DATASET.CLASSES.index("occupied")
        occupancy = occupancy.cpu().numpy()
        return occupancy

    def _classify_pieces(self, img: np.ndarray, turn: chess.Color, corners: np.ndarray, occupancy: np.ndarray) -> np.ndarray:
        occupied_squares = np.array(self._squares)[occupancy]
        
        # If no squares are occupied, return all None pieces
        if len(occupied_squares) == 0:
            all_pieces = np.full(len(self._squares), None, dtype=object)
            return all_pieces
        
        warped = create_piece_dataset.warp_chessboard_image(
            img, corners)
        piece_imgs = map(functools.partial(
            create_piece_dataset.crop_square, warped, turn=turn), occupied_squares)
        piece_imgs = map(Image.fromarray, piece_imgs)
        piece_imgs = map(self._pieces_transforms, piece_imgs)
        piece_imgs = list(piece_imgs)
        piece_imgs = torch.stack(piece_imgs)
        piece_imgs = device(piece_imgs)
        pieces = self._pieces_model(piece_imgs)
        pieces = pieces.argmax(axis=-1).cpu().numpy()
        pieces = self._piece_classes[pieces]
        all_pieces = np.full(len(self._squares), None, dtype=object)
        all_pieces[occupancy] = pieces
        return all_pieces

    def predict(self, img: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray]:
        """Perform an inference.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray]: the predicted position on the board and the four corner points
        """
        with torch.no_grad():
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners, _ = find_corners(self._corner_detection_cfg, img)  # Ignore debug images
            occupancy = self._classify_occupancy(img, turn, corners)
            pieces = self._classify_pieces(img, turn, corners, occupancy)

            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            corners = corners / img_scale
            return board, corners

    def predict_with_debug(self, img: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray, dict]:
        """Perform an inference and return intermediate debug images.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray, dict]: the predicted position, corners, and debug images dict
        """
        with torch.no_grad():
            debug_images = {}
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners, corner_debug = find_corners(self._corner_detection_cfg, img)
            debug_images.update(corner_debug)
            # Warped board image
            warped_board = create_occupancy_dataset.warp_chessboard_image(img, corners)
            debug_images['warped_board'] = warped_board.copy()
            # Occupancy classification
            occupancy = self._classify_occupancy(img, turn, corners)
            # Visualize occupancy map
            debug_images['occupancy_map'] = self._visualize_occupancy_map(warped_board, occupancy, turn)
            # Piece classification
            pieces = self._classify_pieces(img, turn, corners, occupancy)
            # Visualize piece map
            debug_images['piece_map'] = self._visualize_piece_map(warped_board, pieces, occupancy, turn)
            # Build board
            board = chess.Board()
            board.clear_board()
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            corners = corners / img_scale
            return board, corners, debug_images

    def _visualize_occupancy_map(self, warped_board: np.ndarray, occupancy: np.ndarray, turn: chess.Color) -> np.ndarray:
        # Draw green (occupied) and red (empty) overlays on the warped board
        vis = warped_board.copy()
        square_size = vis.shape[0] // 10  # SQUARE_SIZE is 50, board is 8x8, with margin
        for idx, occ in enumerate(occupancy):
            rank = chess.square_rank(self._squares[idx])
            file = chess.square_file(self._squares[idx])
            if turn == chess.WHITE:
                row, col = 7 - rank, file
            else:
                row, col = rank, 7 - file
            x1 = int((col + 1) * square_size)
            y1 = int((row + 1) * square_size)
            x2 = int((col + 2) * square_size)
            y2 = int((row + 2) * square_size)
            color = (0, 255, 0) if occ else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        return vis

    def _visualize_piece_map(self, warped_board: np.ndarray, pieces: np.ndarray, occupancy: np.ndarray, turn: chess.Color) -> np.ndarray:
        # Overlay piece symbols on the warped board
        vis = warped_board.copy()
        square_size = vis.shape[0] // 10
        piece_symbols = {
            'P': '♙', 'R': '♖', 'N': '♘', 'B': '♗', 'Q': '♕', 'K': '♔',
            'p': '♟', 'r': '♜', 'n': '♞', 'b': '♝', 'q': '♛', 'k': '♚'
        }
        for idx, piece in enumerate(pieces):
            if piece is not None:
                rank = chess.square_rank(self._squares[idx])
                file = chess.square_file(self._squares[idx])
                if turn == chess.WHITE:
                    row, col = 7 - rank, file
                else:
                    row, col = rank, 7 - file
                x = int((col + 1.5) * square_size)
                y = int((row + 1.2) * square_size)
                symbol = piece_symbols.get(piece.symbol(), piece.symbol())
                pil_img = Image.fromarray(vis)
                draw = ImageDraw.Draw(pil_img)
                try:
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", int(square_size * 0.8))
                except:
                    font = ImageFont.load_default()
                color = (0, 0, 0) if piece.color == chess.WHITE else (255, 255, 255)
                draw.text((x, y), symbol, fill=color, font=font)
                vis = np.array(pil_img)
        return vis


class TimedChessRecognizer(ChessRecognizer):
    """A subclass of :class:`ChessRecognizer` that additionally records the time taken for each step of the pipeline during inference.
    """

    def predict(self, img: np.ndarray, turn: chess.Color = chess.WHITE) -> typing.Tuple[chess.Board, np.ndarray, dict]:
        """Perform an inference.

        Args:
            img (np.ndarray): the input image (RGB)
            turn (chess.Color, optional): the current player. Defaults to chess.WHITE.

        Returns:
            typing.Tuple[chess.Board, np.ndarray, dict]: the predicted position on the board, the four corner points, and a dict containing the time taken for each stage of the inference pipeline
        """

        from timeit import default_timer as timer
        with torch.no_grad():
            t1 = timer()
            img, img_scale = resize_image(self._corner_detection_cfg, img)
            corners = find_corners(self._corner_detection_cfg, img)
            t2 = timer()
            occupancy = self._classify_occupancy(img, turn, corners)
            t3 = timer()
            pieces = self._classify_pieces(img, turn, corners, occupancy)
            t4 = timer()

            board = chess.Board()
            board.clear()
            board.turn = turn
            for square, piece in zip(self._squares, pieces):
                if piece:
                    board.set_piece_at(square, piece)
            corners = corners / img_scale
            t5 = timer()

            times = {
                "corner_detection": t2-t1,
                "occupancy_classification": t3-t2,
                "piece_classification": t4-t3,
                "prepare_results": t5-t4
            }

            return board, corners, times


def main(classifiers_folder: Path = URI("models://"), setup: callable = lambda: None):
    """Main method for running inference from the command line.

    Args:
        classifiers_folder (Path, optional): the path to the classifiers (supplying a different path is especially useful because the transfer learning classifiers are located at ``models://transfer_learning``). Defaults to ``models://``.
        setup (callable, optional): An optional setup function to be called after the CLI argument parser has been setup. Defaults to lambda:None.
    """

    parser = argparse.ArgumentParser(
        description="Run the chess recognition pipeline on an input image")
    parser.add_argument("file", help="path to the input image", type=str)
    parser.add_argument(
        "--white", help="indicate that the image is from the white player's perspective (default)", action="store_true", dest="color")
    parser.add_argument(
        "--black", help="indicate that the image is from the black player's perspective", action="store_false", dest="color")
    parser.set_defaults(color=True)
    args = parser.parse_args()

    setup()

    img = cv2.imread(str(URI(args.file)))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    recognizer = ChessRecognizer(classifiers_folder)
    board, *_ = recognizer.predict(img, args.color)

    print(board)
    print()
    print(
        f"You can view this position at https://lichess.org/editor/{board.board_fen()}")

    if board.status() != Status.VALID:
        print()
        print("WARNING: The predicted chess position is not legal according to the rules of chess.")
        print("         You might want to try again with another picture.")


if __name__ == "__main__":
    from chesscog.occupancy_classifier.download_model import ensure_model as ensure_occupancy_classifier
    from chesscog.piece_classifier.download_model import ensure_model as ensure_piece_classifier

    main(setup=lambda: [ensure_model(show_size=True)
                        for ensure_model in (ensure_occupancy_classifier, ensure_piece_classifier)])
