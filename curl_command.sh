#!/bin/bash

# Curl command to test the chess position recognition API
# Make sure the server is running on localhost:8001 before executing this

curl -X POST "http://localhost:8001/recognize_chess_position_with_cursor_description" \
  -F "image=@sample.jpeg" \
  -F 'cursor_description=This image displays a chess board with two pieces on it, viewed from a slightly elevated angle.

**High-Level Description:**
The image shows a standard 8x8 chess board with alternating dark green and off-white squares. The board is oriented with algebraic notation visible along its edges. There are two pawns on the board: one black and one white.

**Detailed Description:**
*   **Chess Board:**
    *   The board is a standard 8x8 grid, featuring dark green and off-white (or cream) squares.
    *   It is oriented with algebraic notation: files "a" through "h" are labeled along the left and right edges, and ranks "1" through "8" are labeled along the top and bottom edges.
    *   The "US CHESS FEDERATION" logo is visible on the "c" file, between ranks 4 and 5, on the left side of the board.
    *   The board appears to be a flexible mat, possibly made of vinyl, and shows some minor surface imperfections or creases.
*   **Chess Pieces:**
    *   There are two pieces on the board, both pawns.
    *   A **black pawn** is positioned on square **d4**. This square is a dark green square.
    *   A **white pawn** is positioned on square **e5**. This square is an off-white square.
    *   Both pawns appear to be standard Staunton-style pieces.
*   **Overall Scene:**
    *   The board is placed on a light-colored surface, possibly wood, which is visible around the edges of the board.
    *   The background beyond the board is a solid dark gray, suggesting the image might be cropped or the board is on a dark, unlit surface.' \
  -H "Content-Type: multipart/form-data" \
  | python3 -m json.tool 