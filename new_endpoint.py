Add this endpoint to the existing main.py file on the server:

@app.post("/recognize_chess_position_with_corners_app")
async def recognize_chess_position_with_corners_app(
    image: UploadFile = File(...),
    corners: str = Form(...),  # JSON string of corner coordinates
    turn: str = "white"
):
    """
    Recognize chess position from uploaded image using manually corrected corner coordinates.
    Returns the response format expected by the mobile app.
    """
    try:
        # Validate input
        if not image.filename:
            raise HTTPException(status_code=400, detail="No image file provided")
        
        # Parse corner coordinates
        try:
            import json
            corner_coords = json.loads(corners)
            if not isinstance(corner_coords, list) or len(corner_coords) != 4:
                raise ValueError("Corners must be a list of 4 coordinate pairs")
            corners_array = np.array(corner_coords, dtype=np.float32)
            if corners_array.shape != (4, 2):
                raise ValueError("Each corner must have 2 coordinates (x, y)")
        except (json.JSONDecodeError, ValueError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid corner coordinates: {str(e)}")

        # Read and validate image
        img_bytes = await image.read()
        if not img_bytes:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Decode image
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(status_code=400, detail="Failed to decode image")
        
        # Validate turn parameter
        if turn.lower() not in ["white", "black"]:
            raise HTTPException(status_code=400, detail="Turn must be 'white' or 'black'")

        turn_color = chess.WHITE if turn.lower() == "white" else chess.BLACK
        
        logger.info(f"Processing image with manual corners: {image.filename}")
        logger.info(f"Corner coordinates: {corner_coords}")
        
        # Use the recognizer to classify occupancy and pieces
        try:
            # Classify occupancy
            logger.info("Classifying occupancy...")
            occupancy = recognizer._classify_occupancy(img, turn_color, corners_array)
            
            # Classify pieces
            logger.info("Classifying pieces...")
            pieces = recognizer._classify_pieces(img, turn_color, corners_array, occupancy)
            
            # Create the chess board
            logger.info("Creating chess board...")
            board = chess.Board()
            board.clear()
            
            # Place pieces on the board
            for square, piece in zip(recognizer._squares, pieces):
                if piece is not None:
                    board.set_piece_at(square, piece)
            
            # Set the turn
            board.turn = turn_color
            
            # Generate results
            fen = board.fen()
            
            # Convert pieces to the format expected by the app
            pieces_list = []
            for square, piece in zip(recognizer._squares, pieces):
                if piece is not None:
                    # Convert piece to string representation
                    piece_str = piece.symbol()
                    pieces_list.append(piece_str)
                else:
                    pieces_list.append(None)
            
            # Convert occupancy to list format
            occupancy_list = occupancy.tolist() if hasattr(occupancy, 'tolist') else list(occupancy)
            
            logger.info(f"Recognition successful: FEN={fen}")
            logger.info(f"Pieces found: {sum(1 for p in pieces_list if p is not None)}")
            
            return {
                "fen": fen,
                "pieces": pieces_list,
                "occupancy": occupancy_list,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Recognition failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Recognition failed: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

