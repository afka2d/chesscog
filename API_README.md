# Chess Position Scanner API

A FastAPI-based REST API for recognizing chess positions from images, with comprehensive debug/preprocessed images and human-readable position descriptions for iOS app integration.

## Features

- **Chess Position Recognition**: Convert chess board images to FEN notation
- **Human-Readable Descriptions**: Get natural language descriptions of piece positions (similar to Cursor's image analysis)
- **Corner Detection**: Detect chess board corners with debug visualization
- **Comprehensive Debug Images**: Get preprocessed images showing every step of the recognition process
- **Visual Position Representation**: See the detected chess position as a visual board
- **Health Monitoring**: Built-in health checks and monitoring
- **CORS Support**: Ready for iOS app integration
- **Docker Deployment**: Easy deployment with Docker and Docker Compose

## API Endpoints

### Health Check
- `GET /health` - Check API health and model status

### Chess Recognition
- `POST /recognize_chess_position` - Recognize chess position from image
  - Parameters:
    - `image`: Chess board image (JPEG/PNG)
    - `color`: Color to play as ("white" or "black", default: "white")
    - `debug_image_width`: Maximum width for debug images (default: 800)
    - `debug_image_height`: Maximum height for debug images (default: 600)
  - Returns:
    - `fen`: FEN notation of the position
    - `ascii`: ASCII representation of the board
    - `lichess_url`: Lichess editor URL
    - `legal_position`: Whether the position is legal
    - `position_description`: Human-readable description of piece positions
    - `debug_images`: Base64-encoded debug images
    - `corners`: Detected corner coordinates
    - `processing_time`: Processing timestamp
    - `image_info`: Image metadata
    - `debug_info`: Status of each processing step

- `POST /recognize_chess_position_with_description` - Recognize chess position with enhanced description
  - Same parameters as above
  - Enhanced description generation with more detailed piece information

### Corner Detection
- `POST /detect_corners` - Detect chess board corners
  - Parameters:
    - `image`: Chess board image (JPEG/PNG)
  - Returns:
    - `corners`: Detected corner coordinates
    - `message`: Success message
    - `debug_images`: Base64-encoded debug images
    - `processing_time`: Processing timestamp
    - `image_info`: Image metadata

## Position Description Format

The `position_description` field provides human-readable descriptions similar to Cursor's image analysis:

### Example Description:
```
White pieces: White Queen on e2; White Pawn on f6. Black pieces: Black Pawn on a3; Black Pawn on c6; Black Pawn on e4. White to move. Castling available: White kingside, White queenside, Black kingside, Black queenside.
```

### Description Components:
- **Piece locations**: Organized by color and piece type
- **Turn information**: Which player is to move
- **Castling rights**: Available castling options
- **En passant**: If en passant is available
- **Piece counts**: Multiple pieces of the same type are grouped

## Debug Images

The API generates comprehensive debug images showing each step of the recognition process:

- `resized.png` - Input image after resizing
- `edges.png` - Edge detection results
- `lines.png` - Detected lines
- `filtered_lines.png` - Filtered line detection
- `intersections.png` - Line intersections
- `corners.png` - Final detected corners
- `warped_board.png` - Board after perspective correction
- `occupancy_map.png` - Occupancy classification results
- `piece_map.png` - Piece classification results
- `board_focus.png` - Board focus visualization

## Usage Examples

### Python Client Example
```python
import requests

# Upload chess board image
with open("chess_board.jpg", "rb") as f:
    files = {"image": ("chess_board.jpg", f, "image/jpeg")}
    data = {"color": "white"}
    
    response = requests.post(
        "http://localhost:8000/recognize_chess_position_with_description",
        files=files,
        data=data
    )

if response.status_code == 200:
    result = response.json()
    print(f"FEN: {result['fen']}")
    print(f"Description: {result['position_description']}")
    print(f"Legal: {result['legal_position']}")
    print(f"Lichess URL: {result['lichess_url']}")
```

### cURL Example
```bash
curl -X POST "http://localhost:8000/recognize_chess_position_with_description" \
  -F "image=@chess_board.jpg" \
  -F "color=white" \
  -F "debug_image_width=800" \
  -F "debug_image_height=600"
```

## Response Format

```json
{
  "fen": "rnbqkbnr/pppppppp/8/8/4p3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
  "ascii": "r n b q k b n r\np p p p p p p p\n. . . . . . . .\n. . . . . . . .\n. . . . p . . .\n. . . . . . . .\nP P P P P P P P\nR N B Q K B N R",
  "lichess_url": "https://lichess.org/editor/rnbqkbnr/pppppppp/8/8/4p3/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1?color=white",
  "legal_position": true,
  "position_description": "White pieces: White Pawns on a2, b2, c2, d2, e2, f2, g2, h2; White Rooks on a1, h1; White Knights on b1, g1; White Bishops on c1, f1; White Queen on d1; White King on e1. Black pieces: Black Pawns on a7, b7, c7, d7, f7, g7, h7; Black Pawn on e4; Black Rooks on a8, h8; Black Knights on b8, g8; Black Bishops on c8, f8; Black Queen on d8; Black King on e8. White to move. Castling available: White kingside, White queenside, Black kingside, Black queenside.",
  "debug_images": {
    "corners": "base64_encoded_image_data",
    "edges": "base64_encoded_image_data",
    "lines": "base64_encoded_image_data"
  },
  "corners": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
  "processing_time": 1640995200.123456,
  "image_info": {
    "filename": "chess_board.jpg",
    "content_type": "image/jpeg",
    "size_bytes": 123456,
    "shape": [height, width, channels]
  },
  "debug_info": {
    "corner_detection": "Completed",
    "board_warping": "Completed",
    "position_detection": "Completed",
    "visualization": "Completed",
    "description_generation": "Completed"
    }
}
```

## Error Handling

The API returns appropriate HTTP status codes:

- `200` - Success
- `400` - Bad request (invalid image format, missing parameters)
- `503` - Service unavailable (models not loaded)
- `500` - Internal server error

Error responses include detailed error messages:

```json
{
  "detail": "Recognition failed: Chessboard not found in image"
}
```

## Deployment

### Docker Deployment
   ```bash
# Build and run with Docker Compose
docker-compose up --build

# Or build and run manually
docker build -t chesscog-api .
docker run -p 8000:8000 chesscog-api
```

### Local Development
   ```bash
# Install dependencies
   pip install -r requirements.txt

# Run the API
   python main.py
   ```

## Model Requirements

The API requires pre-trained models in the `models/` directory:
- `models/occupancy_classifier/` - Occupancy classification model
- `models/piece_classifier/` - Piece classification model

Models are automatically downloaded on first run if not present.

## Performance

Typical processing times:
- **Corner detection**: 0.5-2 seconds
- **Piece recognition**: 1-3 seconds
- **Total processing**: 2-5 seconds

Performance depends on image size, complexity, and hardware capabilities.

## Integration with iOS Apps

The API is designed for easy integration with iOS applications:

- **CORS enabled** for web-based clients
- **Base64 encoded debug images** for easy display
- **Human-readable descriptions** for user-friendly interfaces
- **Comprehensive error handling** for robust app behavior

## License

This project is licensed under the MIT License - see the LICENSE file for details. 