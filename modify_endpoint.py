#!/usr/bin/env python3
"""
Script to modify the existing endpoint to return the correct format.
"""

import re

# Read the current main.py
with open('/root/chesscog/main.py', 'r') as f:
    content = f.read()

# Find the existing endpoint and modify its return statement
# Look for the return JSONResponse in the recognize_chess_position_with_corners endpoint
# We need to find the specific return statement that has all the debug info

# Pattern to match the return JSONResponse with all the debug fields
return_pattern = r'(\s+return JSONResponse\(\s*content=\{\s*"fen": fen,\s*"ascii": ascii_board,\s*"lichess_url": lichess_url,\s*"legal_position": legal,\s*"position_description": position_description,\s*"debug_images": debug_images_base64,\s*"debug_image_paths": debug_image_paths,\s*"corners": corners\.tolist\(\) if corners is not None else None,\s*"processing_time": time\.time\(\),\s*"image_info": \{\s*"filename": image\.filename,\s*"content_type": image\.content_type,\s*"size_bytes": len\(img_bytes\),\s*"shape": img\.shape\s*\},\s*"debug_info": \{\s*"corner_detection": "Completed",\s*"board_warping": "Completed",\s*"position_detection": "Completed",\s*"visualization": "Completed",\s*"description_generation": "Completed"\s*\}\s*\}\s*\))'

# New return statement with the correct format
new_return = '''        # Convert pieces to the format expected by the app
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
        
        return {
            "fen": fen,
            "pieces": pieces_list,
            "occupancy": occupancy_list,
            "success": True
        }'''

# Replace the return statement
new_content = re.sub(return_pattern, new_return, content, flags=re.DOTALL)

# Write the modified content back
with open('/root/chesscog/main.py', 'w') as f:
    f.write(new_content)

print("Modified existing endpoint successfully")
