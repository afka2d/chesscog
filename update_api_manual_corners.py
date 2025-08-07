#!/usr/bin/env python3
"""
Update API to support manual corner coordinates
"""

def update_api_with_manual_corners():
    """Update the API to support manual corner coordinates."""
    
    print("🔧 Updating API to support manual corner coordinates...")
    
    # The changes we need to make:
    # 1. Add optional corner coordinates parameter to the endpoint
    # 2. Use manual corners if provided, otherwise use automatic detection
    # 3. Validate corner coordinates format
    
    patch_content = """
# API Update: Add Manual Corner Support
# 
# Changes needed in /root/chesscog/main.py:
#
# 1. Update the endpoint signature to accept corner coordinates:
#    @app.post("/recognize_chess_position_with_cursor_description")
#    async def recognize_chess_position_with_cursor_description(
#        image: UploadFile = File(...), 
#        color: Optional[str] = None,
#        corners: Optional[str] = None  # Add this parameter
#    ):
#
# 2. Replace the corner detection section (around line 943):
#    # Perform corner detection
#    if corners:
#        # Parse manual corners from JSON string
#        try:
#            corner_coords = json.loads(corners)
#            if len(corner_coords) == 4 and all(len(c) == 2 for c in corner_coords):
#                corners = np.array(corner_coords, dtype=np.float32)
#                logger.info(f"Using manual corners: {corners}")
#            else:
#                raise ValueError("Invalid corner format")
#        except Exception as e:
#            logger.warning(f"Failed to parse manual corners: {e}, using automatic detection")
#            corners, debug_images = find_corners(cfg, img)
#    else:
#        # Use automatic corner detection
#        corners, debug_images = find_corners(cfg, img)
#
# 3. Add necessary imports at the top:
#    import json
#    import numpy as np
#
# This will allow the API to accept corner coordinates in the format:
# corners='[[993,2294],[2702,2064],[2755,3892],[542,3864]]'
"""
    
    with open('api_manual_corners_patch.txt', 'w') as f:
        f.write(patch_content)
    
    print("📄 Created api_manual_corners_patch.txt with detailed instructions")
    print("\n🎯 Next Steps:")
    print("1. SSH into the server: ssh root@159.203.102.249")
    print("2. Edit main.py: nano /root/chesscog/main.py")
    print("3. Apply the changes from the patch file")
    print("4. Restart the service: systemctl restart chesscog")
    print("5. Test with manual corners")

if __name__ == "__main__":
    update_api_with_manual_corners() 