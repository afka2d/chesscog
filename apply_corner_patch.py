#!/usr/bin/env python3
"""
Apply corner detection patch to the server
"""

import subprocess
import sys

def apply_corner_patch():
    """Apply the corner detection patch to the server."""
    
    print("🔧 Applying corner detection patch...")
    
    # The command to replace line 943
    cmd = '''ssh root@159.203.102.249 "sed -i '943s/.*/        # Perform corner detection/' /root/chesscog/main.py && sed -i '944s/.*/        if corners:/' /root/chesscog/main.py && sed -i '945s/.*/            # Parse manual corners from JSON string/' /root/chesscog/main.py && sed -i '946s/.*/            try:/' /root/chesscog/main.py && sed -i '947s/.*/                corner_coords = json.loads(corners)/' /root/chesscog/main.py && sed -i '948s/.*/                if len(corner_coords) == 4 and all(len(c) == 2 for c in corner_coords):/' /root/chesscog/main.py && sed -i '949s/.*/                    corners = np.array(corner_coords, dtype=np.float32)/' /root/chesscog/main.py && sed -i '950s/.*/                    logger.info(f\"Using manual corners: {corners}\")/' /root/chesscog/main.py && sed -i '951s/.*/                    debug_images = {}/' /root/chesscog/main.py && sed -i '952s/.*/                else:/' /root/chesscog/main.py && sed -i '953s/.*/                    raise ValueError(\"Invalid corner format\")/' /root/chesscog/main.py && sed -i '954s/.*/            except Exception as e:/' /root/chesscog/main.py && sed -i '955s/.*/                logger.warning(f\"Failed to parse manual corners: {e}, using automatic detection\")/' /root/chesscog/main.py && sed -i '956s/.*/                corners, debug_images = find_corners(cfg, img)/' /root/chesscog/main.py && sed -i '957s/.*/        else:/' /root/chesscog/main.py && sed -i '958s/.*/            # Use automatic corner detection/' /root/chesscog/main.py && sed -i '959s/.*/            corners, debug_images = find_corners(cfg, img)/' /root/chesscog/main.py"'''
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Corner detection patch applied successfully!")
            print("🔄 Restarting the service...")
            
            # Restart the service
            restart_cmd = "ssh root@159.203.102.249 'systemctl restart chesscog'"
            subprocess.run(restart_cmd, shell=True)
            
            print("✅ Service restarted!")
            print("\n🎯 Now you can test the API with manual corners:")
            print("curl -X POST 'https://api.chesspositionscanner.store/recognize_chess_position_with_cursor_description' \\")
            print("  -H 'Content-Type: multipart/form-data' \\")
            print("  -F 'image=@IMG_4752.JPG' \\")
            print("  -F 'corners=[[993,2294],[2702,2064],[2755,3892],[542,3864]]'")
            
        else:
            print(f"❌ Failed to apply patch: {result.stderr}")
            
    except Exception as e:
        print(f"❌ Error applying patch: {e}")

if __name__ == "__main__":
    apply_corner_patch() 