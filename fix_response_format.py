#!/usr/bin/env python3
"""
Script to fix the response format in the working main.py
"""

# Read the current main.py
with open('/root/chesscog/main.py', 'r') as f:
    content = f.read()

# Find and replace the return statement
old_return = '''        return JSONResponse(
            content={
                "fen": fen,
                "pieces": pieces_map,
                "occupancy": occupancy_map,
                "success": True
            }
        )'''

new_return = '''        return {
            "fen": fen,
            "pieces": pieces_map,
            "occupancy": occupancy_map,
            "success": True
        }'''

# Replace the return statement
new_content = content.replace(old_return, new_return)

# Write the modified content back
with open('/root/chesscog/main.py', 'w') as f:
    f.write(new_content)

print("Fixed response format successfully")
