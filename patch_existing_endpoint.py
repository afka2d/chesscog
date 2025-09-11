#!/usr/bin/env python3
"""
Script to modify the existing /recognize_chess_position_with_corners endpoint
to return the correct format for the mobile app.
"""

# First, let's find the existing endpoint and replace its return statement
patch_script = '''
import re

# Read the current main.py
with open('/root/chesscog/main.py', 'r') as f:
    content = f.read()

# Find the existing endpoint and replace its return statement
# Look for the return statement in the recognize_chess_position_with_corners endpoint
pattern = r'(@app\.post\("/recognize_chess_position_with_corners".*?return JSONResponse\(\s*content=\{[^}]+\)\s*\))'

# Find the endpoint
match = re.search(pattern, content, re.DOTALL)
if match:
    print("Found existing endpoint")
    
    # Replace the return statement with the correct format
    new_return = '''return {
                "fen": fen,
                "pieces": pieces_list,
                "occupancy": occupancy_list,
                "success": True
            }'''
    
    # Find the return JSONResponse part and replace it
    return_pattern = r'return JSONResponse\(\s*content=\{[^}]+\}\s*\)'
    new_content = re.sub(return_pattern, new_return, content, flags=re.DOTALL)
    
    # Write the modified content back
    with open('/root/chesscog/main.py', 'w') as f:
        f.write(new_content)
    
    print("Modified existing endpoint successfully")
else:
    print("Could not find the existing endpoint")
'''

# Write the patch script to the server
with open('patch_script.py', 'w') as f:
    f.write(patch_script)

print("Created patch script")
