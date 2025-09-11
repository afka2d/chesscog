#!/usr/bin/env python3
"""
Simple script to modify the existing endpoint.
"""

# Read the current main.py
with open('/root/chesscog/main.py', 'r') as f:
    lines = f.readlines()

# Find the line with the return JSONResponse and replace it
for i, line in enumerate(lines):
    if 'return JSONResponse(' in line and 'content={' in line:
        # Find the end of this return statement
        j = i
        while j < len(lines) and not (lines[j].strip().endswith(')') and '}' in lines[j]):
            j += 1
        
        # Replace the entire return statement
        new_lines = lines[:i]
        new_lines.append('        # Convert pieces to the format expected by the app\n')
        new_lines.append('        pieces_list = []\n')
        new_lines.append('        for square, piece in zip(recognizer._squares, pieces):\n')
        new_lines.append('            if piece is not None:\n')
        new_lines.append('                # Convert piece to string representation\n')
        new_lines.append('                piece_str = piece.symbol()\n')
        new_lines.append('                pieces_list.append(piece_str)\n')
        new_lines.append('            else:\n')
        new_lines.append('                pieces_list.append(None)\n')
        new_lines.append('        \n')
        new_lines.append('        # Convert occupancy to list format\n')
        new_lines.append('        occupancy_list = occupancy.tolist() if hasattr(occupancy, \'tolist\') else list(occupancy)\n')
        new_lines.append('        \n')
        new_lines.append('        return {\n')
        new_lines.append('            "fen": fen,\n')
        new_lines.append('            "pieces": pieces_list,\n')
        new_lines.append('            "occupancy": occupancy_list,\n')
        new_lines.append('            "success": True\n')
        new_lines.append('        }\n')
        new_lines.extend(lines[j+1:])
        
        # Write the modified content back
        with open('/root/chesscog/main.py', 'w') as f:
            f.writelines(new_lines)
        
        print(f"Modified return statement at line {i+1}")
        break
else:
    print("Could not find the return statement to modify")
