#!/usr/bin/env python3
"""
Script to replace the entire recognize_chess_position_with_corners function.
"""

import re

# Read the current main.py
with open('/root/chesscog/main.py', 'r') as f:
    content = f.read()

# Read the replacement function
with open('replacement_endpoint.py', 'r') as f:
    replacement = f.read()

# Find the start and end of the existing function
start_pattern = r'@app\.post\("/recognize_chess_position_with_corners"\)'
end_pattern = r'@app\.post\("/detect_corners"\)'

# Find the start position
start_match = re.search(start_pattern, content)
if not start_match:
    print("Could not find the start of the function")
    exit(1)

start_pos = start_match.start()

# Find the end position
end_match = re.search(end_pattern, content)
if not end_match:
    print("Could not find the end of the function")
    exit(1)

end_pos = end_match.start()

# Replace the function
new_content = content[:start_pos] + replacement + '\n\n' + content[end_pos:]

# Write the modified content back
with open('/root/chesscog/main.py', 'w') as f:
    f.write(new_content)

print("Replaced the function successfully")
