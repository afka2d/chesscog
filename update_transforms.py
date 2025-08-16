#!/usr/bin/env python3
"""
Update transforms in manual_corner_recognizer.py to match training configuration.
"""

import re

def update_transforms():
    """Update the transforms in manual_corner_recognizer.py."""
    with open('/root/chesscog/manual_corner_recognizer.py', 'r') as f:
        content = f.read()
    
    # Update piece transform size
    content = re.sub(
        r'transforms\.Resize\(\(100, 200\)\)',
        'transforms.Resize((224, 448))',
        content
    )
    
    # Update target size in warp_chessboard
    content = re.sub(
        r'target_size = \(800, 800\)',
        'target_size = (1792, 1792)',  # 8 * 224 = 1792
        content
    )
    
    # Update extract_square dimensions
    content = re.sub(
        r'# Calculate square coordinates \(100x100 pixels each\)',
        '# Calculate square coordinates (224x224 pixels each)',
        content
    )
    content = re.sub(
        r'x1 = file \* 100',
        'x1 = file * 224',
        content
    )
    content = re.sub(
        r'y1 = rank \* 100',
        'y1 = rank * 224',
        content
    )
    content = re.sub(
        r'x2 = x1 \+ 100',
        'x2 = x1 + 224',
        content
    )
    content = re.sub(
        r'y2 = y1 \+ 100',
        'y2 = y1 + 224',
        content
    )
    
    with open('/root/chesscog/manual_corner_recognizer.py', 'w') as f:
        f.write(content)
    
    print("✅ Updated transforms in manual_corner_recognizer.py")

if __name__ == "__main__":
    update_transforms()