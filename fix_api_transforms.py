#!/usr/bin/env python3
"""
Fix API transforms to match training configuration
"""

import re

def fix_api_transforms():
    """Fix the API transforms to match training configuration."""
    
    print("🔧 Fixing API transforms...")
    print("⚠️  This requires manual editing of the server's main.py file")
    
    # Find and replace the occupancy classifier transforms
    # Look for patterns like transforms.Resize((224, 224)) and replace with (100, 100)
    
    # This is a simplified fix - in practice, you'd need to identify the exact location
    # where the occupancy classifier transforms are defined
    
    print("🔧 Fixing API transforms...")
    print("⚠️  This requires manual editing of the server's main.py file")
    print("\n📝 Instructions:")
    print("1. SSH into the server: ssh root@159.203.102.249")
    print("2. Edit the main.py file: nano /root/chesscog/main.py")
    print("3. Find the occupancy classifier transforms (likely around line 200-300)")
    print("4. Change transforms.Resize((224, 224)) to transforms.Resize((100, 100))")
    print("5. Save and restart the service: systemctl restart chesscog")
    
    # Alternative: Create a patch file
    patch_content = """
# Patch for main.py - Fix occupancy classifier transforms
# Find this line:
# transforms.Resize((224, 224))
# Replace with:
# transforms.Resize((100, 100))

# The occupancy classifier was trained with 100x100 images, but the API is using 224x224
# This mismatch causes the model to fail completely
"""
    
    with open('api_transform_patch.txt', 'w') as f:
        f.write(patch_content)
    
    print("\n📄 Created api_transform_patch.txt with instructions")

if __name__ == "__main__":
    fix_api_transforms() 