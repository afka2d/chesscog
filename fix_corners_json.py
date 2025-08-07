#!/usr/bin/env python3
"""
Fix all annotation JSON files in grey_background_dataset/annotations so that the 'corners' field is a list of lists, not a list of tuples.
"""
import os
import json
from pathlib import Path
import ast

def fix_corners(obj):
    if isinstance(obj, list):
        return [fix_corners(x) for x in obj]
    if isinstance(obj, tuple):
        return [fix_corners(x) for x in obj]
    return obj

ann_dir = Path('grey_background_dataset/annotations')
for json_file in ann_dir.rglob('*.json'):
    try:
        with open(json_file, 'r') as f:
            data = f.read()
            d = ast.literal_eval(data)
        if 'corners' in d:
            d['corners'] = fix_corners(d['corners'])
        with open(json_file, 'w') as f:
            json.dump(d, f, indent=2)
    except Exception as e:
        print(f"Failed to fix {json_file}: {e}")
print('All annotation files fixed!') 