#!/usr/bin/env python3
"""
Add 'white_turn' field to all annotation JSON files in grey_background_dataset/annotations and subfolders.
"""
import os
import json
from pathlib import Path

def get_white_turn(fen):
    try:
        parts = fen.split()
        if len(parts) > 1:
            return parts[1] == 'w'
    except Exception:
        pass
    return None

ann_dir = Path('grey_background_dataset/annotations')
for json_file in ann_dir.rglob('*.json'):
    with open(json_file, 'r') as f:
        d = json.load(f)
    fen = d.get('fen', '')
    white_turn = get_white_turn(fen)
    if white_turn is None:
        print(f"Skipping {json_file}: missing or invalid FEN")
        continue
    d['white_turn'] = white_turn
    with open(json_file, 'w') as f:
        json.dump(d, f, indent=2) 