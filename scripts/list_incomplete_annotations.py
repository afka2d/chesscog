import os
import json

annotation_dir = 'grey_background_dataset/annotations/train/'

incomplete = []

for fname in sorted(os.listdir(annotation_dir)):
    if not fname.lower().endswith('.json'):
        continue
    path = os.path.join(annotation_dir, fname)
    with open(path, 'r') as f:
        try:
            data = json.load(f)
        except Exception as e:
            print(f"❌ Error reading {fname}: {e}")
            continue
    missing = []
    if 'corners' not in data or not isinstance(data['corners'], list) or len(data['corners']) != 4:
        missing.append('corners')
    if 'fen' not in data or not isinstance(data['fen'], str) or not data['fen'].strip():
        missing.append('fen')
    if missing:
        incomplete.append((fname, missing))

if incomplete:
    print("Incomplete annotation files:")
    for fname, fields in incomplete:
        print(f"- {fname}: missing {', '.join(fields)}")
else:
    print("All annotation files are complete!") 