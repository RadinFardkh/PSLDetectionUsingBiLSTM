import os
import json
import config

# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROCESSED_DIR = config.PROCESSED_DIR
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "samples.json")
CLASS_MAP_FILE = os.path.join(PROCESSED_DIR, "class_map.json")

# --------------------------------------------------
# Load class map
# --------------------------------------------------

with open(os.path.join(PROCESSED_DIR, "class_map.json"), "r", encoding="utf-8") as f:
    class_map = json.load(f)

# Convert:
# {"0": "bashe", "1": "hello"}
#
# into:
# {"bashe": 0, "hello": 1}

class_to_idx = {
    class_name: int(index)
    for class_name, index in class_map.items()
}

# --------------------------------------------------
# Find samples
# --------------------------------------------------

samples = []

for filename in sorted(os.listdir(PROCESSED_DIR)):

    if not filename.endswith(".npy"):
        continue

    name = os.path.splitext(filename)[0]

    # Expected:
    # hello_0
    # bashe_161

    parts = name.rsplit("_", 1)

    if len(parts) != 2:
        print(f"Skipping invalid filename: {filename}")
        continue

    class_name, sample_id = parts

    if not sample_id.isdigit():
        print(f"Skipping invalid filename: {filename}")
        continue

    if class_name not in class_to_idx:
        print(f"Skipping unknown class: {filename}")
        continue

    samples.append({
        "file": filename,
        "class": class_name,
        "label": class_to_idx[class_name]
    })

# --------------------------------------------------
# Save
# --------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(samples, f, indent=4, ensure_ascii=False)

print(f"Found {len(samples)} samples.")
print(f"Saved samples to: {OUTPUT_FILE}")
