import os
import re
import json

# ==========================
# CONFIG
# ==========================
DATASET_DIR = r'D:\SchoolProject\newV\processedV2'   # <-- Change this

# ==========================
# Find all .npy files
# ==========================
files = [f for f in os.listdir(DATASET_DIR) if f.endswith(".npy")]

# Natural sort (0,1,2,...10 instead of 0,1,10,100)
def natural_key(text):
    return [int(c) if c.isdigit() else c.lower()
            for c in re.split(r'(\d+)', text)]

files.sort(key=natural_key)

# ==========================
# Extract class names
# ==========================
classes = []

for f in files:
    m = re.match(r"(.+?)_(\d+)\.npy$", f)
    if not m:
        print(f"Skipping invalid filename: {f}")
        continue

    cls = m.group(1)

    if cls not in classes:
        classes.append(cls)

# Alphabetical labels
classes.sort()

class_map = {cls: i for i, cls in enumerate(classes)}

# ==========================
# Generate samples.json
# ==========================
samples = []

for f in files:
    m = re.match(r"(.+?)_(\d+)\.npy$", f)
    if not m:
        continue

    cls = m.group(1)

    samples.append({
        "file": f,
        "class": cls,
        "label": class_map[cls]
    })

# ==========================
# Save files
# ==========================
with open(os.path.join(DATASET_DIR, "class_map.json"), "w", encoding="utf-8") as fp:
    json.dump(class_map, fp, indent=2, ensure_ascii=False)

with open(os.path.join(DATASET_DIR, "samples.json"), "w", encoding="utf-8") as fp:
    json.dump(samples, fp, indent=2, ensure_ascii=False)

print(f"Found {len(classes)} classes.")
print(f"Found {len(samples)} samples.")
print("\nClasses:")

for c, i in class_map.items():
    print(f"{i:2d} -> {c}")