import os
import json
import config


# --------------------------------------------------
# Configuration
# --------------------------------------------------

PROCESSED_DIR = config.PROCESSED_DIR
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "class_map.json")


# --------------------------------------------------
# Find classes
# --------------------------------------------------

classes = set()

for filename in os.listdir(PROCESSED_DIR):

    if not filename.endswith(".npy"):
        continue

    name = os.path.splitext(filename)[0]

    # Expected:
    # hello_0.npy
    # bashe_161.npy

    parts = name.rsplit("_", 1)

    if len(parts) != 2:
        print(f"Skipping invalid filename: {filename}")
        continue

    class_name, sample_id = parts

    # Make sure the final part is a number
    if not sample_id.isdigit():
        print(f"Skipping invalid filename: {filename}")
        continue

    classes.add(class_name)


# --------------------------------------------------
# Create class -> index mapping
# --------------------------------------------------

classes = sorted(classes)

class_map = {
    class_name: index
    for index, class_name in enumerate(classes)
}


# --------------------------------------------------
# Save
# --------------------------------------------------

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(class_map, f, indent=4, ensure_ascii=False)


print(f"Found {len(classes)} classes.")
print(f"Saved class map to: {OUTPUT_FILE}")

for class_name, index in class_map.items():
    print(f"{class_name}: {index}")