"""
Simply a code for generating the class_map.json
"""
import config
import os
import json

class_dirs = sorted([d for d in os.listdir(config.DATA_DIR) if os.path.isdir(os.path.join(config.DATA_DIR, d))])
# Labels every class
class_map = {cls: idx for idx, cls in enumerate(class_dirs)}
# Dumps it into class_map.json
with open('class_map.json', "w") as f:
    json.dump(class_map, f, indent=2)
