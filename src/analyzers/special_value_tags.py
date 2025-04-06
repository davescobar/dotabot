import json
import os

TAG_FILE = os.path.join(os.path.dirname(__file__), '../mappings/special_value_tags.json')

with open(TAG_FILE, 'r') as f:
    SPECIAL_VALUE_TAG_HELPER = json.load(f)

def tag_special_values(special_values):
    tags = set()
    for value in special_values:
        key = value.get("name")
        if not key:
            continue
        for group, members in SPECIAL_VALUE_TAG_HELPER.items():
            if key in members:
                tags.add(group)
    return list(tags)
