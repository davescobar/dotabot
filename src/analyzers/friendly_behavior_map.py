import json
import os

MAP_FILE = os.path.join(os.path.dirname(__file__), '../mappings/friendly_behavior_map.json')

with open(MAP_FILE, 'r') as f:
    FRIENDLY_BEHAVIOR_MAP = json.load(f)
