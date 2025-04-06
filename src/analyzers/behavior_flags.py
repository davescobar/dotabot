import json
import os

FLAGS_FILE = os.path.join(os.path.dirname(__file__), '../mappings/behavior_flags.json')

with open(FLAGS_FILE, 'r') as f:
    BEHAVIOR_FLAGS = json.load(f)
