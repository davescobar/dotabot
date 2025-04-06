# analyzers/behavior_flag_utils.py

from analyzers.behavior_flags import BEHAVIOR_FLAGS
from analyzers.friendly_behavior_map import FRIENDLY_BEHAVIOR_MAP
from analyzers.categorize_behavior_flag import categorize_behavior_flag

def decode_behavior_flags(raw_value):
    """Returns a list of flag names that are active in the bitmask."""
    return [
        flag_name
        for flag_name, bit in BEHAVIOR_FLAGS.items()
        if raw_value & bit
    ]

def get_structured_behavior_traits(raw_value):
    """Returns a structured dictionary of decoded + categorized behavior flags."""
    flags = decode_behavior_flags(raw_value)

    behavior_structured = {
        "cast_type": [],
        "cast_range": [],
        "special_flags": [],
        "target_team": [],
        "target_type": [],
        "uncategorized": []
    }

    for flag in flags:
        friendly = FRIENDLY_BEHAVIOR_MAP.get(flag, flag)
        category, label = categorize_behavior_flag(flag, friendly)
        behavior_structured.setdefault(category, []).append(label)

    return behavior_structured
