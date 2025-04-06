# You can keep this as a utility module: analyzers/behavior_flag_categorizer.py

BEHAVIOR_FLAG_CATEGORIES = {
    "cast_type": {
        "DOTA_ABILITY_BEHAVIOR_PASSIVE",
        "DOTA_ABILITY_BEHAVIOR_NO_TARGET",
        "DOTA_ABILITY_BEHAVIOR_UNIT_TARGET",
        "DOTA_ABILITY_BEHAVIOR_POINT",
        "DOTA_ABILITY_BEHAVIOR_CHANNELLED",
        "DOTA_ABILITY_BEHAVIOR_TOGGLE",
        "DOTA_ABILITY_BEHAVIOR_HIDDEN",
    },
    "cast_range": {
        "DOTA_ABILITY_BEHAVIOR_AOE",
        "DOTA_ABILITY_BEHAVIOR_POINT",
        "DOTA_ABILITY_BEHAVIOR_DIRECTIONAL",
    },
    "special_flags": {
        "DOTA_ABILITY_BEHAVIOR_IMMEDIATE",
        "DOTA_ABILITY_BEHAVIOR_AUTOCAST",
        "DOTA_ABILITY_BEHAVIOR_IGNORE_BACKSWING",
        "DOTA_ABILITY_BEHAVIOR_ROOT_DISABLES",
        "DOTA_ABILITY_BEHAVIOR_UNRESTRICTED",
        "DOTA_ABILITY_BEHAVIOR_IGNORE_CHANNEL",
        "DOTA_ABILITY_BEHAVIOR_DONT_RESUME_ATTACK",
        "DOTA_ABILITY_BEHAVIOR_NORMAL_WHEN_STOLEN",
        "DOTA_ABILITY_BEHAVIOR_IGNORE_PSEUDO_QUEUE",
        "DOTA_ABILITY_BEHAVIOR_IGNORE_CHANNEL_CANCEL",
        "DOTA_ABILITY_BEHAVIOR_DONT_CANCEL_MOVEMENT",
        "DOTA_ABILITY_BEHAVIOR_DONT_CANCEL_CHANNEL",
        "DOTA_ABILITY_BEHAVIOR_ITEM",
        "DOTA_ABILITY_BEHAVIOR_DONT_ALERT_TARGET",
        "DOTA_ABILITY_BEHAVIOR_DONT_RESUME_MOVEMENT",
        "DOTA_ABILITY_BEHAVIOR_NOT_LEARNABLE",
    }
    # You can optionally define other categories if needed
}

def categorize_behavior_flag(flag_name, friendly_name):
    for category, flags in BEHAVIOR_FLAG_CATEGORIES.items():
        if flag_name in flags:
            return category, friendly_name
    return "uncategorized", friendly_name
