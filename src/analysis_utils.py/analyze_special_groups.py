import json

with open("analysis_utils.py\\special_value_names.json", "r") as f:
    all_special_names = [name for name, _ in json.load(f)]

def get_unmapped_special_keys(all_names, groups):
    grouped_keys = set()
    for keys in groups.values():
        grouped_keys |= set(keys)
    return sorted(name for name in all_names if name not in grouped_keys)

SPECIAL_VALUE_GROUPS = {
    "stats": {"bonus_strength", "bonus_agility", "bonus_intellect", "bonus_all_stats", "bonus_health", "bonus_mana"},
    "regen": {"health_regen", "mana_regen", "bonus_health_regen", "bonus_mana_regen", "aura_health_regen"},
    "utility": {"blink_range", "aura_radius", "projectile_speed", "AbilityCastPoint"},
    "cooldown": {"AbilityCooldown"},
    "mana_cost": {"AbilityManaCost"},
    "damage": {"bonus_damage", "damage"},
    "mobility": {"bonus_movement_speed", "movement_speed"},
    "resistance": {"damage_block", "status_resistance"},
    "lifesteal": {"spell_lifesteal"},
    "slow": {"slow", "slow_duration"},
}

unmapped = get_unmapped_special_keys(all_special_names, SPECIAL_VALUE_GROUPS)
print(json.dumps(unmapped, indent=2))
