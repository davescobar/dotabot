from collections import defaultdict, Counter
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from temp import DotaItemAPI

class DotaItemAnalyzer:
    def __init__(self, api):
        self.api = api

    def analyze_items(self):
        item_list = self.api.fetch_item_list()
        field_counts = defaultdict(int)
        field_types = defaultdict(set)
        value_samples = defaultdict(Counter)
        total = 0
        all_items = []

        for item in item_list:
            item_id = item.get("id")
            item_data = self.api.fetch_item_data(item_id)
            if not item_data:
                continue
            all_items.append(item_data)
            total += 1

            for key, value in item_data.items():
                field_counts[key] += 1
                field_types[key].add(type(value).__name__)

                if isinstance(value, (str, int, float, bool)):
                    value_samples[key][value] += 1
                elif isinstance(value, list) and value and isinstance(value[0], (str, int)):
                    for v in value:
                        value_samples[key][v] += 1
                elif isinstance(value, dict):
                    value_samples[key]["<dict>"] += 1
                elif not value:
                    value_samples[key]["<empty>"] += 1
                else:
                    value_samples[key][str(type(value))] += 1

        return {
            "total_items": total,
            "field_counts": dict(field_counts),
            "field_types": {k: list(v) for k, v in field_types.items()},
            "top_values": {
                k: v.most_common(10)
                for k, v in value_samples.items()
            },
        }
    
    def analyze_special_values(self):
        counter = Counter()
        items = self.api.fetch_item_list()
        for item in items:
            item_id = item.get("id")
            data = self.api.fetch_item_data(item_id)
            for val in data.get("special_values", []):
                name = val.get("name")
                if name:
                    counter[name] += 1
        return counter.most_common()

# # Usage
# api = DotaItemAPI()
# analyzer = DotaItemAnalyzer(api)
# report = analyzer.analyze_items()

# with open("item_field_report.json", "w") as f:
#     json.dump(report, f, indent=2)
api = DotaItemAPI()
analyzer = DotaItemAnalyzer(api)
specials = analyzer.analyze_special_values()

with open("special_value_names.json", "w") as f:
    json.dump(specials, f, indent=2)
