# dota_heroes.py
import requests, json, os, asyncio
from datetime import datetime
from dotenv import load_dotenv
import korvus
from old.dota_utils import format_document

load_dotenv()
database_url = os.getenv("KORVUS_DATABASE_URL")

class DotaHeroAPI:
    def __init__(self):
        self.opendota_url = "https://api.opendota.com/api"
        self.dota_url = "https://www.dota2.com/datafeed"
        self.primary_attr_map = {1: "STR", 2: "AGI", 3: "INT"}
        self.lane_role_map = {1: "Safe Lane", 2: "Mid Lane", 3: "Off Lane", 4: "Jungle"}
        self.complexity_map = {1: "Simple", 2: "Moderate", 3: "Hard"}
        self.hero_id_name_map = self._get_hero_id_name_map()

    def _get_hero_id_name_map(self):
        r = requests.get(f"{self.opendota_url}/heroes")
        return {h["id"]: h["localized_name"] for h in r.json()} if r.ok else {}

    def fetch_hero_list(self):
        r = requests.get(f"{self.dota_url}/herolist?language=english")
        return r.json()['result']['data']['heroes'] if r.ok else []

    def fetch_hero_data(self, hero_id):
        r = requests.get(f"{self.dota_url}/herodata?language=english&hero_id={hero_id}")
        return r.json()['result']['data']['heroes'] if r.ok else {}

    def fetch_opendota_hero_stats(self):
        r = requests.get(f"{self.opendota_url}/heroStats")
        return {h["id"]: h for h in r.json()} if r.ok else {}

    def fetch_opendota_item_popularity(self, hero_id):
        r = requests.get(f"{self.opendota_url}/heroes/{hero_id}/itemPopularity")
        return r.json() if r.ok else {}

    def fetch_opendota_benchmarks(self, hero_id):
        r = requests.get(f"{self.opendota_url}/benchmarks?hero_id={hero_id}")
        return r.json().get("result", {}) if r.ok else {}

    def fetch_opendota_matchups(self, hero_id):
        r = requests.get(f"{self.opendota_url}/heroes/{hero_id}/matchups")
        return r.json() if r.ok else []

    def fetch_opendota_durations(self, hero_id):
        r = requests.get(f"{self.opendota_url}/heroes/{hero_id}/durations")
        return r.json() if r.ok else []

    def fetch_opendota_lane_roles(self, hero_id):
        r = requests.get(f"{self.opendota_url}/scenarios/laneRoles?hero_id={hero_id}")
        return r.json() if r.ok else []

    def fetch_opendota_scenarios(self, hero_id):
        r = requests.get(f"{self.opendota_url}/scenarios?hero_id={hero_id}")
        return r.json() if r.ok else []

    def format_hero_documents(self, hero, opendota_stats):
        hero = hero[0]
        hero_id = hero['id']
        now = datetime.utcnow().isoformat()

        primary_attr = self.primary_attr_map.get(hero["primary_attr"], str(hero["primary_attr"]))
        complexity = self.complexity_map.get(hero.get("complexity"), str(hero.get("complexity")))
        roles = hero.get("roles", [])
        stats = hero.get("stats", {})
        name = hero["name_loc"]

        hero_stat = opendota_stats.get(hero_id, {})
        performance = {
            "pick_rates": {k: hero_stat.get(f"{k}_pick") for k in range(1, 9)},
            "win_rates": {k: hero_stat.get(f"{k}_win") for k in range(1, 9)}
        }

        abilities = [a.get("name_loc", a.get("name")) for a in hero.get("abilities", [])]
        talents = [t.get("name_loc", t.get("name")) for t in hero.get("talents", [])]
        facets = [f.get("title_loc", f.get("name")) for f in hero.get("facets", [])]

        item_builds = self.fetch_opendota_item_popularity(hero_id)
        benchmarks = self.fetch_opendota_benchmarks(hero_id)
        matchups = [
            {
                "opponent": self.hero_id_name_map.get(m["hero_id"]),
                "games_played": m["games_played"],
                "wins": m["wins"]
            }
            for m in self.fetch_opendota_matchups(hero_id)
        ]
        win_by_duration = self.fetch_opendota_durations(hero_id)
        lane_roles = [
            {
                "lane": self.lane_role_map.get(int(lr["lane_role"]), "Unknown"),
                "games": int(lr["games"]),
                "wins": int(lr["wins"])
            }
            for lr in self.fetch_opendota_lane_roles(hero_id)
        ]
        scenarios = self.fetch_opendota_scenarios(hero_id)

        text = (
            f"Hero: {name}\n"
            f"Primary Attribute: {primary_attr}\n"
            f"Roles: {', '.join(roles)}\n"
            f"Complexity: {complexity}\n"
            f"Stats: {json.dumps(stats)}\n"
            f"Pick/Win Rates: {json.dumps(performance)}\n"
            f"Popular Items: {json.dumps(item_builds)}\n"
            f"Benchmarks: {json.dumps(benchmarks)}\n"
            f"Matchups: {json.dumps(matchups)}\n"
            f"Win by Duration: {json.dumps(win_by_duration)}\n"
            f"Lane Roles: {json.dumps(lane_roles)}\n"
            f"Scenarios: {json.dumps(scenarios)}"
        )

        doc = format_document(
            doc_id=f"hero_{hero_id}",
            title=name,
            text=text,
            metadata={
                "type": "hero",
                "hero_id": hero_id,
                "name": name,
                "primary_attr": primary_attr,
                "roles": roles,
                "complexity": complexity,
                "stats": stats,
                "performance": performance,
                "item_builds": item_builds,
                "benchmarks": benchmarks,
                "matchups": matchups,
                "win_by_duration": win_by_duration,
                "lane_roles": lane_roles,
                "scenarios": scenarios,
                "abilities": abilities,
                "talents": talents,
                "facets": facets
            }
        )

        return [doc]

async def ingest_all_heroes():
    dota = DotaHeroAPI()
    collection =  korvus.Collection("dota_collection", database_url)
    pipeline = await collection.get_pipeline("heroesv1")
    collection.add_pipeline(pipeline)
    stats = dota.fetch_opendota_hero_stats()
    all_docs = []  # Collect all documents here

    for hero in dota.fetch_hero_list():
        data = dota.fetch_hero_data(hero["id"])
        if not data:
            continue
        docs = dota.format_hero_documents(data, stats)
        all_docs.extend(docs)  # Add documents to the list

    if all_docs:
        await collection.upsert_documents(all_docs)  # Upload all documents at once

if __name__ == "__main__":
    asyncio.run(ingest_all_heroes())