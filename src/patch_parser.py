import requests
import asyncio
import korvus
from datetime import datetime
import os
from dotenv import load_dotenv
from old.dota2_base import Dota2API, convert_scalars_to_strings

load_dotenv()
database_url = os.getenv("KORVUS_DATABASE_URL")

class DotaPatchAPI:
    def __init__(self, lang="english"):
        self.base_url = "https://www.dota2.com/datafeed"
        self.lang = lang
        self.dota = Dota2API()
        self.hero_name_map = self.dota.get_hero_id_name_map()
        self.item_ability_map = self._get_item_ability_map()

    def _get_item_ability_map(self):
        abilities = self.dota.fetch_item_list()
        return {str(item.get("id")): item.get("name_loc", item.get("name", "Unknown")) for item in abilities}

    def get_patch_list(self):
        url = f"{self.base_url}/patchnoteslist?language={self.lang}"
        res = requests.get(url)
        data = res.json()
        if not data.get("success"):
            raise ValueError("Failed to fetch patch list")
        return data["patches"]

    def get_patch_details(self, version):
        url = f"{self.base_url}/patchnotes?version={version}&language={self.lang}"
        res = requests.get(url)
        return res.json()

    def normalize_note(self, entry):
        if isinstance(entry, dict) and "note" in entry:
            return entry["note"]
        return entry

    def format_patch_document(self, patch_meta, patch_data):
        version = patch_meta["patch_number"]
        timestamp = datetime.utcfromtimestamp(patch_meta["patch_timestamp"]).isoformat()

        # --- General Notes ---
        general_notes = []
        for group in patch_data.get("general_notes", []):
            general_notes.extend([self.normalize_note(n) for n in group.get("generic", [])])

        # --- Items ---
        items = []
        item_names = []
        for item in patch_data.get("items", []):
            ability_id = str(item.get("ability_id"))
            name = self.item_ability_map.get(ability_id, ability_id)
            notes = [self.normalize_note(n) for n in item.get("ability_notes", [])]
            items.append({"name": name, "notes": notes})
            item_names.append(name)

        # --- Heroes ---
        heroes = []
        hero_names = []
        for hero in patch_data.get("heroes", []):
            hero_id = str(hero.get("hero_id"))
            hero_name = self.hero_name_map.get(int(hero_id), hero_id)
            notes = [self.normalize_note(n) for n in hero.get("hero_notes", [])]
            abilities = []
            for ability in hero.get("abilities", []):
                ability_id = str(ability.get("ability_id"))
                ability_notes = [self.normalize_note(n) for n in ability.get("ability_notes", [])]
                abilities.append({"id": ability_id, "notes": ability_notes})
            heroes.append({"name": hero_name, "notes": notes, "abilities": abilities})
            hero_names.append(hero_name)

        # --- Body Summary ---
        text = "\n".join(
            ["General Changes:"] + general_notes +
            ["\nItem Changes:"] + [f"{i['name']}: {note}" for i in items for note in i['notes']] +
            ["\nHero Changes:"] + [f"{h['name']}: {note}" for h in heroes for note in h['notes']]
        )

        metadata = convert_scalars_to_strings({
            "type": "patch",
            "version": version,
            "title": patch_meta["patch_name"],
            "timestamp": timestamp,
            "general": general_notes,
            "items": items,
            "item_names": item_names,
            "heroes": heroes,
            "hero_names": hero_names,
            "source": "dota2.com",
            
        })

        return {
            "id": f"patch_{version.replace('.', '_').lower()}",
            "title": f"Dota 2 Patch {version}",
            "text": text,
            "metadata": metadata
        }

    def get_all_patch_documents(self):
        docs = []
        for patch_meta in self.get_patch_list():
            data = self.get_patch_details(patch_meta["patch_number"])
            if not data.get("success"):
                continue
            content = data
            doc = self.format_patch_document(patch_meta, content)
            docs.append(doc)
        return docs


async def ingest_dota_patch_api():
    api = DotaPatchAPI()
    docs = api.get_all_patch_documents()

    pipeline = korvus.Pipeline(
        "v0",
        {
            "text": {
                "splitter": {
                    "model": "recursive_character",
                    "parameters": {"chunk_size": 1500, "chunk_overlap": 40}
                },
                "semantic_search": {"model": "Alibaba-NLP/gte-base-en-v1.5"},
                "full_text_search": {"configuration": "english"}
            }
        }
    )

    collection = korvus.Collection("dota_collection", database_url)
    await collection.add_pipeline(pipeline)
    await collection.upsert_documents(docs)
    print(f"Ingested {len(docs)} patches from dota2.com API.")

if __name__ == "__main__":
    asyncio.run(ingest_dota_patch_api())
