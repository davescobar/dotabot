import xml.etree.ElementTree as ET
import mwparserfromhell
import korvus
import os
from datetime import datetime
from dotenv import load_dotenv
import asyncio

load_dotenv()
database_url = os.getenv("KORVUS_DATABASE_URL")

class LiquipediaGameMechanicsIngestor:
    def __init__(self, xml_path):
        self.xml_path = xml_path

    def parse_wikitext_from_xml(self):
        tree = ET.parse(self.xml_path)
        root = tree.getroot()
        ns = {'mw': 'http://www.mediawiki.org/xml/export-0.11/'}
        pages = root.findall('mw:page', ns)
        docs = []

        for page in pages:
            title = page.find('mw:title', ns).text
            text_elem = page.find('.//mw:text', ns)
            if text_elem is not None and text_elem.text:
                docs.append((title, text_elem.text))
        return docs

    def strip_wikitext(self, wikitext):
        return mwparserfromhell.parse(wikitext).strip_code()

    def format_game_mechanics_documents(self, pages):
        now = datetime.utcnow().isoformat()
        docs = []
        for title, raw_text in pages:
            plain_text = self.strip_wikitext(raw_text)
            docs.append({
                "id": f"game_mechanics_{title.replace(' ', '_')}",
                "title": f"Game Mechanics: {title}",
                "text": plain_text,
                "metadata": {
                    "type": "mechanic",
                    "category": "Game Mechanics",
                    "source": "Liquipedia",
                    "title": title,
                    "first_seen": now,
                    "last_updated": now
                }
            })
        return docs

    async def ingest(self):
        pages = self.parse_wikitext_from_xml()
        docs = self.format_game_mechanics_documents(pages)

        collection = korvus.Collection("dota_collection", database_url)
        await collection.upsert_documents(docs)
        print(f"Ingested {len(docs)} game mechanics documents from Liquipedia.")

# Usage Example:
if __name__ == "__main__":
    ingestor = LiquipediaGameMechanicsIngestor("Liquipedia+Dota+2+Wiki-20250327184725.xml")
    asyncio.run(ingestor.ingest())
