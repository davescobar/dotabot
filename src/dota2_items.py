import requests
import json
import asyncio
from dotenv import load_dotenv
from datetime import datetime
import psycopg
import os
import time
import re

# Assuming these utility modules are correctly defined elsewhere
from analyzers.special_value_tags import SPECIAL_VALUE_TAG_HELPER, tag_special_values
# from analyzers.behavior_flags import BEHAVIOR_FLAGS # Not used directly?
# from analyzers.friendly_behavior_map import FRIENDLY_BEHAVIOR_MAP # Not used directly?
# from analyzers.categorize_behavior_flag import categorize_behavior_flag # Not used directly?
from analyzers.behavior_flag_utils import get_structured_behavior_traits

# Import for Windows compatibility if needed, though generally not required
from asyncio import WindowsSelectorEventLoopPolicy
from asyncio import Semaphore # Removed pgml semaphore

# Set the correct event loop policy for Windows if necessary
if os.name == "nt":
    asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())

# --- pgai Helper Functions ---

async def execute_pgai_query(database_url, sql, params=(), retry_attempts=3, retry_delay=2):
    """
    Executes a pgai SQL query with retry logic.

    Args:
        database_url (str): The database connection URL.
        sql (str): The SQL query string.
        params (tuple, optional): Parameters for the SQL query. Defaults to ().
        retry_attempts (int, optional): Number of retry attempts. Defaults to 3.
        retry_delay (int, optional): Delay between retries in seconds. Defaults to 2.

    Returns:
        Any: The result of the query (typically the first column of the first row).

    Raises:
        psycopg.Error: If the query fails after all retry attempts.
    """
    last_exception = None
    for attempt in range(retry_attempts):
        try:
            # Establish a new connection for each attempt/query
            # Consider using a connection pool for better performance in a real application
            conn = await psycopg.AsyncConnection.connect(database_url)
            async with conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql, params)
                    result = await cur.fetchone()
                    return result[0] if result else None # Return the first column or None
        except psycopg.Error as e:
            print(f"❌ PGAI query failed (attempt {attempt + 1}/{retry_attempts}): {e}")
            last_exception = e
            if attempt < retry_attempts - 1:
                await asyncio.sleep(retry_delay * (attempt + 1)) # Exponential backoff might be better
            else:
                print(f"❌ PGAI query failed permanently after {retry_attempts} attempts.")
                raise last_exception
        except Exception as e: # Catch other potential errors
             print(f"❌ An unexpected error occurred during PGAI query (attempt {attempt + 1}/{retry_attempts}): {e}")
             last_exception = e
             if attempt < retry_attempts - 1:
                 await asyncio.sleep(retry_delay * (attempt + 1))
             else:
                 print(f"❌ PGAI query failed permanently after {retry_attempts} attempts due to unexpected error.")
                 # Optionally re-raise a custom exception or the last exception
                 raise last_exception if last_exception else e
    # This part should ideally not be reached if retries are handled correctly
    return None


async def extract_item_actions_llm(description, database_url, model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """
    Uses pgai + LLM to classify Dota 2 item actions from description.
    Returns a list like ["silence", "stun"].
    """

    context = description.strip()

    system_prompt = {
        "role": "system",
        "content": (
            "You are a Dota 2 expert. You extract item effects from item descriptions. "
            "Only return recognized action types from a fixed list."
        )
    }

    user_prompt = {
        "role": "user",
        "content": (
            "Your task is to classify what effects a Dota 2 item performs based on its description.\n"
            "Only return values from this list:\n"
            "['stun', 'silence', 'dispel', 'slow', 'root', 'hex', 'break', "
            "'mana_drain', 'purge', 'buff', 'heal', 'reveal', 'teleport']\n\n"

            "Respond with only a valid JSON list like this: [\"stun\", \"silence\"]\n"
            "Do not include explanations or wrap your answer in markdown/code blocks.\n"
            "Do not guess or infer anything not clearly stated.\n\n"

            "✅ Positive examples:\n"
            "Description: 'Teleports the user to a targeted allied structure.'\n"
            "Response: [\"teleport\"]\n\n"

            "Description: 'Silences enemies in a radius for 5 seconds.'\n"
            "Response: [\"silence\"]\n\n"

            "Description: 'Passively grants bonus health regeneration and armor.'\n"
            "Response: [\"buff\", \"heal\"]\n\n"

            "❌ Negative examples:\n"
            "Description: 'The target becomes immune to stuns for 4 seconds.'\n"
            "Response: []\n\n"

            "Description: 'Cannot be dispelled.'\n"
            "Response: []\n\n"

            "Description: 'Breaks enemy passives and reveals invisible units.'\n"
            "Response: [\"break\", \"reveal\"]\n\n"

            f"Now analyze this description:\n{context}\n"
            "Response:"
        )
    }

    sql = """
    SELECT pgml.transform(
        task => %s::jsonb,
        inputs => ARRAY[
            %s::jsonb,
            %s::jsonb
        ]
    ) AS answer;
    """

    task_config = {
        "task": "text-generation",
        "model": model
    }

    result = await execute_pgai_query(
        database_url,
        sql,
        (json.dumps(task_config), json.dumps(system_prompt), json.dumps(user_prompt))
    )

    if result:
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return sorted(set(str(x).strip().lower() for x in parsed if isinstance(x, str) and x.strip()))
            else:
                print(f"⚠️ Unexpected format (not a list): {parsed}")
                return []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e} – Raw: {result}")
            fallback = re.findall(
                r'"(stun|silence|dispel|slow|root|hex|break|mana_drain|purge|buff|heal|reveal|teleport)"',
                result
            )
            if fallback:
                print(f"ℹ️ Fallback extracted: {fallback}")
                return sorted(set(fallback))
            return []
        except Exception as e:
            print(f"⚠️ Unexpected error parsing result: {e}")
            return []
    else:
        print("⚠️ No result from LLM.")
        return []


async def generate_additional_tags_from_report(report, existing_tags, database_url, model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """
    Uses pgai + LLM to suggest additional strategic tags based on an item report.
    """

    system_prompt = {
        "role": "system",
        "content": (
            "You are a Dota 2 analyst. Your job is to enrich item metadata "
            "by suggesting strategic and semantic tags that describe the item's usage, role, and value, "
            "focusing on gameplay aspects like positioning, team utility, map control, or timing."
        )
    }

    user_prompt = {
        "role": "user",
        "content": (
            "Here is a full report of a Dota 2 item and its existing tags.\n"
            "Your task is to suggest **additional** strategic or gameplay-related tags that are **not** already in the existing list.\n\n"
            "**Focus on:** Positioning, team utility, map control, timing, counters, synergies, item roles (e.g., initiation, escape, sustain, burst).\n\n"
            "**Do NOT suggest:** Basic stats (already covered), cooldown/cost tags (already tagged), generic terms like 'item', 'useful', 'good'.\n\n"
            "**Examples of good additional tags:**\n"
            "- If item provides invisibility or blink -> [\"escape\", \"initiation\", \"positioning\"]\n"
            "- If item burns mana or silences -> [\"anti_caster\", \"mana_control\"]\n"
            "- If item provides AoE lockdown -> [\"team_fight_tool\", \"control\", \"area_denial\"]\n"
            "- If item significantly boosts farming speed -> [\"farming_accelerator\"]\n"
            "- If item counters specific mechanics (e.g., evasion) -> [\"evasion_counter\"]\n\n"
            f"**Existing tags (Do NOT repeat these):**\n{json.dumps(existing_tags)}\n\n"
            f"**Item report:**\n{report}\n\n"
            "Respond ONLY with a valid JSON list of NEW tag strings, like: [\"mobility\", \"escape\", \"anti_caster\"]"
        )
    }

    sql = """
    SELECT pgml.transform(
        task => %s::jsonb,
        inputs => ARRAY[
            %s::jsonb,
            %s::jsonb
        ]
    ) AS answer;
    """

    task_config = {
        "task": "text-generation",
        "model": model
    }

    result = await execute_pgai_query(
        database_url,
        sql,
        (json.dumps(task_config), json.dumps(system_prompt), json.dumps(user_prompt))
    )

    if result:
        try:
            parsed = json.loads(result)
            if isinstance(parsed, list):
                return sorted(set(
                    str(tag).strip().lower().replace(" ", "_")
                    for tag in parsed
                    if isinstance(tag, str) and tag.strip() and str(tag).strip().lower() not in existing_tags
                ))
            else:
                print(f"⚠️ LLM result was not a list: {parsed}")
                return []
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON decode error: {e} – Raw: {result}")
            match = re.search(r'\[(.*?)\]', result)
            if match:
                try:
                    tags = [
                        t.strip().strip('"\'').lower().replace(" ", "_")
                        for t in match.group(1).split(',') if t.strip()
                    ]
                    filtered = [tag for tag in tags if tag and tag not in existing_tags]
                    if filtered:
                        print(f"ℹ️ Fallback extracted: {filtered}")
                        return sorted(set(filtered))
                except Exception as e2:
                    print(f"⚠️ Fallback failed: {e2}")
            return []
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            return []
    else:
        print("⚠️ No result from LLM.")
        return []


async def generate_summary_llm(item_report, database_url, model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """
    Uses pgai + LLM to generate a concise summary of an item report.
    """
    system_prompt = {
        "role": "system",
        "content": (
            "You are a Dota 2 expert. Summarize the factual details and primary function "
            "of the provided Dota 2 item report into 1–2 clear and informative sentences. "
            "Focus on what the item *does* and its main purpose. "
            "Do not introduce the summary (e.g., 'This item...', 'The summary is...'). "
            "Do not say how many sentences there are. Be direct and factual."
        )
    }

    user_prompt = {
        "role": "user",
        "content": f"Generate a concise summary for the following item report:\n\n{item_report}\n\nSummary:"
    }

    sql = """
    SELECT pgml.transform(
        task => %s::jsonb,
        inputs => ARRAY[
            %s::jsonb,
            %s::jsonb
        ]
    ) AS answer;
    """

    task_config = {
        "task": "text-generation",
        "model": model,
        "parameters": {
            "max_tokens": 100,
            "temperature": 0.5
        }
    }

    result = await execute_pgai_query(
        database_url,
        sql,
        (json.dumps(task_config), json.dumps(system_prompt), json.dumps(user_prompt))
    )

    if result:
        try:
            return result.strip()
        except Exception as e:
            print(f"⚠️ Failed to process summary result: {e} – Raw: {result}")
            return "Error generating summary."
    else:
        print("⚠️ Summary generation query returned no result.")
        return "Summary not available."

async def generate_item_report_llm(item_doc, database_url):
    """
    Uses pgai + LLM to generate a structured report about a Dota 2 item.
    """
    context = json.dumps(item_doc, indent=2)

    system_prompt_content = {
        "role": "system",
        "content": (
            "You are a Dota 2 subject matter expert. Your task is to generate a detailed, structured report "
            "about a Dota 2 item using the provided context data. Stick strictly to the requested format. "
            "Analyze the provided 'metadata' and 'description' fields carefully. "
            "Do not add information not present in the context."
        )
    }

    user_input = {
        "role": "user",
        "content": (
            f"Generate a detailed report for the Dota 2 item described in the context below.\n\n"
            "Use this exact format, providing information for each section based *only* on the context:\n"
            "1.  **Overview:** A brief summary of the item's primary function and stats (use the 'description' and key 'metadata').\n"
            "2.  **Components:** List the items required to build this item (use 'metadata.components'). If none, state 'None'.\n"
            "3.  **Stats & Effects:** Detail the numerical stats, passive effects, and active abilities (use 'description', 'metadata.special_attributes', 'metadata.damage_tag', etc.). Be specific.\n"
            "4.  **Usage Notes:** Mention any important usage details or mechanics (use 'description', 'metadata.notes_loc', 'metadata.behavior_traits').\n"
            "5.  **Strategic Value:** Briefly describe its typical role and when it's usually purchased (use 'metadata.role', 'metadata.game_stage', 'metadata.suggested_times').\n\n"
            f"**Context Data:**\n```json\n{context}\n```\n\n"
            "**Report:**"
        )
    }

    sql = """
    SELECT pgml.transform(
        task => '{"task": "text-generation", "model": "meta-llama/Meta-Llama-3.1-8B-Instruct"}'::JSONB,
        inputs => ARRAY[
            %s::JSONB,
            %s::JSONB
        ]
    ) AS answer;
    """
    result = await execute_pgai_query(
        database_url,
        sql,
        (json.dumps(system_prompt_content), json.dumps(user_input))
    )

    return result



async def answer_question_with_rag_llm(question, retrieved_docs, database_url, model="meta-llama/Meta-Llama-3.1-8B-Instruct"):
    """
    Uses pgai + LLM to answer a question based on retrieved documents (RAG).
    """
    if not retrieved_docs:
        return "I couldn't find relevant information to answer that question."

    context = "\n\n---\n\n".join(
        f"**Document {i+1} (ID: {doc.get('id', 'N/A')})**:\n{doc.get('text', '')}"
        for i, doc in enumerate(retrieved_docs)
    )

    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful Dota 2 assistant. Answer the user's question based *only* on the provided context documents. "
            "Be concise and directly answer the question. "
            "If the context does not contain the answer, explicitly state that the information is not available in the provided documents. "
            "Do not make up information or use external knowledge. Cite the document ID(s) if possible (e.g., 'According to item_123...')."
        )
    }

    user_prompt = {
        "role": "user",
        "content": (
            f"**Context Documents:**\n{context}\n\n"
            f"**Question:** {question}\n\n"
            "**Answer:**"
        )
    }

    sql = """
    SELECT pgml.transform(
        task => %s::jsonb,
        inputs => ARRAY[
            %s::jsonb,
            %s::jsonb
        ]
    ) AS answer;
    """

    task_config = {
        "task": "text-generation",
        "model": model,
        "parameters": {
            "max_tokens": 500,
            "temperature": 0.2
        }
    }

    result = await execute_pgai_query(
        database_url,
        sql,
        (json.dumps(task_config), json.dumps(system_prompt), json.dumps(user_prompt))
    )

    if result:
        try:
            answer = json.loads(result)
            return answer.strip() if isinstance(answer, str) else str(answer)
        except Exception as e:
            print(f"⚠️ Failed to parse RAG answer: {e} – Raw: {result}")
            return f"Error processing the answer. Raw response: {result}"
    else:
        print("⚠️ RAG answer generation query returned no result.")
        return "Could not generate an answer."


# --- Main Class ---

load_dotenv()

class DotaItemDocumentGenerator:
    # Constants moved inside or passed as arguments if they vary
    DOTA_DATA_URL = "https://www.dota2.com/datafeed"
    DOTA_LIST_URL = "https://www.dota2.com/datafeed/itemlist?language=english"
    DEFAULT_EMBEDDING_MODEL = 'voyage-3-lite' # Make this configurable if needed
    DEFAULT_LLM_MODEL = 'meta-llama/Meta-Llama-3.1-405B-Instruct' # Make this configurable

    def __init__(self, database_url=None):
        """
        Initializes the generator. Requires database_url.
        """
        self.korvus_url = database_url or os.getenv("KORVUS_DATABASE_URL")
        self.timescale_url = os.getenv("TIMESCALE_DATABASE_URL", None) # Optional, for TimescaleDB if used in the future
        if not self.korvus_url:
            raise ValueError("Database URL must be provided either as an argument or via KORVUS_DATABASE_URL env var.")
        self.dota_list = None # Lazy loaded
        self.item_id_to_name = None # Lazy loaded
        self.summaries = {} # Cache for summaries

    async def initialize_data(self):
        """
        Asynchronously fetches initial data like the item list.
        Call this method after creating an instance.
        """
        print("Initializing Dota item data...")
        await self._fetch_item_list() # Fetch and populate item list and ID map
        await self._fetch_all_summaries() # Fetch existing summaries
        print(f"Initialization complete. Found {len(self.dota_list)} items.")

    def _fetch_item_list_sync(self):
        """Synchronous fetch for item list (consider making async if possible)."""
        try:
            print(f"Fetching item list from {self.DOTA_LIST_URL}...")
            r = requests.get(self.DOTA_LIST_URL, timeout=15) # Added timeout
            r.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            data = r.json()
            items = data.get('result', {}).get('data', {}).get('itemabilities', [])
            print(f"Successfully fetched {len(items)} item definitions.")
            return items
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch item list: {e}")
            return []
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse item list JSON: {e}")
            return []

    async def _fetch_item_list(self):
        """Async wrapper for fetching item list."""
        # In a real async application, use an async HTTP client like aiohttp
        loop = asyncio.get_running_loop()
        self.dota_list = await loop.run_in_executor(None, self._fetch_item_list_sync)
        # Build the id -> name map
        self.item_id_to_name = {
            item["id"]: item.get("name_loc", item.get("name", f"Unknown_{item['id']}"))
            for item in self.dota_list
            if "id" in item # Ensure item has an ID
        } if self.dota_list else {}


    def _fetch_item_data_sync(self, item_id):
        """Synchronous fetch for specific item data."""
        url = f"{self.DOTA_DATA_URL}/itemdata?language=english&item_id={item_id}"
        try:
            r = requests.get(url, timeout=10) # Added timeout
            r.raise_for_status()
            data = r.json()
            items = data.get('result', {}).get('data', {}).get('items', [])
            return items[0] if items else {}
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to fetch item data for ID {item_id}: {e}")
            return {}
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse item data JSON: {e}")
            return {}
        except IndexError:
             print(f"❌ No item data found in response for ID {item_id}")
             return {}

    async def _fetch_item_data(self, item_id):
        """Async wrapper for fetching item data."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._fetch_item_data_sync, item_id)


    async def _fetch_all_summaries(self):
        """Fetches existing summaries from the database."""
        sql_fetch = "SELECT item_id, summary FROM item_summaries"
        print("Fetching existing item summaries from DB...")
        try:
            conn = await psycopg.AsyncConnection.connect(self.timescale_url)
            async with conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql_fetch)
                    results = await cur.fetchall()
                    self.summaries = {str(row[0]): row[1] for row in results} # Ensure ID is string
                    print(f"Fetched {len(self.summaries)} existing summaries.")
        except psycopg.Error as e:
            print(f"❌ Error fetching summaries from DB: {e}")
            self.summaries = {} # Reset cache on error


    async def insert_or_update_summary(self, item_id, item_report, force_update=False):
        """
        Generates (if needed) and upserts an item summary into the item_summaries table.
        Uses the cached summary if available and force_update is False.
        """
        item_id_str = str(item_id) # Ensure string key

        # Check cache first
        if not force_update and item_id_str in self.summaries and self.summaries[item_id_str]:
            # print(f"ℹ️ Using cached summary for item {item_id_str}.")
            return self.summaries[item_id_str]

        # Generate summary using LLM
        print(f"⚙️ Generating summary for item {item_id_str}...")
        item_summary = await generate_summary_llm(item_report, self.korvus_url, model=self.DEFAULT_LLM_MODEL)

        if not item_summary or "Error generating summary" in item_summary:
             print(f"⚠️ Failed to generate summary for item {item_id_str}. Skipping DB update.")
             return self.summaries.get(item_id_str, "Summary not available.") # Return old summary or default

        # Upsert into the database
        sql_upsert = """
            INSERT INTO item_summaries (item_id, summary, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (item_id)
            DO UPDATE SET
                summary = EXCLUDED.summary,
                updated_at = EXCLUDED.updated_at;
        """
        now = datetime.now()
        try:
            conn = await psycopg.AsyncConnection.connect(self.timescale_url)
            async with conn:
                async with conn.cursor() as cur:
                    await cur.execute(sql_upsert, (item_id_str, item_summary, now, now))
                    await conn.commit() # Explicit commit needed for INSERT/UPDATE
                    self.summaries[item_id_str] = item_summary # Update cache
                    print(f"✅ Summary for item {item_id_str} updated in DB.")
                    return item_summary
        except psycopg.Error as e:
            print(f"❌ Failed to upsert summary for item {item_id_str}: {e}")
            # Return the newly generated summary even if DB fails, but don't cache
            return item_summary
        except Exception as e:
            print(f"❌ An unexpected error occurred during summary upsert for {item_id_str}: {e}")
            return item_summary


    def get_item_metadata_from_list(self, item_id):
        """Extracts components and suggested times from the pre-fetched item list."""
        if not self.dota_list:
            print("⚠️ Item list not loaded. Cannot get metadata.")
            return [], []

        for entry in self.dota_list:
            if entry.get("id") == item_id:
                # Extract components from recipes
                readable_components = []
                recipes = entry.get("recipes", [])
                if recipes: # Check if recipes list exists and is not empty
                    for recipe in recipes:
                        if isinstance(recipe, dict): # Ensure recipe is a dictionary
                             component_ids = recipe.get("items", [])
                             if isinstance(component_ids, list): # Ensure items is a list
                                 for component_id in component_ids:
                                     # Look up name, default to Unknown if not found
                                     name = self.item_id_to_name.get(component_id, f"Unknown_{component_id}")
                                     readable_components.append(name)

                # Extract suggested purchase times
                suggested_times = []
                if entry.get("is_pregame_suggested"): suggested_times.append("pregame")
                if entry.get("is_earlygame_suggested"): suggested_times.append("early")
                if entry.get("is_midgame_suggested"): suggested_times.append("mid") # Added midgame
                if entry.get("is_lategame_suggested"): suggested_times.append("late")

                return sorted(list(set(readable_components)), sorted(list(set(suggested_times))))
        return [], [] # Return empty lists if item_id not found

    def build_component_map_from_recipes(self):
        """Builds a map of item_id -> list_of_component_names."""
        if not self.dota_list or not self.item_id_to_name:
            print("⚠️ Item list or ID map not loaded. Cannot build component map.")
            return {}

        component_map = {}
        # First pass: map recipe name to component names
        recipe_to_components = {}
        for item in self.dota_list:
            item_name = item.get("name", "")
            # Check if it's a recipe item and has recipe details
            if item_name.startswith("item_recipe_") and "recipes" in item:
                 recipes_data = item.get("recipes", [])
                 if recipes_data: # Check if recipes list is not empty
                     # A recipe item usually defines the components for ONE resulting item
                     # Let's assume the first recipe list is the relevant one
                     first_recipe = recipes_data[0]
                     if isinstance(first_recipe, dict):
                         component_ids = first_recipe.get("items", [])
                         component_names = [self.item_id_to_name.get(cid) for cid in component_ids if cid in self.item_id_to_name]
                         if component_names:
                             recipe_to_components[item_name] = sorted(component_names)

        # Second pass: map resulting item ID to component names using the recipe map
        for item in self.dota_list:
             item_id = item.get("id")
             item_components_raw = item.get("item_components") # Official component list
             if item_components_raw and isinstance(item_components_raw, list):
                 # Prefer the official component list if available
                 component_names = []
                 for comp_entry in item_components_raw:
                      comp_name = comp_entry.get("itemname")
                      # Check if it's a recipe name
                      if comp_name and comp_name.startswith("item_recipe_"):
                           # If it's a recipe, add its components from our map
                           if comp_name in recipe_to_components:
                               component_names.extend(recipe_to_components[comp_name])
                           else:
                               component_names.append(f"Recipe ({comp_name.replace('item_recipe_', '')})") # Add recipe cost placeholder
                      elif comp_name:
                           # Look up the component item's display name
                           comp_id = next((i.get("id") for i in self.dota_list if i.get("name") == comp_name), None)
                           display_name = self.item_id_to_name.get(comp_id, comp_name) # Fallback to internal name
                           component_names.append(display_name)

                 if component_names:
                     component_map[item_id] = sorted(list(set(component_names))) # Unique, sorted names

             # Fallback: If official components missing, try finding recipe by name convention
             elif "name" in item and not item["name"].startswith("item_recipe_"):
                  potential_recipe_name = f"item_recipe_{item['name'].replace('item_', '')}"
                  if potential_recipe_name in recipe_to_components:
                      component_map[item_id] = recipe_to_components[potential_recipe_name]


        return component_map


    @staticmethod
    def _deep_clean(value):
        """Recursively removes empty values (None, "", [], {}) from dicts and lists."""
        if isinstance(value, dict):
            cleaned = {
                k: DotaItemDocumentGenerator._deep_clean(v)
                for k, v in value.items()
                # Keep key if cleaned value is not None (0 and False are valid)
                if DotaItemDocumentGenerator._deep_clean(v) is not None
            }
            # Return dict if not empty, else None
            return cleaned if cleaned else None
        elif isinstance(value, list):
            cleaned = [DotaItemDocumentGenerator._deep_clean(v) for v in value]
            # Filter out None values from the list
            cleaned = [v for v in cleaned if v is not None]
            # Return list if not empty, else None
            return cleaned if cleaned else None
        elif value in ("", [], {}): # Explicitly check for empty strings, lists, dicts
             return None
        # Keep None as None, keep other values (including 0, False)
        return value


    async def _process_single_item(self, item, component_map):
        """Processes a single item dictionary to create a document."""
        try:
            item_id = item.get("id")
            if not item_id:
                print(f"⚠️ Skipping item due to missing ID: {item.get('name_loc', 'N/A')}")
                return None

            print(f"⚙️ Processing item ID: {item_id} ({item.get('name_loc', item.get('name', f'Unknown_{item_id}'))})...")
            item_data = await self._fetch_item_data(item_id)
            if not item_data:
                print(f"⚠️ No detailed data fetched for item {item_id}. Skipping.")
                return None

            # --- Basic Info ---
            name_loc = item_data.get("name_loc", item.get("name", f"Unknown_{item_id}"))
            if not name_loc or name_loc.startswith("Unknown_"):
                print(f"⚠️ Unknown name for item ID {item_id}. Skipping.")
                return None

            english_name = item_data.get("name_english_loc", name_loc) # Fallback to name_loc
            lore = item_data.get("lore_loc", "")
            description = item_data.get("desc_loc", "")
            notes = item_data.get("notes_loc", []) # Ensure notes is a list
            if isinstance(notes, list) and notes:
                 full_description = description + "\n\n**Notes:**\n" + "\n".join(f"- {note}" for note in notes if note)
            else:
                 full_description = description

            # --- Components & Suggested Times ---
            components, suggested_times = self.get_item_metadata_from_list(item_id)
            # Alternative component fetching using the map (might be more reliable)
            components_from_map = component_map.get(item_id, [])
            if not components and components_from_map:
                 components = components_from_map # Use map data if list data failed

            # --- Behavior Traits ---
            raw_behavior_flags = item_data.get("behavior", 0)
            # Ensure it's an integer or list/string that can be interpreted
            behavior_int = 0
            if isinstance(raw_behavior_flags, (int, float)):
                 behavior_int = int(raw_behavior_flags)
            elif isinstance(raw_behavior_flags, str) and raw_behavior_flags.isdigit():
                 behavior_int = int(raw_behavior_flags)
            # Add handling for list if needed based on `get_structured_behavior_traits`
            behavior_traits = get_structured_behavior_traits(behavior_int) # Expects int flag

            # --- Cost/Cooldown/Damage Tagging ---
            cooldowns = item_data.get("cooldowns", [])
            cooldown = float(cooldowns[0]) if cooldowns and isinstance(cooldowns[0], (int, float, str)) and str(cooldowns[0]).replace('.', '', 1).isdigit() else 0.0
            cooldown_tag = "no_cooldown"
            if cooldown > 60: cooldown_tag = "very_long_cooldown"
            elif cooldown > 30: cooldown_tag = "long_cooldown"
            elif cooldown > 10: cooldown_tag = "medium_cooldown"
            elif cooldown > 0: cooldown_tag = "short_cooldown"

            mana_costs = item_data.get("mana_costs", [])
            mana_cost = int(mana_costs[0]) if mana_costs and isinstance(mana_costs[0], (int, str)) and str(mana_costs[0]).isdigit() else 0
            mana_tag = "no_mana"
            if mana_cost > 150: mana_tag = "high_mana"
            elif mana_cost > 50: mana_tag = "medium_mana"
            elif mana_cost > 0: mana_tag = "low_mana"

            health_costs = item_data.get("health_costs", [])
            health_cost = int(health_costs[0]) if health_costs and isinstance(health_costs[0], (int, str)) and str(health_costs[0]).isdigit() else 0
            health_cost_tag = "no_health_cost"
            if health_cost > 150: health_cost_tag = "high_health_cost"
            elif health_cost > 50: health_cost_tag = "medium_health_cost"
            elif health_cost > 0: health_cost_tag = "low_health_cost"

            gold_cost = item_data.get("item_cost", 0)
            cost_tag = "free"
            if gold_cost > 4000: cost_tag = "very_expensive"
            elif gold_cost > 2000: cost_tag = "expensive"
            elif gold_cost > 500: cost_tag = "moderate_cost"
            elif gold_cost > 0: cost_tag = "cheap"

            damage = item_data.get("damage", 0) # Assuming single damage value
            damage_tag = "no_damage"
            if damage > 150: damage_tag = "high_damage"
            elif damage > 50: damage_tag = "medium_damage"
            elif damage > 0: damage_tag = "low_damage"

            # --- Special Values ---
            special_values = item_data.get("special_values", [])
            special_value_tags = tag_special_values(special_values) # From your analyzer
            special_attributes = {}
            if isinstance(special_values, list):
                for sv in special_values:
                     if isinstance(sv, dict):
                         name = sv.get("name")
                         # Prefer float values, fallback to int
                         vals_float = sv.get("values_float", [])
                         vals_int = sv.get("values_int", [])
                         vals = vals_float if vals_float else vals_int

                         # Clean values (convert strings if possible)
                         cleaned_vals = []
                         if isinstance(vals, list):
                              for v in vals:
                                   if isinstance(v, (int, float)):
                                        cleaned_vals.append(v)
                                   elif isinstance(v, str):
                                        try:
                                             cleaned_vals.append(float(v) if '.' in v else int(v))
                                        except ValueError:
                                             pass # Ignore non-numeric strings

                         if name and cleaned_vals:
                              # Store single value directly, multiple values as dict
                              special_attributes[name] = cleaned_vals[0] if len(cleaned_vals) == 1 else {"values": cleaned_vals}

            # --- Item Type & Role ---
            item_neutral_tier_str = item_data.get("item_neutral_tier", "0")
            item_neutral_tier = int(item_neutral_tier_str) if isinstance(item_neutral_tier_str, str) and item_neutral_tier_str.isdigit() else 0
            item_type = f"neutral_tier_{item_neutral_tier}" if item_neutral_tier > 0 else "regular"
            item_quality = item_data.get("item_quality", "common") # e.g., component, common, rare, artifact

            # Basic role - refine later if needed
            role = "damage" if damage > 0 and cost_tag != "free" else "utility" # Simple initial role

            # --- Build Metadata Dict ---
            # Start with raw data, clean later
            raw_metadata = {
                # Core Identifiers
                "id": item_id,
                "internal_name": item.get("name"), # Keep internal name
                "name_loc": name_loc,
                "name_english_loc": english_name,
                "lore_loc": lore,
                "desc_loc": description, # Original description without notes
                "notes_loc": notes if notes else None, # Keep notes separate

                # Gameplay Tags & Categories
                "item_type": item_type,
                "item_quality": item_quality,
                "role_guess": role, # Simple guess
                "game_stage_suggested": suggested_times if suggested_times else None,

                # Cost & Cooldown Tags
                "cooldown_tag": cooldown_tag,
                "mana_cost_tag": mana_tag,
                "health_cost_tag": health_cost_tag,
                "gold_cost_tag": cost_tag,
                "damage_tag": damage_tag,

                 # Numerical Values (prefer lists even if single value for consistency)
                "item_cost": gold_cost if gold_cost > 0 else None,
                "cooldowns": [cooldown] if cooldown > 0 else None,
                "mana_costs": [mana_cost] if mana_cost > 0 else None,
                "health_costs": [health_cost] if health_cost > 0 else None,
                # "damage": damage if damage > 0 else None, # Maybe add damage type later

                # Behavior & Effects
                "behavior_flags_int": behavior_int if behavior_int > 0 else None,
                "behavior_traits": behavior_traits if behavior_traits else None,
                "special_value_tags": special_value_tags if special_value_tags else None,
                "special_attributes": special_attributes if special_attributes else None,

                # Components & Recipe
                "components": components if components else None,

                # Other useful flags from data
                "item_initial_charges": item_data.get("item_initial_charges"),
                "item_stock_max": item_data.get("item_stock_max"),
                "item_stock_time": item_data.get("item_stock_time"),
                "max_level": item_data.get("max_level"),
                "is_permanent": item_data.get("is_permanent"), # Useful flag
                "disassemblable": item_data.get("disassemblable"), # Useful flag
                "declarations": item_data.get("declarations"), # e.g., "DECLARE_PURCHASES_TO_TEAM"

                # Scepter/Shard related (important)
                "ability_has_scepter": item_data.get("ability_has_scepter"),
                "ability_has_shard": item_data.get("ability_has_shard"),
                "ability_is_granted_by_scepter": item_data.get("ability_is_granted_by_scepter"),
                "ability_is_granted_by_shard": item_data.get("ability_is_granted_by_shard"),
                "scepter_loc": item_data.get("scepter_loc"),
                "shard_loc": item_data.get("shard_loc"),
            }

            # Deep clean the metadata to remove empty/null values
            metadata = self._deep_clean(raw_metadata) or {} # Ensure metadata is at least {}

            # --- Generate LLM Content ---
            # 1. Generate the main item report
            # Context for report generation should be selective
            report_context = {
                 "name": english_name,
                 "description": full_description, # Use description with notes
                 "metadata": { # Select key metadata for the report LLM
                      "item_cost": gold_cost,
                      "components": components,
                      "cooldowns": [cooldown] if cooldown > 0 else None,
                      "mana_costs": [mana_cost] if mana_cost > 0 else None,
                      "special_attributes": special_attributes,
                      "behavior_traits": behavior_traits,
                      "item_quality": item_quality,
                      "item_type": item_type,
                      "suggested_times": suggested_times,
                      "scepter_effect": metadata.get("scepter_loc"),
                      "shard_effect": metadata.get("shard_loc"),
                 }
            }
            item_report = await generate_item_report_llm(self._deep_clean(report_context), self.korvus_url)

            # 2. Extract item actions/effects from the description
            item_actions = await extract_item_actions_llm(item_report, self.korvus_url, model=self.DEFAULT_LLM_MODEL)
            # Add extracted actions to metadata (if not empty)
            if item_actions:
                 metadata["llm_extracted_actions"] = item_actions

            # --- Compile Final Tags ---
            # Combine tags from various sources
            base_tags = [
                item_type,
                item_quality,
                cost_tag,
                cooldown_tag,
                mana_tag,
                health_cost_tag,
                damage_tag,
            ]
            behavior_tag_list = []
            if isinstance(behavior_traits, dict):
                 for category, traits in behavior_traits.items():
                      if isinstance(traits, list):
                           behavior_tag_list.extend(traits)
                      elif isinstance(traits, str): # Handle single string value case
                           behavior_tag_list.append(traits)

            all_tags_before_llm = list(set(
                base_tags +
                (suggested_times or []) +
                behavior_tag_list +
                (special_value_tags or []) +
                (item_actions or []) # Add LLM extracted actions here
            ))
            # Filter out None or empty strings, convert to lower case, replace spaces
            all_tags_before_llm = sorted([tag.lower().replace(" ", "_") for tag in all_tags_before_llm if tag and isinstance(tag, str)])


            # 3. Generate additional strategic tags based on the report and existing tags
            additional_tags = await generate_additional_tags_from_report(
                report=item_report,
                existing_tags=all_tags_before_llm, # Pass current tags to avoid duplicates
                database_url=self.korvus_url,
                model=self.DEFAULT_LLM_MODEL
            )

            # Final combined tag list (unique and sorted)
            final_tags = sorted(list(set(all_tags_before_llm + additional_tags)))

            # Add final tags to metadata
            metadata["crit_tags_generated"] = final_tags # Store the final list in metadata as well

            # --- Generate Summary (using the full report) ---
            item_summary = await self.insert_or_update_summary(item_id, item_report) # Uses LLM

            # --- Construct Final Document ---
            document = {
                "id": f"item_{item_id}", # Standardized ID format
                "title": f"Item: {english_name}", # User-friendly title
                "text": item_report, # The detailed LLM-generated report
                "abstract": item_summary, # The concise LLM-generated summary
                "metadata": metadata, # The cleaned, structured metadata
                "crit_tags": final_tags # The final combined list of tags for filtering/search
            }

            print(f"✅ Successfully processed item ID: {item_id}")
            return document

        except Exception as e:
            print(f"❌❌❌ Unhandled error processing item ID {item.get('id', 'N/A')}: {e}")
            import traceback
            traceback.print_exc() # Print stack trace for debugging
            return None


    async def format_item_documents_in_batches(self, items_to_process=None, batch_size=5):
        """
        Processes items in batches to generate item documents concurrently.
        If items_to_process is None, uses self.dota_list.
        """
        if items_to_process is None:
            items_to_process = self.dota_list

        if not items_to_process:
            print("⚠️ No items to process.")
            return []

        print(f"Starting processing of {len(items_to_process)} items in batches of {batch_size}...")

        # Pre-build the component map once
        component_map = self.build_component_map_from_recipes()

        all_documents = []
        # Split items into batches
        batches = [items_to_process[i:i + batch_size] for i in range(0, len(items_to_process), batch_size)]

        total_batches = len(batches)
        for i, batch in enumerate(batches):
            print(f"\n--- Processing Batch {i+1}/{total_batches} ---")
            # Create tasks for each item in the batch
            tasks = [self._process_single_item(item, component_map) for item in batch]
            # Run tasks concurrently within the batch
            batch_results = await asyncio.gather(*tasks)
            # Filter out None results (errors) and extend the main list
            successful_docs = [doc for doc in batch_results if doc is not None]
            all_documents.extend(successful_docs)
            print(f"--- Batch {i+1} complete. Processed {len(batch)} items ({len(successful_docs)} successful). Total docs: {len(all_documents)} ---")
            # Optional: Add a small delay between batches if needed
            # await asyncio.sleep(1)

        print(f"\n✅ Finished processing all batches. Generated {len(all_documents)} documents.")
        return all_documents

    def _sanitize_name(self, name):
        """Basic sanitization for identifiers."""
        return re.sub(r'[^a-zA-Z0-9_]', '', name)
    
    def create_schema(self, table_name="dota_items", embedding_model=None, vector_size=1024): # Example: Updated vector_size for voyage-lite-02
        """
        Creates the database table, indexes, triggers, and pg_vectorize jobs.
        Uses synchronous connection for schema setup.
        """
        # Use default embedding model if not provided, assuming it's relevant for API key lookup etc.
        # Note: embedding_model parameter here is slightly confusing as the model is hardcoded in SQL below.
        # Consider passing the full model details or API key source here.
        if embedding_model is None:
            embedding_model = self.DEFAULT_EMBEDDING_MODEL # This might be 'ollama/nomic-embed-text' based on your class variable

        sanitized_table_name = self._sanitize_name(table_name)
        trigger_func_name = f"{sanitized_table_name}_tsvector_update"
        trigger_name = f"trg_{sanitized_table_name}_tsvector_update"

        print(f"⚙️ Ensuring schema and vectorizers exist for table `{table_name}`...")
        print(f"   Using vector size: {vector_size}") # Print the vector size being used

        # Ensure vector_size parameter is used in table creation
        create_item_table = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id TEXT PRIMARY KEY,
            title TEXT,
            text TEXT,
            abstract TEXT,
            metadata JSONB,
            crit_tags TEXT[],
            search_tags_text TEXT,
            search_effects_text TEXT,
            search_components_text TEXT,
            fulltext_tsv TSVECTOR,
            tags_tsv TSVECTOR,
            effects_tsv TSVECTOR,
            components_tsv TSVECTOR,
            text_embedding VECTOR({vector_size}), -- Use the vector_size parameter
            abstract_embedding VECTOR({vector_size}) -- Use the vector_size parameter
        );
        """
        
        create_item_summary_table = """
        CREATE TABLE public.item_summaries (
            item_id    VARCHAR(255) PRIMARY KEY,
            summary    TEXT,
            created_at TIMESTAMP,
            updated_at TIMESTAMP
        );
        """
        trigger_function_sql = f"""
        CREATE OR REPLACE FUNCTION {trigger_func_name}() RETURNS trigger AS $$
        BEGIN
            NEW.fulltext_tsv = to_tsvector('english', coalesce(NEW.title, '') || ' ' || coalesce(NEW.text, '') || ' ' || coalesce(NEW.abstract, ''));
            NEW.tags_tsv = to_tsvector('english', coalesce(array_to_string(NEW.crit_tags, ' '), '') || ' ' || coalesce(NEW.search_tags_text, ''));
            NEW.effects_tsv = to_tsvector('english', coalesce(NEW.search_effects_text, ''));
            NEW.components_tsv = to_tsvector('english', coalesce(NEW.search_components_text, ''));
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
        """

        call_trigger_sql = f"""
        DROP TRIGGER IF EXISTS {trigger_name} ON {table_name};
        CREATE TRIGGER {trigger_name}
        BEFORE INSERT OR UPDATE ON {table_name}
        FOR EACH ROW EXECUTE FUNCTION {trigger_func_name}();
        """

        create_indexes_sql = f"""
        CREATE INDEX IF NOT EXISTS idx_{sanitized_table_name}_fulltext_gin ON {table_name} USING GIN(fulltext_tsv);
        CREATE INDEX IF NOT EXISTS idx_{sanitized_table_name}_tags_gin ON {table_name} USING GIN(tags_tsv);
        CREATE INDEX IF NOT EXISTS idx_{sanitized_table_name}_effects_gin ON {table_name} USING GIN(effects_tsv);
        CREATE INDEX IF NOT EXISTS idx_{sanitized_table_name}_components_gin ON {table_name} USING GIN(components_tsv);
        -- Optional: Add HNSW index for the text_embedding column if using pgvector >= 0.5.0
        -- CREATE INDEX IF NOT EXISTS idx_{sanitized_table_name}_text_embedding_hnsw ON {table_name} USING hnsw (text_embedding vector_cosine_ops);
        """

        # --- Modified Vectorizer Job SQL ---
        create_text_vectorizer_job_sql = f"""
        SELECT ai.create_vectorizer(
        '{sanitized_table_name}'::regclass,
        destination => 'dota_item_embeddings',
        embedding => ai.embedding_voyageai(
                'voyage-3-lite',
                512
        ),
        chunking => ai.chunking_recursive_character_text_splitter(
                'text',
                600,
                100,
                separators => array[E'\\n\\n', E'\\n', '.', '!', '?']
            )
        );
        """
        # --- End of Modified Vectorizer Job SQL ---


        try:
            # Use synchronous connection for schema modifications
            with psycopg.connect(self.timescale_url) as conn:
                conn.autocommit = True # Use autocommit for DDL and potentially job creation
                with conn.cursor() as cur:
                    print(f"Creating table {table_name}...")
                    cur.execute(create_item_table)

                    print(f"Creating item summary table...")
                    cur.execute(create_item_summary_table) # Ensure the summary table exists first
                    
                    print("Creating trigger function...")
                    cur.execute(trigger_function_sql)

                    print("Creating trigger...")
                    cur.execute(call_trigger_sql)

                    print("Creating FTS indexes...")
                    cur.execute(create_indexes_sql)

                    # --- Execute Job Creation ---
                    print("Creating/updating vectorizer job for 'text' column...")
                    try:
                        # Execute the vectorizer creation SQL
                        cur.execute(create_text_vectorizer_job_sql)
                        job_result_text = cur.fetchone()
                        print(f"Text vectorizer job setup result: {job_result_text}")
                    except psycopg.errors.UniqueViolation:
                         # This specific error might not occur if create_vectorizer handles updates gracefully
                         print("Vectorizer job configuration possibly already exists or was updated.")
                    except psycopg.Error as job_e:
                         # Catch specific database errors during job creation
                         print(f"❌ Error creating/updating text vectorizer job: {job_e}")
                         print("   Check database logs, pgai extension status, and API key.")
                    except Exception as job_e:
                         # Catch any other unexpected errors
                         print(f"❌ Unexpected error during text vectorizer job setup: {job_e}")


            print(f"✅ Schema and vectorizer job setup potentially complete for table `{table_name}`.")
            print("   Note: Vectorization runs in the background based on the schedule.")

        except psycopg.Error as e:
            print(f"❌ Database error during schema setup for table `{table_name}`: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred during schema setup for `{table_name}`: {e}")

    async def insert_documents(self, documents, table_name="dota_items"):
        """
        Inserts or updates documents into the specified table.
        Populates helper text columns for FTS.
        Embeddings are handled automatically by pgai vectorizer triggers (usually).
        """
        if not documents:
            print("⚠️ No documents provided for insertion.")
            return

        print(f"⚙️ Preparing to insert/update {len(documents)} documents into `{table_name}`...")

        upsert_sql = f"""
        INSERT INTO {table_name} (
            id, title, text, abstract, metadata, crit_tags,
            search_tags_text, search_effects_text, search_components_text
            -- embedding column is handled by pgai vectorizer trigger
        )
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (id) DO UPDATE SET
            title = EXCLUDED.title,
            text = EXCLUDED.text,
            abstract = EXCLUDED.abstract,
            metadata = EXCLUDED.metadata,
            crit_tags = EXCLUDED.crit_tags,
            search_tags_text = EXCLUDED.search_tags_text,
            search_effects_text = EXCLUDED.search_effects_text,
            search_components_text = EXCLUDED.search_components_text;
            -- Don't update embedding here, trigger should handle it if text/abstract changes
        """

        insert_count = 0
        error_count = 0
        conn = None # Initialize conn outside try block

        try:
            conn = await psycopg.AsyncConnection.connect(self.timescale_url)
            async with conn.pipeline(): # Use pipeline for potential performance boost
                async with conn.cursor() as cur:
                    for doc in documents:
                        try:
                            metadata = doc.get("metadata", {})
                            crit_tags = doc.get("crit_tags", []) # Use the final combined tags

                            # --- Prepare helper text for FTS ---
                            # 1. Tags Text: Combine crit_tags and maybe other relevant metadata tags
                            search_tags_list = list(crit_tags) # Start with crit_tags
                            # Add other potential tag sources from metadata if needed
                            if metadata.get("item_quality"): search_tags_list.append(metadata["item_quality"])
                            if metadata.get("role_guess"): search_tags_list.append(metadata["role_guess"])
                            search_tags_text = " ".join(sorted(list(set(str(tag) for tag in search_tags_list))))

                            # 2. Effects Text: Combine actions, special value tags, behavior traits
                            effect_terms = []
                            if isinstance(metadata.get("llm_extracted_actions"), list):
                                effect_terms.extend(metadata["llm_extracted_actions"])
                            if isinstance(metadata.get("special_value_tags"), list):
                                effect_terms.extend(metadata["special_value_tags"])
                            if isinstance(metadata.get("behavior_traits"), dict):
                                for traits in metadata["behavior_traits"].values():
                                    if isinstance(traits, list): effect_terms.extend(traits)
                                    elif isinstance(traits, str): effect_terms.append(traits)
                            search_effects_text = " ".join(sorted(list(set(effect_terms))))

                            # 3. Components Text: Just the list of component names
                            components = metadata.get("components", [])
                            search_components_text = " ".join(components) if isinstance(components, list) else ""

                            # --- Execute Upsert ---
                            await cur.execute(
                                upsert_sql,
                                (
                                    doc.get("id", ""), # Ensure ID exists
                                    doc.get("title", ""),
                                    doc.get("text", ""),
                                    doc.get("abstract", ""),
                                    json.dumps(metadata) if metadata else None, # Store metadata as JSON string
                                    crit_tags if crit_tags else None, # Store tags as array
                                    search_tags_text,
                                    search_effects_text,
                                    search_components_text
                                )
                            )
                            insert_count += 1
                        except psycopg.Error as e:
                            error_count += 1
                            print(f"❌ DB Error inserting/updating doc ID {doc.get('id', 'N/A')}: {e}")
                        except Exception as e:
                            error_count += 1
                            print(f"❌ Unexpected Error processing doc ID {doc.get('id', 'N/A')} for DB insert: {e}")

            # Explicit commit after the loop (though pipeline might handle it)
            await conn.commit()
            print(f"✅ Finished DB insert/update. Successful: {insert_count}, Errors: {error_count}")

        except psycopg.Error as e:
            print(f"❌ Database connection or transaction failed: {e}")
            # Rollback might be needed if not using autocommit or pipeline handles it
        except Exception as e:
            print(f"❌ An unexpected error occurred during document insertion: {e}")
        finally:
             if conn and not conn.closed:
                 await conn.close()


    async def retrieve_documents_vector(self, question, top_k=5, table_name="dota_items", embedding_model=None, vector_column='embedding'):
        """
        Retrieves documents based on vector similarity using pgai's embedding generation.
        """
        if embedding_model is None:
            embedding_model = self.DEFAULT_EMBEDDING_MODEL

        print(f"🔍 Retrieving top {top_k} documents for question using vector search ('{embedding_model}')...")

        # SQL using ai.embed for the query vector and vector similarity search (<=>)
        # Assumes the 'embedding' column exists and is populated by the vectorizer
        query_sql = f"""
            WITH query_embedding AS (
                SELECT ai.voyageai_embed(
                    %s,
                    %s
                ) AS q_vec
            )
            SELECT
                t.id, t.title, t.text, t.abstract, t.metadata, t.crit_tags,
                -- Calculate cosine similarity (1 - cosine distance)
                1 - (t.{vector_column} <=> qe.q_vec) AS score
            FROM
                {table_name} t, query_embedding qe
            WHERE
                t.{vector_column} IS NOT NULL -- Ensure embedding exists
            ORDER BY
                score DESC -- Order by similarity score descending
            LIMIT %s;
        """
        results = []
        conn = None
        try:
            conn = await psycopg.AsyncConnection.connect(self.timescale_url)
            async with conn.cursor() as cur:
                await cur.execute(query_sql, (embedding_model, question, top_k))
                results = await cur.fetchall()
                print(f"Retrieved {len(results)} documents.")
        except psycopg.Error as e:
            print(f"❌ Failed to retrieve documents via vector search: {e}")
            # Check if vector extension is enabled, table/column exists, model is valid etc.
            if "function ai.embed(" in str(e):
                 print("⚠️ Hint: Ensure the 'pg_vectorize' (or relevant pgai) extension is installed and ai.embed function is available.")
            if f'column "{vector_column}" does not exist' in str(e):
                 print(f"⚠️ Hint: Ensure the '{vector_column}' column exists in the '{table_name}' table and is of VECTOR type.")
            if "<=>" in str(e):
                 print("⚠️ Hint: Ensure the 'vector' extension is installed for the <=> operator.")

        except Exception as e:
            print(f"❌ An unexpected error occurred during vector retrieval: {e}")
        finally:
             if conn and not conn.closed:
                 await conn.close()

        # Format results into dictionaries
        return [
            {
                "id": row[0],
                "title": row[1],
                "text": row[2],
                "abstract": row[3],
                "metadata": json.loads(row[4]) if isinstance(row[4], str) else row[4], # Parse JSONB
                "crit_tags": row[5],
                "score": row[6],
            }
            for row in results
        ]


    async def answer_question_with_rag(self, question, top_k=3, table_name="dota_items"):
        """
        Performs RAG: retrieves documents using vector search and generates an answer.
        """
        # 1. Retrieve relevant documents
        retrieved_docs = await self.retrieve_documents_vector(question, top_k, table_name)

        if not retrieved_docs:
            print("⚠️ No relevant documents found for the question.")
            return "I could not find any relevant information to answer your question."

        # 2. Generate answer using LLM with the retrieved context
        print("🧠 Generating answer based on retrieved documents...")
        answer = await answer_question_with_rag_llm(
            question,
            retrieved_docs,
            self.korvus_url,
            model=self.DEFAULT_LLM_MODEL
        )

        return answer


    def write_documents_to_json(self, documents, path="dota_items_output.json"):
        """Writes the generated documents list to a JSON file."""
        if not documents:
             print("⚠️ No documents to write to JSON.")
             return
        try:
            print(f"💾 Writing {len(documents)} documents to {path}...")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(documents, f, indent=2, ensure_ascii=False) # Use indent=2 for readability
            print(f"✅ Successfully wrote documents to {path}")
        except IOError as e:
            print(f"❌ Failed to write JSON file {path}: {e}")
        except TypeError as e:
             print(f"❌ Failed to serialize documents to JSON: {e}")
        except Exception as e:
            print(f"❌ An unexpected error occurred writing JSON: {e}")


# --- Main Execution ---

async def ingest_all_items(load_from_json=False, json_path="dota_items_output.json", table_name="dota_items"):
    """
    Main workflow: Initialize, fetch/load data, process, create schema, insert.
    """
    start_time = time.time()
    print("-" * 30)
    print("🚀 Starting Dota Item Ingestion Process...")
    print("-" * 30)

    # Initialize the generator (requires DB URL)
    generator = DotaItemDocumentGenerator()

    docs = []
    if load_from_json:
        # Load documents from JSON backup
        print(f"💾 Loading documents from JSON file: {json_path}")
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                docs = json.load(f)
            print(f"✅ Loaded {len(docs)} documents from {json_path}")
        except FileNotFoundError:
             print(f"❌ JSON file not found: {json_path}. Will attempt to fetch and generate.")
             load_from_json = False # Force fetch if file not found
        except json.JSONDecodeError as e:
            print(f"❌ Failed to parse JSON file {json_path}: {e}. Will attempt to fetch and generate.")
            load_from_json = False # Force fetch on parse error
        except Exception as e:
            print(f"❌ Failed to load JSON ({json_path}): {e}. Will attempt to fetch and generate.")
            load_from_json = False # Force fetch on other errors

    #generator.create_schema(table_name=table_name, vector_size=768) # Adjust vector_size if needed
    if not load_from_json or not docs:
        # Fetch item list and process items if not loading from JSON or if loading failed
        await generator.initialize_data() # Fetch item list, summaries
        if not generator.dota_list:
             print("❌ Cannot proceed without Dota item list. Exiting.")
             return

        # Filter out recipe items before processing
        items_to_process = [item for item in generator.dota_list if not item.get("name", "").startswith("item_recipe_")]
        print(f"ℹ️ Filtered out recipes. Processing {len(items_to_process)} actual items.")

        # Process items in batches to generate documents
        docs = await generator.format_item_documents_in_batches(items_to_process, batch_size=5) # Smaller batch size

        # Save the generated documents to JSON as a backup
        generator.write_documents_to_json(docs, json_path)


    if not docs:
         print("❌ No documents generated or loaded. Cannot insert into database. Exiting.")
         return

    # Create schema and vectorizers in the database
    # Pass the vector size used by your embedding model
    

    # Insert documents into the database
    await generator.insert_documents(docs, table_name=table_name)

    end_time = time.time()
    print("-" * 30)
    print(f"✅ Ingestion Process Completed in {end_time - start_time:.2f} seconds.")
    print("-" * 30)

    # Example RAG Question
    print("\n--- RAG Example ---")
    # question = input("Ask a question about Dota items (e.g., 'What does Black King Bar do?', 'Suggest items for survivability'): ")
    question = "What items provide invisibility?"
    print(f"❓ Question: {question}")
    if question:
        answer = await generator.answer_question_with_rag(question, table_name=table_name)
        print(f"\n🤖 Answer:\n{answer}")


if __name__ == "__main__":
    # Use asyncio.run() for the main entry point
    try:
        # Set load_from_json=True to skip fetching/processing and load from file
        asyncio.run(ingest_all_items(load_from_json=True))
    except RuntimeError as e:
        # Handle nested event loop issue if running in an environment like Jupyter
        if "cannot be called from a running event loop" in str(e):
            print("Detected running event loop. Applying nest_asyncio workaround...")
            import nest_asyncio
            nest_asyncio.apply()
            # Rerun the main function after applying the workaround
            asyncio.run(ingest_all_items(load_from_json=False))
        else:
            print(f"An unexpected runtime error occurred: {e}")
    except Exception as e:
         print(f"An critical error occurred in the main execution: {e}")
         import traceback
         traceback.print_exc()
