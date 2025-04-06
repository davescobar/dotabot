import os
import psycopg2
import json
from typing import TypedDict, Annotated, Any, List
from langgraph.graph import StateGraph
from dotenv import load_dotenv

class DotaItemSearchGraph:
    def __init__(self):
        self.compiled = self._build_graph()
        load_dotenv()
        self.database_url = os.getenv("KORVUS_DATABASE_URL")

    def extract_search_filters(self, state):
        question = state["question"]

        system_prompt = {
            "role": "system",
            "content": (
                "You are a Dota 2 assistant that extracts user intent and structured search filters.\n"
                "Return a JSON with the keys: 'intent' and 'filters'.\n"
                "Intent can be one of: retrieve, compare, identify, recommend, explain, combo.\n"
                "Filters can include: tags, effects, components, items, hero_targets, stats.\n"
                "Example: {\"intent\": \"compare\", \"filters\": {\"items\": [\"Orchid Malevolence\", \"Bloodthorn\"]}}"
            )
        }

        user_prompt = {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                "Return format: {\"intent\": ..., \"filters\": { ... }}"
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

        with psycopg2.connect(self.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (json.dumps(system_prompt), json.dumps(user_prompt)))
                result = cur.fetchone()[0]

        try:
            parsed = json.loads(result)
            intent = parsed.get("intent", "retrieve")
            filters = parsed.get("filters", {})
        except Exception:
            intent = "retrieve"
            filters = {"tags": [], "effects": [], "components": []}  # fallback

        print("[LOG] Extracted intent:", intent)
        print("[LOG] Extracted filters:", filters)

        return {"intent": intent, "filters": filters, "question": question}

    def plan_query(self, state):
        filters = state["filters"]

        clauses = []
        if filters["tags"]:
            tag_query = " & ".join(filters["tags"])
            clauses.append(f"search_tags @@ plainto_tsquery('{tag_query}')")
        if filters["effects"]:
            effect_query = " | ".join(filters["effects"])
            clauses.append(f"search_effects @@ plainto_tsquery('{effect_query}')")
        if filters["components"]:
            component_query = " | ".join(filters["components"])
            clauses.append(f"search_components @@ plainto_tsquery('{component_query}')")

        tsv_query = (
            "SELECT id, title, abstract, 0.95 AS score "
            "FROM dota_items "
            f"WHERE {' AND '.join(clauses)} "
            "LIMIT 10"
        )

        print("[LOG] Generated TSVECTOR SQL:", tsv_query)
        return {"tsv_query": tsv_query, "question": state["question"]}

    def run_combined_query(self, state):
        question = state["question"]
        tsv_query = state["tsv_query"]

        embedding_query = """
            SELECT id, title, abstract, 1 - (embedding <=> pgml.embed('mixedbread-ai/mxbai-embed-large-v1', %s)) AS score
            FROM dota_items
            ORDER BY embedding <=> pgml.embed('mixedbread-ai/mxbai-embed-large-v1', %s)
            LIMIT 10;
        """

        combined_results = []

        with psycopg2.connect(os.getenv("KORVUS_DATABASE_URL")) as conn:
            with conn.cursor() as cur:
                cur.execute(tsv_query)
                combined_results += cur.fetchall()

                cur.execute(embedding_query, (question, question))
                combined_results += cur.fetchall()

        # Merge and deduplicate by ID with highest score
        seen = {}
        for row in combined_results:
            id_, title, abstract, score = row
            if id_ not in seen or score > seen[id_]["score"]:
                seen[id_] = {"id": id_, "title": title, "abstract": abstract, "score": score}

        final = sorted(seen.values(), key=lambda x: -x["score"])
        print(f"[LOG] Combined ranked results: {len(final)}")
        return {"results": final[:10], "question": question}

    def summarize_results(self, state):
        results = state["results"]
        question = state["question"]

        if not results:
            return {"final_answer": "No items found matching your query."}

        items_text = "\n".join([f"- {r['title']}: {r['abstract']}" for r in results])

        system_prompt = {
            "role": "system",
            "content": "You are a Dota 2 expert. Summarize the results of a query in a helpful, concise list."
        }
        user_prompt = {
            "role": "user",
            "content": f"User asked: {question}\n\nMatching items:\n{items_text}\n\nSummarize the items and why they are relevant."
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

        with psycopg2.connect(os.getenv("KORVUS_DATABASE_URL")) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, (json.dumps(system_prompt), json.dumps(user_prompt)))
                result = cur.fetchone()[0]

        print("[LOG] Final LLM summary generated")
        return {"final_answer": result.strip()}

    def _build_graph(self):
        class State(TypedDict):
            question: str
            filters: Annotated[Any, None]
            tsv_query: Annotated[str, None]
            results: Annotated[List[Any], None]
            final_answer: Annotated[str, None]

        builder = StateGraph(state_schema=State)
        builder.add_node("extract_filters", self.extract_search_filters)
        builder.add_node("plan_query", self.plan_query)
        builder.add_node("run_combined_query", self.run_combined_query)
        builder.add_node("summarize_results", self.summarize_results)

        builder.set_entry_point("extract_filters")
        builder.add_edge("extract_filters", "plan_query")
        builder.add_edge("plan_query", "run_combined_query")
        builder.add_edge("run_combined_query", "summarize_results")
        builder.set_finish_point("summarize_results")

        return builder.compile()

    def run(self, question):
        return self.compiled.invoke({"question": question})
