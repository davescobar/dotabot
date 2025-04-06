# langgraph_ingest_pgml.py
import os
import asyncio
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import Json
from langgraph.graph import StateGraph
from typing import Any, List, TypedDict
from typing_extensions import Annotated
import json
load_dotenv()
database_url = os.getenv("KORVUS_DATABASE_URL")

def build_langgraph_query():
    class State(TypedDict):
        input: str
        embedding: Annotated[Any, None]  # Vector, use Any unless you define a custom type
        results: Annotated[List[Any], None]
        response: str
        answer: str

    builder = StateGraph(state_schema=State)
    builder.add_node("embed_query", embed_query_node)
    builder.add_node("vector_search", vector_search_node)
    builder.add_node("format_result", format_result_node)
    builder.add_node("question_answer", question_answer_node)

    builder.set_entry_point("embed_query")
    builder.add_edge("embed_query", "vector_search")
    builder.add_edge("vector_search", "format_result")
    builder.add_edge("vector_search", "question_answer")
    builder.set_finish_point("question_answer")

    return builder.compile()

def upsert_pgml_documents(docs, table_name):
    sql_create = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        id TEXT PRIMARY KEY,
        body TEXT,
        summary TEXT,
        metadata JSONB,
        embedding VECTOR(1024) GENERATED ALWAYS AS (
            pgml.normalize_l2(pgml.embed('mixedbread-ai/mxbai-embed-large-v1', body))
        ) STORED,
        summary_embedding VECTOR(1024) GENERATED ALWAYS AS (
            pgml.normalize_l2(pgml.embed('mixedbread-ai/mxbai-embed-large-v1', summary))
        ) STORED,
        title_and_body_text TSVECTOR GENERATED ALWAYS AS (
            to_tsvector('english', body || ' ' || summary)
        ) STORED
    );
    """
    sql_index = f"""
    CREATE INDEX IF NOT EXISTS {table_name}_tsvector_index
    ON {table_name} USING GIN (title_and_body_text);
    """
    sql_upsert = f"""
    INSERT INTO {table_name} (id, body, summary, metadata)
    VALUES (%s, %s, %s, %s)
    ON CONFLICT (id) DO UPDATE SET
        body = EXCLUDED.body,
        summary = EXCLUDED.summary,
        metadata = EXCLUDED.metadata;
    """
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql_create)
            cur.execute(sql_index)
            for doc in docs:
                cur.execute(sql_upsert, (
                    doc.get("id"),
                    doc.get("body"),
                    doc.get("summary", ""),
                    Json(doc.get("metadata", {}))
                ))
            conn.commit()

def format_pgml_documents(raw_docs):
    return [
        {
            "id": doc.get("id"),
            "body": doc.get("text"),
            "summary": doc.get("summary", ""),
            "metadata": doc.get("metadata", {})
        }
        for doc in raw_docs
    ]

def embed_query_node(state):
    query = state.get("input", "")
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT pgml.embed('mixedbread-ai/mxbai-embed-large-v1', %s)::vector(1024)", (query,))
            embedding = cur.fetchone()[0]
    return {"embedding": embedding, "input": query}

def vector_search_node(state):
    embedding = state["embedding"]
    sql = """
        SELECT title, text, abstract, metadata
        FROM dota_items
        ORDER BY embedding <=> %s::vector(1024)
        LIMIT 3;
    """
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (embedding,))
            results = cur.fetchall()
    return {"results": results, "input": state.get("input", "")}

def format_result_node(state):
    input_query = state.get("input", "")
    result_lines = [
        f"Result {i+1}:\nBody: {row[1]}\nSummary: {row[2]}\n"
        for i, row in enumerate(state.get("results", []))
    ]
    return {"response": f"Top matches for query '{input_query}':\n\n" + "\n".join(result_lines)}

def question_answer_node(state):
    question = state.get("input", "")
    context = "\n\n".join(
        f"ID: {row[0]}\nBody: {row[1]}\nSummary: {row[2]}\nMetadata: {row[3]}"
        for row in state["results"]
    )

    system_prompt = {
        "role": "system",
        "content": (
            "You are a Dota 2 subject matter expert. You only answer questions strictly related to Dota 2, "
            "such as heroes, items, gameplay mechanics, patches, strategies, or tournaments. "
            "You must not engage in or respond to any queries that are not directly related to Dota 2. "
            "If a user asks about anything outside of Dota 2, respond with: 'I'm only here to answer questions about Dota 2.'"
        )
    }

    user_input = {
        "role": "user",
        "content": f"Question: {question}\n\nContext: {context}"
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

    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (json.dumps(system_prompt), json.dumps(user_input)))
            answer = cur.fetchone()[0]

    return {"answer": answer, "input": question, "results": state.get("results", [])}

async def ingest_items_with_pgml():
    # dota = DotaItemAPI()
    # items = dota.fetch_item_list()
    # raw_docs, _ = dota.format_item_documents(items)
    # documents = format_pgml_documents(raw_docs)
    # upsert_pgml_documents(documents, "dota_items_pgml")
    # print(f"Ingested {len(documents)} item documents using pgml.embed().")

    graph = build_langgraph_query()
    user_input = input("Enter your question about Dota 2: ")
    initial_state = {"input": user_input}
    result = graph.invoke(initial_state)
    print(result["answer"])

if __name__ == "__main__":
    asyncio.run(ingest_items_with_pgml())
