import korvus
from dotenv import load_dotenv
import os
import asyncio
import json

load_dotenv()
database_url = os.getenv("KORVUS_DATABASE_URL")

class RAGHandler:
    def __init__(self, collection_name="dota_collection"):
        self.pipeline = korvus.Pipeline("v0", {
            "text": {
                "splitter": {
                    "model": "recursive_character",
                    "parameters": {"chunk_size": 4096, "chunk_overlap": 0}
                },
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                    "parameters": {
                        "filterable": ["type", "tags"]
                    }
                },
                "full_text_search": {
                    "configuration": "english"
                }
            },
            "metadata": {
                "semantic_search": {
                        "model": "Alibaba-NLP/gte-base-en-v1.5",
                        "parameters": {
                            "filterable": ["type", "tags"]
                        }
                    },
                    "full_text_search": {
                        "configuration": "english"
                    }
                }
        })
        self.collection = korvus.Collection(collection_name, database_url)
        self.infer_collection = korvus.Collection("dota_type_examples", database_url)

        self.type_examples = [
            {"type": "item", "text": "best items to buy in dota"},
            {"type": "hero", "text": "strongest heroes in current meta"},
            {"type": "patch", "text": "recent patch changes"},
            {"type": "mechanic", "text": "how magic immunity works"},
            {"type": "item_summary", "text": "overview of item roles"}
        ]

    async def setup(self):
        await self.collection.add_pipeline(self.pipeline)  # ✅ Register pipeline first
        await self.infer_collection.add_pipeline(self.pipeline)


    async def infer_doc_type_semantic(self, query):
        docs = [
            {"id": f"type_example_{i}", "text": ex["text"], "metadata": {"type": ex["type"]}}
            for i, ex in enumerate(self.type_examples)
        ]
        await self.infer_collection.upsert_documents(docs)

        results = await self.infer_collection.query() \
            .vector_recall(query, self.pipeline) \
            .limit(1) \
            .fetch_all()

        if results:
            return results[0][2]["metadata"].get("type")
        return None

    async def run_query(self, query):
        # Ensure setup has run (adds the pipeline if not already done)
        await self.setup()

        # Use the pipeline object stored in the instance
        pipeline_to_use = await self.collection.get_pipeline("v3")  
        doc_type = await self.infer_doc_type_semantic(query)

        # Perform RAG
        results = await self.collection.rag(
            {
                "CONTEXT": {
                    "vector_search": {
                        "query": {
                            "fields": {
                                "text": {
                                    "query": query,  # Main search query
                                },
                                "metadata": {  # Filters based on doc_type
                                    "query": "item",  # Searching by the doc_type (tags can also be added here)
                                    "parameters": {
                                        "prompt": "Represent this sentence for searching relevant passages: ",
                                    },
                                }
                            },
                        },
                        "document": {"keys": ["text", "id", "metadata"]},  # Specify which fields to retrieve
                    },
                    "aggregate": {"join": "\n\n---\n\n"},  # Join contexts clearly
                },
                "chat": {
                    "model": "meta-llama/Meta-Llama-3.1-405B-Instruct7",
                    "messages": [
                        {"role": "system", "content": "You are a helpful Dota 2 assistant. Use only the provided context to answer the question. If the context doesn't contain the answer, say so."},
                        {"role": "user", "content": f"Using the provided context, answer the question: {query}"},
                    ],
                    "max_tokens": 250,  # Increased slightly
                },
            },
            pipeline_to_use,  # Pass the pipeline object
        )
        return results


if __name__ == "__main__":
    query = "what is the best late game dota item"
    handler = RAGHandler()
    result = asyncio.run(handler.run_query(query))
    print("\n=== RAG Results ===")
    print(json.dumps(result, indent=2))
801