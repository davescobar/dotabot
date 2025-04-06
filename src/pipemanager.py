import korvus
import json
import asyncio
import os
from dotenv import load_dotenv
import psycopg2

# Initialize logging for the korvus module
korvus.init_logger(level="INFO", format="%(asctime)s - %(levelname)s - %(message)s")

class PipelineManager:
    def __init__(self, collection):
        load_dotenv()  # Load environment variables from .env
        self.db_url = os.getenv("KORVUS_DATABASE_URL")
        if not self.db_url:
            raise ValueError("Database URL not found in .env file.")
        self.collection = korvus.Collection(collection, self.db_url)

    async def add_pipeline(self, pipeline):
        """Add a new pipeline to the collection, using default schema if not provided."""
      
        # Add the pipeline to the collection
        await self.collection.add_pipeline(pipeline)
        print(f"Added pipeline to collection.")

    async def update_pipeline(self, pipeline_name, new_schema):
        """Update an existing pipeline with a new schema."""
        pipeline = korvus.Pipeline(pipeline_name)
        await self.collection.remove_pipeline(pipeline)  # Remove the old pipeline
        print(f"Pipeline '{pipeline_name}' removed.")
        await self.add_pipeline(pipeline_name, new_schema)  # Add the new pipeline

    async def disable_pipeline(self, pipeline_name):
        """Disable an existing pipeline."""
        pipeline = korvus.Pipeline(pipeline_name)
        await self.collection.disable_pipeline(pipeline)
        print(f"Pipeline '{pipeline_name}' disabled.")

    async def enable_pipeline(self, pipeline_name):
        """Enable a previously disabled pipeline."""
        pipeline = korvus.Pipeline(pipeline_name)
        await self.collection.enable_pipeline(pipeline)
        print(f"Pipeline '{pipeline_name}' enabled.")

    async def remove_pipeline(self, pipeline_name):
        """Remove a pipeline and its data from the collection."""
        pipeline = korvus.Pipeline(pipeline_name)
        await self.collection.remove_pipeline(pipeline)
        print(f"Pipeline '{pipeline_name}' removed.")

    async def get_pipeline_schema(self, pipeline_name=None):
        """Query the database to get the schema of a specific pipeline or all pipelines if no name is provided."""
        try:
            # Connect to the database
            conn = psycopg2.connect(self.db_url)
            cursor = conn.cursor()

            if pipeline_name:
                # Query for a specific pipeline
                query = f"SELECT schema FROM dota_collection.pipelines WHERE name = %s;"
                cursor.execute(query, (pipeline_name,))
                result = cursor.fetchone()

                if result:
                    print(f"Schema for '{pipeline_name}':")
                    print(json.dumps(result[0], indent=2))
                else:
                    print(f"No schema found for pipeline '{pipeline_name}'.")
            else:
                # Query for all pipelines
                query = f"SELECT name, schema FROM dota_collection.pipelines;"
                cursor.execute(query)
                results = cursor.fetchall()

                if results:
                    print("Schemas for all pipelines:")
                    for name, schema in results:
                        print(f"Pipeline '{name}':")
                        print(json.dumps(schema, indent=2))
                else:
                    print("No pipelines found.")

        except Exception as e:
            print(f"Error querying schema: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_hero_pipeline_schema(self):
        """Return the default schema for a given pipeline."""
        # Default schemas for heroes, items, and all item summaries
        return korvus.Pipeline("heroesv2", {
            "text": {
                "splitter": {"model": "recursive_character"},
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                    "parameters": {"filterable": ["type", "tags"]}
                },
                "full_text_search": {"configuration": "english"}
            },
            "metadata": {
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                    "parameters": {"filterable": ["type", "tags"]}
                },
                "full_text_search": {"configuration": "english"}
            }
        })

    def get_items_pipeline_schema(self, pipeline):
        return korvus.Pipeline("itemsv2", {
            "text": {
                "splitter": {"model": "recursive_character"},
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                },
                "full_text_search": {"configuration": "english"}
            },
            "metadata": {
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                    "parameters": {"filterable": ["type", "tags"]}
                },
                "full_text_search": {"configuration": "english"}
            }
        })

    def get_item_summaries_pipeline_schema(self):
        return korvus.Pipeline("all_item_summariesv2", {
            "text": {
                "splitter": {"model": "recursive_character"},
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                },
            },
        })
    
    def get_rag_pipeline(self):
        return korvus.Pipeline("ragv2", {
            "text": {
                "splitter": {"model": "recursive_character"},
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                },
                "full_text_search": {"configuration": "english"}
            },
            "metadata": {
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                    "parameters": {"filterable": ["type", "tags"]}
                },
                "full_text_search": {"configuration": "english"}
            }
        })

# Example usage
if __name__ == "__main__":
    async def asyncmain():
        pipeline = korvus.Pipeline("itemsv2", {
            "text": {
                "splitter": {"model": "recursive_character"},
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                },
                "full_text_search": {"configuration": "english"}
            },
            "metadata": {
                "semantic_search": {
                    "model": "Alibaba-NLP/gte-base-en-v1.5",
                    "parameters": {"filterable": ["type", "tags"]}
                },
                "full_text_search": {"configuration": "english"}
            }
        })
        # pipeline = korvus.Pipeline(
        #     "test_pipeline",
        #     {
        #         "body": {
        #             "splitter": {"model": "recursive_character"},
        #             "semantic_search": {"model": "Alibaba-NLP/gte-base-en-v1.5"},
        #         },
        #     },
        # )

        load_dotenv()  # Load environment variables from .env
        db_url = os.getenv("KORVUS_DATABASE_URL")
        collection =  korvus.Collection("dota_collection", db_url)
        await collection.add_pipeline(pipeline)  # Add the pipeline to the collecti
    # pipemanager = PipelineManager("dota_collection")
    
    # # Adding predefined pipelines for heroes, items, and all item summaries
    # asyncio.run(pipemanager.disable_pipeline("all_item_summariesv1"))
    # asyncio.run(pipemanager.disable_pipeline("itemsv1"))
    # asyncio.run(pipemanager.disable_pipeline("heroesv1"))
    # asyncio.run(pipemanager.add_pipeline(pipemanager.get_hero_pipeline_schema()))
    # asyncio.run(pipemanager.get_items_pipeline_schema(pipemanager))
    # asyncio.run(pipemanager.add_pipeline(pipemanager.get_item_summaries_pipeline_schema()))
    # asyncio.run(pipemanager.add_pipeline(pipemanager.get_rag_pipeline()))
    asyncio.run(asyncmain())  # Run the main function to execute the pipeline manager operations