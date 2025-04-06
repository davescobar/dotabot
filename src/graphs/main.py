import os
import psycopg2
import json
from langgraph.graph import StateGraph

from items.item_search_graph import DotaItemSearchGraph

class DotaLangGraph:
    def __init__(self):
        self.item_graph = DotaItemSearchGraph()

    def run(self, question: str):
        # In future: route to hero/ability/mechanic graphs based on classification
        return self.item_graph.run(question)

# Optional entry point for quick CLI/dev testing
if __name__ == "__main__":
    graph = DotaLangGraph()
    result = graph.run("What are good early-game escape items that silence?")
    print("\n[ANSWER]\n", result.get("final_answer"))
