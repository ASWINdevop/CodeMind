import os
import pickle
import networkx as nx
from src.config import REPOS_DIR, GRAPH_FILE_PATH

def calculate_metrics():
    print("\n" + "="*40)
    print(" 🧠 CODEMIND: RESUME METRICS ENGINE")
    print("="*40 + "\n")

    # 1. Calculate File Scale
    total_files = 0
    if os.path.exists(REPOS_DIR):
        for root, dirs, files in os.walk(REPOS_DIR):
            for file in files:
                if file.endswith(('.py', '.md')): 
                    total_files += 1
        print(f"✅ Total Ingested Files: {total_files}")
    else:
        print("❌ Error: REPOS_DIR not found. Have you cloned anything?")

    # 2. Calculate Topological Graph Scale
    if os.path.exists(GRAPH_FILE_PATH):
        try:
            # Open as Read-Binary ("rb") and load the pickle object directly
            with open(GRAPH_FILE_PATH, "rb") as f:
                graph = pickle.load(f)
                
                num_nodes = len(graph.nodes)
                num_edges = len(graph.edges)
                
                print(f"✅ Total Graph Nodes (Functions/Classes): {num_nodes}")
                print(f"✅ Total Graph Edges (Execution Paths): {num_edges}")
        except Exception as e:
            print(f"❌ Error loading graph: {e}")
    else:
        print("❌ Error: graph.json not found. Have you clicked 'Ingest & Embed'?")

    print("\n" + "="*40 + "\n")

if __name__ == "__main__":
    calculate_metrics()