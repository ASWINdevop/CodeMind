import os
import pickle
import chromadb
from chromadb.api.types import EmbeddingFunction
from sentence_transformers import SentenceTransformer
from src.config import (
    CHROMA_DB_DIR, 
    CHROMA_COLLECTION_NAME, 
    EMBEDDING_MODEL_NAME, 
    GRAPH_FILE_PATH, 
    REPOS_DIR
)
class NomicEmbeddingFunction(EmbeddingFunction):
    """Custom wrapper to securely load code-native HF models."""
    def __init__(self, model_name):
        # trust_remote_code=True is mathematically required for Nomic architectures
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        
    def __call__(self, input: list[str]) -> list[list[float]]:
        # Prefix required by Nomic for document retrieval
        prefixed_inputs = [f"search_document: {text}" for text in input]
        return self.model.encode(prefixed_inputs).tolist()

class CodeVectorStore:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
        
        # Inject the custom Code-Native Embedding Layer
        self.embedding_fn = NomicEmbeddingFunction(EMBEDDING_MODEL_NAME)
        
        self.collection = self.chroma_client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            embedding_function=self.embedding_fn
        )

    def populate_from_graph(self, graph_path=GRAPH_FILE_PATH, target_repos=None):
        """Reads the NetworkX graph and embeds source code for specified repositories."""
        if not os.path.exists(graph_path):
            print(f"Error: Graph file missing at {graph_path}. Run indexer.py first.")
            return

        with open(graph_path, 'rb') as f:
            graph = pickle.load(f)

        documents = []
        metadatas = []
        ids = []


        for node_id, data in graph.nodes(data=True):
            if data.get('type') == 'function':
                repo_name = data.get('repo')
                
                # REPOSITORY FILTER LOGIC
                if target_repos is not None and repo_name not in target_repos:
                    continue
                
                parts = node_id.split('::')
                if len(parts) < 3:
                    continue 
                
                rel_path = parts[1]
                func_name = parts[2]
                filepath = os.path.join(REPOS_DIR, repo_name, rel_path)
                
                start_byte = data.get('metadata', {}).get('start_byte')
                end_byte = data.get('metadata', {}).get('end_byte')

                if start_byte is None or end_byte is None or not os.path.exists(filepath):
                    continue

                try:
                    with open(filepath, 'rb') as f:
                        file_bytes = f.read()
                        func_code = file_bytes[start_byte:end_byte].decode('utf-8')

                    documents.append(func_code)
                    metadatas.append({
                        "repo": repo_name,
                        "filepath": rel_path,
                        "function": func_name,
                        "node_id": node_id
                    })
                    ids.append(node_id)
                    
                except Exception as e:
                    pass

        if not ids:
            print("No functions found to embed for the specified criteria.")
            return

        batch_size = 500
        total_added = 0
        for i in range(0, len(ids), batch_size):
            self.collection.upsert(
                documents=documents[i:i+batch_size],
                metadatas=metadatas[i:i+batch_size],
                ids=ids[i:i+batch_size]
            )
            total_added += len(ids[i:i+batch_size])
        
        print(f"Successfully vectorized and stored {total_added} functions in ChromaDB.")

    def search(self, query: str, n_results: int = 5, repo_filter: str = None, max_initial_k: int = 20, max_char_budget: int = 25000, **kwargs):
        """
        Executes an adaptive k-retrieval while maintaining backwards compatibility 
        with the native ChromaDB dictionary return format.
        """
        where_clause = {"repo": repo_filter} if repo_filter else None
        
        # 1. Over-fetch the initial candidate pool (ignoring the legacy n_results limit)
        raw_results = self.collection.query(
            query_texts=[query],
            n_results=max_initial_k,
            where=where_clause
        )
        
        if not raw_results['ids'] or not raw_results['ids'][0]:
            return {'ids': [[]], 'documents': [[]], 'distances': [[]]}

        ids = raw_results['ids'][0]
        documents = raw_results['documents'][0]
        distances = raw_results['distances'][0]
        
        final_ids = []
        final_docs = []
        final_dists = []
        
        current_budget = 0
        baseline_distance = distances[0] 

        # 2. The Dynamic Optimization Funnel
        for i in range(len(distances)):
            current_id = ids[i]
            current_doc = documents[i]
            current_distance = distances[i]
            
            # RULE A: The Distance Cliff (Entropy Filter)
            distance_delta = current_distance - baseline_distance
            if i > 0 and distance_delta > 0.45: 
                print(f"[Adaptive Optimization] Cutoff reached at k={i} due to semantic cliff (Delta: {distance_delta:.2f})")
                break
                
            # RULE B: The Context Budget (Token Filter)
            doc_len = len(current_doc)
            if current_budget + doc_len > max_char_budget:
                print(f"[Adaptive Optimization] Cutoff reached at k={i} due to token budget limits.")
                break
                
            current_budget += doc_len
            final_ids.append(current_id)
            final_docs.append(current_doc)
            final_dists.append(current_distance)
            
        # 3. Return the exact structure rag.py expects
        return {
            'ids': [final_ids],
            'documents': [final_docs],
            'distances': [final_dists]
        }

if __name__ == "__main__":
    store = CodeVectorStore()
    # Example usage: store.populate_from_graph(target_repos=["LexAI", "VeritasAI"])
    store.populate_from_graph()