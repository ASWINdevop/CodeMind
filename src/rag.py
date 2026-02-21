import os
import pickle
import networkx as nx
from google import genai
from src.config import GOOGLE_API_KEY, GEMINI_MODEL_NAME, GRAPH_FILE_PATH
from src.vector_store import CodeVectorStore

class CodeMindAgent:
    def __init__(self):
        import os
        import json
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self.vector_store = CodeVectorStore()
        
        if not os.path.exists(GRAPH_FILE_PATH):
            raise FileNotFoundError(f"Missing graph file at {GRAPH_FILE_PATH}. Run indexer first.")
        with open(GRAPH_FILE_PATH, 'rb') as f:
            self.graph = pickle.load(f)
            
        # --- CACHE INITIALIZATION ---
        self.cache_path = os.path.join("data", "query_cache.json")
        self.cache = self._load_cache()

    def _load_cache(self):
        import json
        import os
        if os.path.exists(self.cache_path):
            with open(self.cache_path, 'r') as f:
                return json.load(f)
        return {}

    def _save_cache(self):
        import json
        import os
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        with open(self.cache_path, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def retrieve_context(self, query: str, top_k: int = 3, repo_filter: str = None):
        """Retrieves vectors, maps topology, and tracks involved repositories and raw code."""
        results = self.vector_store.search(query, n_results=top_k, repo_filter=repo_filter)
        
        if not results['ids'] or not results['ids'][0]:
            return "No relevant codebase context found.", "", set(), {}
            
        target_nodes = results['ids'][0]
        context_parts = []
        topology_map = set()
        repos_involved = set()
        context_dict = {} # NEW: Dictionary to hold raw code for the UI Citation Viewer
        
        for node_id in target_nodes:
            if node_id not in self.graph: continue

            repo_name = self.graph.nodes[node_id].get('repo')
            if repo_name: repos_involved.add(repo_name)

            predecessors = list(self.graph.predecessors(node_id))
            file_nodes = [p for p in predecessors if self.graph.nodes[p].get('type') == 'file']
            
            for file_node in file_nodes:
                imports = [v.split('::')[-1] for u, v, data in self.graph.out_edges(file_node, data=True) if data.get('relationship') == 'DEPENDS_ON']
                siblings = list(self.graph.successors(file_node))
                func_names = [s.split('::')[-1] for s in siblings if self.graph.nodes[s].get('type') == 'function']
                
                blueprint = f"FILE: {file_node}\n"
                if imports: blueprint += f"IMPORTS:\n" + "\n".join([f" - {imp}" for imp in imports]) + "\n"
                blueprint += f"CONTAINS COMPONENTS:\n" + "\n".join([f" - {fn}" for fn in func_names])
                topology_map.add(blueprint)

                sibling_ids = [s for s in siblings if self.graph.nodes[s].get('type') == 'function']
                if sibling_ids:
                    context_parts.append(f"--- SOURCE CODE FOR FILE ({file_node}) ---")
                    sib_res = self.vector_store.collection.get(ids=sibling_ids) 
                    for sib_id, sib_doc in zip(sib_res['ids'], sib_res['documents']):
                        context_parts.append(f"Component {sib_id}:\n{sib_doc}\n")
                        context_dict[sib_id] = sib_doc # Store raw code for UI

            outgoing_calls = [v.split('::')[-1] for u, v, data in self.graph.out_edges(node_id, data=True) if data.get('relationship') == 'CALLS']
            incoming_calls = [u.split('::')[-1] for u, v, data in self.graph.in_edges(node_id, data=True) if data.get('relationship') == 'CALLS']
            
            if outgoing_calls or incoming_calls:
                flow_map = f"EXECUTION FLOW FOR: {node_id}\n"
                if incoming_calls: flow_map += f"CALLED BY:\n" + "\n".join([f" - {caller}" for caller in incoming_calls]) + "\n"
                if outgoing_calls: flow_map += f"MAKES CALLS TO:\n" + "\n".join([f" - {callee}" for callee in outgoing_calls]) + "\n"
                topology_map.add(flow_map)
                                
        return "\n\n".join(list(topology_map)), "\n".join(context_parts), repos_involved, context_dict

    def read_file_content(self, file_node_id: str) -> str:
        """TOOL: Retrieves raw source code directly from the local disk."""
        import os
        from src.config import REPOS_DIR
        try:
            # Graph IDs are mapped as "repo_name::path/to/file.py"
            parts = file_node_id.split("::", 1)
            if len(parts) != 2:
                return "Error: Invalid file ID format."
                
            repo_name, file_path = parts
            full_path = os.path.join(REPOS_DIR, repo_name, file_path)
            
            if os.path.exists(full_path):
                with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            return f"Error: File {full_path} not found on disk."
        except Exception as e:
            return f"Error executing READ_FILE: {e}"

    def ask(self, query: str, repo_filter: str = None, use_self_healing: bool = False):
        """Assembles the prompt and executes an autonomous ReAct loop with asymmetric caching."""
        import re
        import time 

        # --- 1. ASYMMETRIC CACHE INTERCEPTION ---
        query_key = f"{query.strip().lower()}::{repo_filter}"
        
        if query_key in self.cache:
            cached_data = self.cache[query_key]
            cached_is_healed = cached_data.get("is_self_healed", False)
            
            # Asymmetric Logic Matrix
            can_use_cache = False
            if use_self_healing and cached_is_healed:
                can_use_cache = True # Healed query -> Healed Cache exists
            elif not use_self_healing:
                can_use_cache = True # Standard query -> Accepts any cache (Healed or Unhealed)
                
            if can_use_cache:
                print(f"\n[Cache Hit] Returning 0s latency response (Healed: {cached_is_healed})")
                mod_tel = cached_data.get("telemetry", {}).copy()
                mod_tel["total_latency_sec"] = 0.0
                mod_tel["retrieval_latency_sec"] = 0.0
                mod_tel["llm_reasoning_sec"] = 0.0
                mod_tel["prompt_tokens"] = 0
                mod_tel["completion_tokens"] = 0
                
                return {
                    "answer": cached_data['answer'],  # Reverted to raw answer
                    "references": cached_data["references"],
                    "telemetry": mod_tel,
                    "cache_status": "healed" if cached_is_healed else "standard" # NEW: Structural UI Flag
                }

        telemetry = {
            "retrieval_latency_sec": 0.0,
            "llm_reasoning_sec": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_latency_sec": 0.0,
            "react_iterations": 0
        }
        
        t_start_total = time.perf_counter()

        # MEASURE RETRIEVAL LATENCY (Vector + Graph Expansion)
        t0_retrieval = time.perf_counter()
        topology, context, repos_involved, context_dict = self.retrieve_context(query, repo_filter=repo_filter)
        telemetry["retrieval_latency_sec"] = round(time.perf_counter() - t0_retrieval, 3)
        
        base_docs = ""
        for repo in repos_involved:
            readme_id = f"{repo}::readme"
            if readme_id in self.graph:
                readme_text = self.graph.nodes[readme_id].get('content', '')[:3000] 
                base_docs += f"--- README FOR {repo} ---\n{readme_text}\n\n"
                context_dict[readme_id] = readme_text

        prompt = f"""You are an expert Senior Software Architect with autonomous reasoning capabilities.
You operate on a ReAct (Reason + Act) loop. You do not guess; you investigate.

### SYSTEM DOCUMENTATION
{base_docs if base_docs else "No system documentation available."}

### ARCHITECTURAL MAP (Dependencies & Call Graphs)
{topology}

### INITIAL CODEBASE CONTEXT (Vector Search Hits)
{context}

### YOUR TOOL: READ_FILE
If the Architectural Map shows a file that you need to understand, but its source code is missing from the Initial Context, you MUST fetch it.
To use the tool, you must reply with EXACTLY this syntax and stop reasoning:
[ACTION: READ_FILE | INPUT: repo_name::path/to/file.py]

CRITICAL CITATION RULE: Whenever you use information from a Component or README, you MUST cite it immediately using the exact component ID:.

The User's Query: {query}

Execute this reasoning chain:
1. Analyze the Map and Context.
2. If you lack critical source code, output the ACTION string to fetch the file.
3. If you have enough information, output your Final Answer."""

        max_steps = 5 
        current_prompt = prompt
        
        for step in range(max_steps):
            telemetry["react_iterations"] += 1
            
            # MEASURE LLM GENERATION LATENCY
            t0_llm = time.perf_counter()
            response = self.client.models.generate_content(
                model=GEMINI_MODEL_NAME,
                contents=current_prompt,
            )
            telemetry["llm_reasoning_sec"] += (time.perf_counter() - t0_llm)
            
            # EXTRACT TOKEN USAGE
            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                telemetry["prompt_tokens"] += getattr(response.usage_metadata, 'prompt_token_count', 0)
                telemetry["completion_tokens"] += getattr(response.usage_metadata, 'candidates_token_count', 0)

            reply = response.text
            
            action_match = re.search(r'\[ACTION:\s*READ_FILE\s*\|\s*INPUT:\s*(.+?)\]', reply)
            
            if action_match:
                file_id = action_match.group(1).strip()
                print(f"\n[ReAct Triggered]: LLM requested file -> {file_id}") 
                observation = self.read_file_content(file_id)
                context_dict[file_id] = observation
                current_prompt += f"\n\n{reply}\n\n[OBSERVATION from {file_id}]:\n{observation}\n\nNow continue your analysis."
            else:
                # --- FAST-PATH BYPASS ---
                if not use_self_healing:
                    telemetry["llm_reasoning_sec"] = round(telemetry["llm_reasoning_sec"], 3)
                    telemetry["total_latency_sec"] = round(time.perf_counter() - t_start_total, 3)
                    
                    # Caching logic: Do not overwrite a healed cache with an unhealed one
                    if query_key not in self.cache or not self.cache[query_key].get("is_self_healed"):
                        self.cache[query_key] = {"answer": reply, "references": context_dict, "telemetry": telemetry, "is_self_healed": False}
                        self._save_cache()
                        
                    return {"answer": reply, "references": context_dict, "telemetry": telemetry}

                # --- EVALUATOR PASS (SELF-HEALING) ---
                print("\n[Self-Healing]: Evaluating proposed answer...")
                eval_prompt = f"""You are a strict QA Evaluator.
Analyze if the Proposed Answer is 100% factual, answers the User Query, and strictly relies on the Context provided below.

### RAW CODEBASE CONTEXT
{context}

### USER QUERY
{query}

### PROPOSED ANSWER
{reply}

Rule 1: If the Proposed Answer is complete, accurate according to the RAW CODEBASE CONTEXT, and contains NO hallucinations, output EXACTLY: [EVAL: PASS]
Rule 2: If the answer hallucinates variables not in the context, fails to answer the prompt, or needs to read more files, output EXACTLY: [EVAL: FAIL | CRITIQUE: <brief reason>]"""
                
                t0_eval = time.perf_counter()
                eval_response = self.client.models.generate_content(
                    model=GEMINI_MODEL_NAME,
                    contents=eval_prompt,
                )
                telemetry["llm_reasoning_sec"] += (time.perf_counter() - t0_eval)
                
                if hasattr(eval_response, 'usage_metadata') and eval_response.usage_metadata:
                    telemetry["prompt_tokens"] += getattr(eval_response.usage_metadata, 'prompt_token_count', 0)
                    telemetry["completion_tokens"] += getattr(eval_response.usage_metadata, 'candidates_token_count', 0)
                
                eval_text = eval_response.text
                
                if "[EVAL: PASS]" in eval_text:
                    print("[Self-Healing]: Answer PASSED.")
                    telemetry["llm_reasoning_sec"] = round(telemetry["llm_reasoning_sec"], 3)
                    telemetry["total_latency_sec"] = round(time.perf_counter() - t_start_total, 3)
                    
                    # Overwrite any existing cache with this high-tier healed answer
                    self.cache[query_key] = {"answer": reply, "references": context_dict, "telemetry": telemetry, "is_self_healed": True}
                    self._save_cache()
                    
                    return {"answer": reply, "references": context_dict, "telemetry": telemetry}
                else:
                    print(f"[Self-Healing]: Answer FAILED. Forcing retry. Critique: {eval_text}")
                    current_prompt += f"\n\n[SYSTEM EVALUATOR REJECTED YOUR ANSWER]:\n{eval_text}\nCorrect your answer, or use [ACTION: READ_FILE | INPUT: <path>] to get missing data before answering again."
        
        telemetry["llm_reasoning_sec"] = round(telemetry["llm_reasoning_sec"], 3)
        telemetry["total_latency_sec"] = round(time.perf_counter() - t_start_total, 3)
        return {"answer": reply + "\n\n[System Warning: Autonomous reasoning loop reached max iterations.]", "references": context_dict, "telemetry": telemetry}
    
if __name__ == "__main__":
    agent = CodeMindAgent()
    print("CodeMind Agent Initialized. Type 'exit' to quit.")
    while True:
        user_query = input("\nAsk an architectural question: ")
        if user_query.lower() == 'exit':
            break
        print("\nThinking...")
        print(agent.ask(user_query))