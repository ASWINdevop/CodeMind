import os
import pickle
import networkx as nx
from tree_sitter_languages import get_parser
from src.config import REPOS_DIR, GRAPH_FILE_PATH

# Configuration based on TDD specs
EXCLUDED_DIRS = {'.git', 'node_modules', '__pycache__', 'venv', 'env', 'images'}

# Language mapping for dynamic Tree-sitter initialization
LANG_MAP = {
    '.py': 'python',
    '.js': 'javascript',
    '.ts': 'typescript',
    '.java': 'java',
    '.cpp': 'cpp',
    '.go': 'go'
}

# AST node types that indicate a function/method across different languages
FUNC_NODE_TYPES = {
    'function_definition',   # Python, C++, Go
    'function_declaration',  # JS, TS, C++
    'method_declaration',    # Java, JS, TS
    'method_definition',     # C++
    'arrow_function'         # JS, TS
}

class CodeIndexer:
    def __init__(self, repos_dir):
        self.repos_dir = repos_dir
        self.graph = nx.DiGraph()
        self.parsers = {}

    def get_parser_for_ext(self, ext):
        lang = LANG_MAP.get(ext)
        if not lang: 
            return None
        if lang not in self.parsers:
            self.parsers[lang] = get_parser(lang)
        return self.parsers[lang]

    def index_all_repos(self):
        if not os.path.exists(self.repos_dir):
            print(f"Directory {self.repos_dir} not found. Please clone repos first.")
            return

        for repo_name in os.listdir(self.repos_dir):
            repo_path = os.path.join(self.repos_dir, repo_name)
            
            if os.path.isdir(repo_path):
                self.graph.add_node(repo_name, type='repository', id=repo_name)
                self._index_single_repo(repo_name, repo_path)

    def _index_single_repo(self, repo_name, repo_path):
        for root, dirs, files in os.walk(repo_path):
            dirs[:] = [d for d in dirs if d not in EXCLUDED_DIRS]
            
            for file in files:
                filepath = os.path.join(root, file)
                ext = os.path.splitext(file)[1]
                
                # NEW: Explicit Documentation Extraction
                if file.lower() == 'readme.md':
                    try:
                        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                            readme_content = f.read()
                        
                        readme_id = f"{repo_name}::readme"
                        self.graph.add_node(
                            readme_id, 
                            type='readme', 
                            id=readme_id, 
                            repo=repo_name, 
                            content=readme_content  # Store the raw text directly in the Graph memory
                        )
                        self.graph.add_edge(repo_name, readme_id, relationship='HAS_DOCS')
                    except Exception as e:
                        print(f"FAILED to read README for {repo_name} | Error: {e}")

                # Existing Code Parser
                elif ext in LANG_MAP:
                    self._process_file(repo_name, repo_path, filepath, ext)

    def _process_file(self, repo_name, repo_path, filepath, ext):
        rel_path = os.path.relpath(filepath, repo_path)
        file_node_id = f"{repo_name}::{rel_path}"
        
        self.graph.add_node(file_node_id, type='file', id=file_node_id, repo=repo_name)
        self.graph.add_edge(repo_name, file_node_id, relationship='CONTAINS_FILE')

        parser = self.get_parser_for_ext(ext)
        if not parser: 
            return

        try:
            # FIX 1: Read strictly as binary bytes to align with Tree-sitter architecture
            with open(filepath, 'rb') as f:
                source_bytes = f.read()
            
            tree = parser.parse(source_bytes)
            self._extract_definitions(repo_name, file_node_id, tree.root_node, source_bytes)
        except Exception as e:
            # FIX 2: Expose the error trace directly to the terminal rather than swallowing it
            print(f"FAILED to parse {rel_path} | Error: {e}")

    def _extract_definitions(self, repo_name, file_node_id, node, source_bytes, current_scope_id=None):
        # Determine current structural scope (Are we inside a file or a specific function?)
        scope_id = current_scope_id if current_scope_id else file_node_id

        # 1. Extract Functions
        if node.type in FUNC_NODE_TYPES:
            func_name = "unknown_func"
            for child in node.children:
                if child.type == 'identifier':
                    func_name = source_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='ignore')
                    break
            
            node_id = f"{file_node_id}::{func_name}"
            self.graph.add_node(
                node_id, type='function', id=node_id, repo=repo_name,
                metadata={'start_byte': node.start_byte, 'end_byte': node.end_byte}
            )
            self.graph.add_edge(file_node_id, node_id, relationship='CONTAINS_FUNC')
            # Update scope so nested calls are attributed to this function
            scope_id = node_id

        # 2. Extract Global Variables & UI Blocks (e.g., Streamlit UI code)
        elif node.type in ['assignment', 'expression_statement'] and getattr(node.parent, 'type', '') in ['module', 'program']:
            chunk_id = f"{file_node_id}::global_{node.start_byte}"
            self.graph.add_node(
                chunk_id, type='function', id=chunk_id, repo=repo_name,
                metadata={'start_byte': node.start_byte, 'end_byte': node.end_byte}
            )
            self.graph.add_edge(file_node_id, chunk_id, relationship='CONTAINS_GLOBAL')
            scope_id = chunk_id

        # 3. NEW: Extract Import Dependencies (Cross-File)
        elif node.type in ['import_statement', 'import_from_statement']:
            import_name = source_bytes[node.start_byte:node.end_byte].decode('utf-8', errors='ignore')
            # Create a placeholder node for the imported module
            module_id = f"{repo_name}::module::{import_name}"
            self.graph.add_node(module_id, type='module', id=module_id, repo=repo_name)
            self.graph.add_edge(file_node_id, module_id, relationship='DEPENDS_ON')

        # 4. NEW: Extract Function Calls (Execution Flow)
        elif node.type == 'call':
            for child in node.children:
                if child.type in ['identifier', 'attribute']:
                    called_func = source_bytes[child.start_byte:child.end_byte].decode('utf-8', errors='ignore')
                    target_id = f"{repo_name}::external_call::{called_func}"
                    
                    self.graph.add_node(target_id, type='call_target', id=target_id, repo=repo_name)
                    # Link the current scope (Function or Global) to the function it is calling
                    self.graph.add_edge(scope_id, target_id, relationship='CALLS')
                    break

        # 5. Recursive Traversal (Pass the updated scope down the tree)
        for child in node.children:
            self._extract_definitions(repo_name, file_node_id, child, source_bytes, scope_id)

    def save_graph(self, output_path):
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        print(f"Graph saved: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges.")

if __name__ == "__main__":
    indexer = CodeIndexer(REPOS_DIR)
    indexer.index_all_repos()
    indexer.save_graph(GRAPH_FILE_PATH)