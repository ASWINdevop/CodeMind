import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")     

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")
CHROMA_DB_DIR = os.path.join(DATA_DIR, "chroma")
GRAPH_FILE_PATH = os.path.join(DATA_DIR, "graph", "code_graph.pkl")

os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GRAPH_FILE_PATH), exist_ok=True)

REPOS_DIR = os.path.join(BASE_DIR, "repos")
TARGET_REPO_PATH = os.path.join(REPOS_DIR, "ASWINdevop")

EMBEDDING_MODEL_NAME = "nomic-ai/nomic-embed-text-v1.5"
GEMINI_MODEL_NAME = "gemini-2.5-flash"
CHROMA_COLLECTION_NAME = "codebase_v1"