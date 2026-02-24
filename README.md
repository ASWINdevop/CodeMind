# 🧠 CodeMind: Autonomous Graph-RAG Codebase Agent

CodeMind is a stateful, autonomous Retrieval-Augmented Generation (RAG) microservice engineered to ingest, map, and reason over multi-repository codebases. Unlike standard RAG pipelines that rely purely on semantic similarity—often resulting in fragmented context and high hallucination rates—CodeMind bridges vector search with deterministic graph topology to enforce strict structural comprehension.

#### 🏗️ Core Architecture & Data Flow

CodeMind operates in two distinct, decoupled phases: Knowledge Ingestion and Autonomous Retrieval.

#### Phase 1: Knowledge Ingestion Pipeline

>When a user inputs a GitHub repository, the system executes a deterministic parsing loop:

>Repository Cloning: Physically clones the target codebase to a local, Docker-mounted volume (/app/repos).

>AST Parsing (Tree-sitter): Scans the source code and extracts the Abstract Syntax Tree (AST) using strictly version-locked C-bindings (tree_sitter==0.21.3).

>Topological Mapping (NetworkX): Identifies function calls, imports, and class dependencies, mapping them into a directed mathematical graph.

>Vector Embedding (ChromaDB & PyTorch): Chunks the raw code and generates dense embeddings using a CPU-optimized Nomic embedding model (via sentence_transformers and einops).

>Stateful Persistence: Saves the graph topology to data/graph.json and the vector index to data/chroma.sqlite3.

##### Phase 2: Autonomous ReAct Retrieval Loop

>When a user queries the system, it does not simply pass a prompt to an LLM. It executes an adversarial reasoning chain:

>Asymmetric Cache Interception: Checks the query_cache.json. If a previously "Healed" (verified) answer exists, it bypasses the LLM entirely, dropping latency from ~40s to 0s.

>Hybrid Retrieval: If no cache exists, it queries ChromaDB for semantic matches and expands those nodes via the NetworkX graph to fetch immediately adjacent dependencies.

>ReAct Generator Pass: The Gemini model analyzes the retrieved context. If it lacks critical information (e.g., a utility file referenced in the graph but missing from context), it outputs a strict system command: [ACTION: READ_FILE | INPUT: path/to/file.py].

>Physical Tool Execution: The backend intercepts this action, physically reads the requested file from the mounted drive, and injects the raw code back into the prompt.

>Adversarial Evaluator Pass (Self-Healing): A secondary LLM agent acts as a strict QA Evaluator. It audits the Generator's proposed answer against the raw context. If it detects a hallucination, it mathematically rejects the output ([EVAL: FAIL]) and forces regeneration.

>UI Rendering: Outputs the verified answer to the Streamlit UI, appending strict citation markers and dynamic telemetry/cache badges.

#### 🚀 Key Engineering Features

>Zero-Hallucination Framework: The adversarial evaluator mathematically ensures all outputs are explicitly grounded in the source code.

>READ_FILE Autonomous Tooling: Allows the LLM to dynamically browse the local hard drive to resolve its own blindspots before rendering an answer.

>Asymmetric Caching Engine: Differentiates between standard generations and high-tier "Healed" generations, permanently overwriting weak cache entries to optimize API token burn.

>Optimized Docker Containerization: Engineered with a strict .dockerignore and explicit CPU-only PyTorch wheels to strip 10GB+ of NVIDIA CUDA binaries, reducing the final image footprint from 13.7GB to 3.38GB.

#### ⚙️ Tech Stack

>Backend & Logic: Python 3.11, Google GenAI SDK (Gemini API)

>Data Structures: ChromaDB (Vector Index), NetworkX (Directed Graphs)

>Parsing & ML: Tree-sitter (AST), Sentence Transformers, Einops

>Frontend & Telemetry: Streamlit, Streamlit-Agraph

>Infrastructure: Docker, Docker Compose

#### 🛠️ Installation & Deployment

CodeMind is designed to run in a mathematically isolated Linux environment using Docker, while maintaining stateful persistence on your host machine.

###### 1. Prerequisites

    Docker Desktop installed and running.

    A valid Google Gemini API Key.

###### 2. Environment Setup

    Clone the repository and create a .env file in the root directory:
```
git clone <your-repo-url>
cd CODEMIND
echo "GOOGLE_API_KEY=your_gemini_api_key_here" > .env
```


**Note**: Ensure the local directories data/chroma, data/graph, and data/repos exist with their respective .gitkeep files so Docker can mount the volumes properly.

###### 3. Build and Execute

Because the architecture relies on heavy C++ bindings, launch the system using the pre-configured Compose controller:
```
docker-compose up -d --build
```

Docker Compose will automatically bridge port 8501 and mount your physical hard drive to prevent data loss when the container spins down.

###### 4. Access the Application

Navigate your browser strictly to:
```http://localhost:8501```

##### 📊 Usage Guide & Telemetry

**Ingestion**: Use the left sidebar to fetch and ingest a GitHub repository. Wait for the Knowledge Graph and Vector Index to compile.

**Querying**: Ask architectural questions in the main chat interface.

**Toggle Self-Healing**: Use the UI toggle to enable the Adversarial Evaluator. This increases latency (~40s) but guarantees zero hallucinations.

**Interpreting Cache Badges:**

⚡ CACHED : STANDARD: A 0-second response from a standard, non-evaluated query.

🏥 CACHED : HEALED: A 0-second response from a strictly evaluated, hallucination-free reasoning chain.

📁 Project Structure
```
CODEMIND/
├── data/                  # Stateful volume mounts (Ignored by Git)
│   ├── chroma/            # Vector embeddings database
│   ├── graph/             # NetworkX JSON topology
├── repos/             # Cloned target codebases
├── src/                   # Core Application Source
│   ├── app.py             # Streamlit UI & CSS overrides
│   ├── rag.py             # ReAct Loop, Caching & Gemini Integration
│   ├── indexer.py         # Tree-sitter AST & Graph Mapping
│   ├── clone_repos.py     # Clones selected repos to local drive
│   ├── config.py          # Contains configurations and paths
│   ├── logo.png           # Contains logo of the system
│   └── vector_store.py    # ChromaDB & Nomic Embedding config
├── .dockerignore          # Prevents 13GB build context bloat
├── .env                   # API Keys (Ignored by Git)
├── .gitignore             # Contains files to be ignored by git 
├── get_metrics.py         # Fetches metrics of system in real-time
├── docker-compose.yml     # Volume and port controller
├── Dockerfile             # 3.38GB optimized Python 3.11-slim image
└── requirements.txt       # Version-locked dependencies
```
