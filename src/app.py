import os
import base64
import streamlit as st
from streamlit_agraph import agraph, Node, Edge, Config
from src.rag import CodeMindAgent
from src.config import REPOS_DIR, GRAPH_FILE_PATH
from src.clone_repos import get_available_repos, clone_selected_repos
from src.indexer import CodeIndexer
from src.vector_store import CodeVectorStore

# 1. PAGE CONFIGURATION (Must be first)
st.set_page_config(
    page_title="CodeMind Agent", 
    page_icon="🧠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to load local image as base64 for inline HTML
def get_base64_of_bin_file(bin_file):
    try:
        with open(bin_file, 'rb') as f:
            data = f.read()
        return base64.b64encode(data).decode()
    except Exception as e:
        print(f"Error loading logo: {e}")
        return None

# Get dynamic path to logo.png inside the src folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(CURRENT_DIR, "logo.png")

# 2. CUSTOM CSS
st.markdown("""
    <style>
        /* Hide default Streamlit elements safely */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {background-color: transparent !important;}
        
        /* Main App Background Gradient */
        .stApp {
            background: linear-gradient(135deg, #d8f3f0 0%, #e9f7f6 35%, #f8fbfb 100%);
        }
        /* --- FIX METRICS GAP (OVERRIDE 75vh INHERITANCE) --- */
        [data-testid="stSidebar"] [data-testid="stColumn"],
        [data-testid="stExpander"] [data-testid="stColumn"] {
            height: auto !important;
            max-height: none !important;
            overflow-y: visible !important;
        }
        /* ===== STATIC CODE VIEWER PANEL ===== */
/* RIGHT COLUMN - STATIC CODE VIEWER PANEL */
[data-testid="stColumn"]:nth-child(2) 
[data-testid="stVerticalBlockBorderWrapper"] {

    height: 75vh;
    max-height: 75vh;
    overflow-y: auto;

    padding: 1.5rem;
    border-radius: 28px;

    background: linear-gradient(
        135deg,
        rgba(255,255,255,0.85),
        rgba(225,239,254,0.6)
    );

    backdrop-filter: blur(24px);
    border: 2px solid rgba(59,130,246,0.3);
    box-shadow: 0 12px 40px rgba(31,38,135,0.12);
}


/* Make chat area scrollable too */
[data-testid="stColumn"]:nth-child(1) {
    height: 75vh;
    overflow-y: auto;
}
    


        /* Global Text Color Override */
        h2, h3, h4, h5, h6, p, label, .stMarkdown:not(.stCodeBlock *), [data-testid="stText"] {
            font-family: 'Inter', -apple-system, sans-serif;
            color: #0f172a !important; 
        }
        
        /* CodeMind Title */
        h1 {
            font-size: 3.5rem !important;
            font-weight: 800 !important;
            letter-spacing: -1px;
            background: linear-gradient(90deg, #0f172a, #3b82f6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            padding-bottom: 0rem;
            margin-bottom: -15px;
            display: flex;
            align-items: center;
        }

        /* --- GLASSMORPHISM & PILL SHAPES --- */
        
        /* Sidebar Base */
        [data-testid="stSidebar"] {
            background: rgba(255, 255, 255, 0.3) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-right: 1px solid rgba(255, 255, 255, 0.6) !important;
        }
        
        /* Left Panel Container (Standard Glass) */
        [data-testid="stColumn"]:nth-child(1) [data-testid="stVerticalBlockBorderWrapper"] {
            background: rgba(255, 255, 255, 0.4) !important;
            backdrop-filter: blur(20px) !important;
            border: 1px solid rgba(255, 255, 255, 0.8) !important;
            border-radius: 32px !important; 
            box-shadow: 0 8px 32px rgba(31, 38, 135, 0.05) !important;
            padding: 2rem !important;
        }

        /* ===== Response Badge ===== */
        .response-badge {
        display: inline-block;
        margin-bottom: 6px;

        background: rgba(59,130,246,0.10);
        color: #1e3a8a;

        border: 1px solid rgba(59,130,246,0.25);
        font-weight: 600;
        font-size: 0.68rem;

        padding: 2px 8px;
        border-radius: 6px;
    }



        /* Deep overrides to strip Streamlit's solid dark backgrounds */
        div[data-baseweb="input"], 
        div[data-baseweb="select"] {
            background-color: transparent !important;
        }

        /* Glass Pill Inputs */
        div[data-baseweb="input"] > div,
        div[data-baseweb="select"] > div {
            background-color: rgba(255, 255, 255, 0.5) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.9) !important;
            border-radius: 9999px !important; 
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04) !important;
        }

        div[data-baseweb="input"] input, 
        div[data-baseweb="select"] div {
            color: #0f172a !important;
            background-color: transparent !important;
        }

        /* --- CHAT INPUT FIX (NUKE DARK BACKGROUND) --- */
        [data-testid="stChatInput"] {
            background-color: transparent !important;
        }
        
        /* Force transparency on all nested BaseWeb divs inside the chat input */
        [data-testid="stChatInput"] div[data-baseweb="base-input"],
        [data-testid="stChatInput"] div[data-baseweb="textarea"],
        [data-testid="stChatInput"] textarea {
            background-color: transparent !important;
            color: #0f172a !important;
            caret-color: #0f172a !important; /* Fixes missing cursor */
        }

        /* Target the outer visible pill wrapper */
        [data-testid="stChatInput"] > div { 
            background-color: rgba(255, 255, 255, 0.7) !important;
            backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(255, 255, 255, 0.9) !important;
            border-radius: 9999px !important;
            box-shadow: 0 4px 16px rgba(31, 38, 135, 0.05) !important;
        }

        [data-testid="stChatInput"] button {
            background-color: transparent !important;
        }
        
        [data-testid="stChatInput"] button svg {
            fill: #3b82f6 !important; 
            color: #3b82f6 !important;
        }

        /* --- BUTTONS FIX (Fetch Repositories & Ingest) --- */
        .stButton > button, 
        button[data-testid="baseButton-secondary"], 
        button[data-testid="baseButton-primary"] {
            background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.4) !important;
            border-radius: 9999px !important; 
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3) !important;
            transition: all 0.2s ease !important;
            color: #ffffff !important; 
        }
        
        .stButton > button p,
        .stButton > button span,
        button[data-testid="baseButton-secondary"] p, 
        button[data-testid="baseButton-primary"] p {
            color: #ffffff !important;
            font-weight: 600 !important;
        }

        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
            border: 1px solid rgba(255, 255, 255, 0.7) !important;
        }

        /* Expander Fix */
        [data-testid="stExpander"] {
            background-color: rgba(255, 255, 255, 0.6) !important;
            backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(255, 255, 255, 0.9) !important;
            border-radius: 16px !important;
            overflow: hidden !important;
        }
        [data-testid="stExpander"] summary {
            background-color: rgba(255, 255, 255, 0.8) !important;
            padding: 10px 15px !important;
        }
        [data-testid="stExpander"] summary p, 
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary svg {
            color: #0f172a !important; 
            font-weight: 700 !important;
        }
        [data-testid="stExpanderDetails"] {
            background-color: transparent !important;
        }

        /* Chat Messages */
        [data-testid="stChatMessage"] {
            background: rgba(255, 255, 255, 0.5) !important;
            backdrop-filter: blur(12px) !important;
            border: 1px solid rgba(255, 255, 255, 0.8) !important;
            border-radius: 24px !important;
            padding: 1rem 1.5rem !important;
            margin-bottom: 1rem !important;
        }

        [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
            background: rgba(240, 247, 255, 0.6) !important;
            border: 1px solid rgba(225, 239, 254, 0.8) !important;
        }

        /* Citation Pills */
        [data-testid="stChatMessage"] code {
            background: rgba(241, 245, 249, 0.7) !important; 
            color: #475569 !important; 
            border-radius: 9999px !important;
            padding: 2px 10px !important;
            font-weight: 600 !important;
            font-size: 0.85em !important;
            border: 1px solid rgba(203, 213, 225, 0.8) !important;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02) !important;
        }

        /* Main Source Code Block */
        .stCodeBlock {
            border-radius: 16px !important; 
            border: 1px solid rgba(203, 213, 225, 0.6) !important;
            background-color: rgba(248, 250, 252, 0.8) !important; 
            backdrop-filter: blur(10px) !important;
            overflow: hidden !important;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_agent():
    return CodeMindAgent()

def main():
    # --- LOGO & TITLE INTEGRATION ---
    logo_b64 = get_base64_of_bin_file(LOGO_PATH)
    
    # We use a container div to manage the layout of the title and icon
    if logo_b64:
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; flex-wrap: wrap; gap: 20px; margin-bottom: 5px;">
                <img src="data:image/png;base64,{logo_b64}" width="200" style="border-radius: 12px; flex-shrink: 0;">
                <h1 style="margin: 0; padding: 0; line-height: 1;">CodeMind</h1>
            </div>
            """,
            unsafe_allow_html=True
        )
        # We add a margin-left to the caption so it aligns with the text, not the logo
        st.markdown(
            f"""<p style="margin-left: 170px; margin-top: -30px; opacity: 0.8; font-size: 0.9rem;">
            Local Codebase Memory & Architectural Assistant
            </p>""", 
            unsafe_allow_html=True
        )
    else:
        st.title("🧠 CodeMind")
        st.caption("Local Codebase Memory & Architectural Assistant")

    st.markdown("<br>", unsafe_allow_html=True)


    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "citation_mapping" not in st.session_state:
        st.session_state.citation_mapping = {}
    if "latest_references" not in st.session_state:
        st.session_state.latest_references = {}

    # --- SIDEBAR: SETTINGS, CITATIONS & INGESTION ---
    with st.sidebar:
        # Add Logo to top of sidebar
        if logo_b64:
            # We can pass the path directly here for Streamlit's native image handler
            st.image(LOGO_PATH, width=200)
            
        st.header("Settings")
        
        available_cloned_repos = [d for d in os.listdir(REPOS_DIR) if os.path.isdir(os.path.join(REPOS_DIR, d))] if os.path.exists(REPOS_DIR) else []
        selected_target = st.selectbox("Target Context", ["All Repositories"] + available_cloned_repos)
        repo_filter = None if selected_target == "All Repositories" else selected_target

        st.divider()

        st.subheader("Citation Index")
        if st.session_state.citation_mapping:
            for marker, full_name in st.session_state.citation_mapping.items():
                st.markdown(f"**{marker}**: `{full_name}`")
        else:
            st.caption("No citations active.")

        st.divider()

        # --- NEW: SESSION ANALYTICS ENGINE ---
        st.subheader(" Session Analytics")
        
        valid_responses = 0
        sum_total_latency = 0.0
        sum_retrieval_latency = 0.0
        sum_react_loops = 0
        sum_tokens_in = 0
        sum_tokens_out = 0

        for msg in st.session_state.messages:
            if msg.get("role") == "assistant" and msg.get("telemetry"):
                # Mathematical Filter: Ignore loops that hit max iterations
                if "[System Warning: Autonomous reasoning loop reached max iterations.]" not in msg.get("content", ""):
                    tel = msg["telemetry"]
                    valid_responses += 1
                    sum_total_latency += tel.get("total_latency_sec", 0)
                    sum_retrieval_latency += tel.get("retrieval_latency_sec", 0)
                    sum_react_loops += tel.get("react_iterations", 0)
                    sum_tokens_in += tel.get("prompt_tokens", 0)
                    sum_tokens_out += tel.get("completion_tokens", 0)

        if valid_responses > 0:
            avg_total_latency = round(sum_total_latency / valid_responses, 2)
            avg_retrieval = round(sum_retrieval_latency / valid_responses, 2)
            avg_loops = round(sum_react_loops / valid_responses, 1)
            avg_tokens_in = int(sum_tokens_in / valid_responses)
            avg_tokens_out = int(sum_tokens_out / valid_responses)
            
            # Calculate total burned tokens across all successful queries
            total_tokens_burned = sum_tokens_in + sum_tokens_out

            col1, col2 = st.columns(2)
            col1.metric("Avg Latency", f"{avg_total_latency}s")
            col2.metric("Avg Retrieval", f"{avg_retrieval}s")
            
            col3, col4 = st.columns(2)
            col3.metric("Avg ReAct Loops", f"{avg_loops}")
            col4.metric("Avg Tokens (In/Out)", f"{avg_tokens_in} / {avg_tokens_out}")
            
            # Display Total Tokens
            st.metric("Total Tokens Burned", f"{total_tokens_burned:,}")
            
            st.caption(f"Calculated from {valid_responses} successful queries.")
        else:
            st.caption("No valid telemetry data yet.")

    
        st.divider()
        # --- END SESSION ANALYTICS ENGINE ---

        with st.expander("🛠️ Knowledge Ingestion Panel", expanded=not bool(available_cloned_repos)):
            github_user = st.text_input("GitHub Username", value="ASWINdevop")
            
            if st.button("Fetch Repositories", use_container_width=True):
                with st.spinner("Querying API..."):
                    repos = get_available_repos(github_user)
                    if repos:
                        st.session_state.available_repos = repos
                        st.success(f"Found {len(repos)} repos.")
                    else:
                        st.error("Failed to fetch repositories.")
                        st.session_state.available_repos = []

            if st.session_state.get("available_repos"):
                selected_repos = st.multiselect("Select Repositories", options=st.session_state.available_repos)
                if st.button("Ingest & Embed", type="primary", use_container_width=True):
                    if selected_repos:
                        with st.status("Building Knowledge Graph...", expanded=True) as status:
                            st.write("Step 1: Cloning/Pulling repositories...")
                            clone_selected_repos(github_user, selected_repos)
                            
                            st.write("Step 2: Parsing AST & Indexing...")
                            indexer = CodeIndexer(REPOS_DIR)
                            indexer.index_all_repos()
                            indexer.save_graph(GRAPH_FILE_PATH)
                            
                            st.write("Step 3: Generating Vector Embeddings...")
                            store = CodeVectorStore()
                            store.populate_from_graph(target_repos=selected_repos)
                            
                            status.update(label="Ingestion Complete!", state="complete", expanded=False)
                        st.cache_resource.clear()
                        st.rerun()

    # --- INITIALIZATION CHECKS ---
    agent = None
    agent_ready = False
    try:
        if os.path.exists(GRAPH_FILE_PATH):
            agent = load_agent()
            agent_ready = True
    except Exception as e:
        st.error(f"Error loading agent: {e}")

    # --- MAIN UI: DUAL-PANE LAYOUT ---
    chat_col, cite_col = st.columns([1, 1], gap="large")

    # LEFT PANEL: Chat Window
    with chat_col:
        with st.container(border=True):
            st.subheader(" Chat")
            
            if not agent_ready:
                st.info("👋 Welcome! Please use the **Knowledge Ingestion Panel** in the sidebar to load repositories before chatting.")
            else:
                assistant_counter = 0
                user_counter = 0

                for msg in st.session_state.messages:
                    role = msg["role"]

                    if role == "assistant":
                        assistant_counter += 1
                        label = f"🤖 Response #{assistant_counter}"
                    else:
                        user_counter += 1
                        label = f"👤 Query #{user_counter}"

                    with st.chat_message(role):
                        st.markdown(
                            f'<div class="response-badge">{label}</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(msg["content"])
                        
                        # NEW: PERSISTENT TELEMETRY RENDERER
                        # NEW: PERSISTENT TELEMETRY RENDERER
                        if role == "assistant" and "telemetry" in msg and msg["telemetry"]:
                            tel = msg["telemetry"]
                            with st.expander("Performance Telemetry"):
                                met1, met2, met3, met4 = st.columns(4)
                                met1.metric("Total Latency", f"{tel.get('total_latency_sec', 0)}s")
                                met2.metric("Retrieval", f"{tel.get('retrieval_latency_sec', 0)}s")
                                met3.metric("ReAct Loops", f"{tel.get('react_iterations', 0)}")
                                met4.metric("Tokens (In/Out)", f"{tel.get('prompt_tokens', 0)} / {tel.get('completion_tokens', 0)}")

                if prompt := st.chat_input("Ask an architectural question..."):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": prompt
                    })

                    with st.chat_message("user"):
                        st.markdown(prompt)

                    with st.chat_message("assistant"):
                        with st.spinner("Analyzing codebase..."):
                            # 1. Extract the payload including our new telemetry dictionary
                            payload = agent.ask(prompt, repo_filter=repo_filter)
                            raw_answer = payload["answer"]
                            references = payload["references"]
                            telemetry = payload.get("telemetry", {}) # Retrieve telemetry

                            mapped_answer = raw_answer
                            current_mapping = {}
                            r_counter = 1
                            num_counter = 1
                            
                            sorted_keys = sorted(references.keys(), key=len, reverse=True)
                            total_readmes = sum(1 for k in sorted_keys if "README" in k.upper())
                            
                            for key in sorted_keys:
                                if key in mapped_answer:
                                    is_readme = "README" in key.upper()
                                    marker_text = f"[R{r_counter}]" if total_readmes > 1 else "[R]" if is_readme else f"[{num_counter}]"
                                    if is_readme: r_counter += 1
                                    else: num_counter += 1

                                    markdown_marker = f"`{marker_text}`"
                                    mapped_answer = mapped_answer.replace(key, markdown_marker)
                                    current_mapping[marker_text] = key

                            st.session_state.citation_mapping = current_mapping
                            st.session_state.latest_references = references

                            # 2. Render the primary answer
                            st.markdown(mapped_answer)
                            
                            # 3. RENDER THE TELEMETRY DASHBOARD
                            # 3. RENDER THE TELEMETRY DASHBOARD
                            if telemetry:
                                with st.expander("Performance Telemetry"):
                                    met1, met2, met3, met4 = st.columns(4)
                                    met1.metric("Total Latency", f"{telemetry['total_latency_sec']}s")
                                    met2.metric("Retrieval (Graph+Vector)", f"{telemetry['retrieval_latency_sec']}s")
                                    met3.metric("ReAct Iterations", f"{telemetry['react_iterations']}")
                                    met4.metric("Tokens (In / Out)", f"{telemetry['prompt_tokens']} / {telemetry['completion_tokens']}")

                            # 4. Save to session state
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": mapped_answer,
                                "citations": current_mapping,
                                "references": references,
                                "telemetry": telemetry # Save telemetry history
                            })

                            st.rerun() 

    # RIGHT PANEL: Source Code Viewer
    # RIGHT PANEL: Source Code & Graph Viewer
    # RIGHT PANEL: Context Explorer
    with cite_col:
        with st.container():
            st.subheader("Context Explorer")

            assistant_messages = []
            response_number = 0

            # Gather all responses that contain citations
            for i, msg in enumerate(st.session_state.messages):
                if msg["role"] == "assistant" and "citations" in msg:
                    response_number += 1
                    assistant_messages.append((response_number, i, msg))

            if not assistant_messages:
                st.caption("No citations active. Ask a question to view context.")
            else:
                # 1. MOVED ABOVE TABS: Global Selection Dropdown
                selected_tuple = st.selectbox(
                    "Select a response context:",
                    options=assistant_messages,
                    format_func=lambda x: f"Response #{x[0]}"
                )

                response_number, msg_index, selected_msg = selected_tuple
                citation_mapping = selected_msg.get("citations", {})
                references = selected_msg.get("references", {})

                # 2. DECLARE TABS AFTER DROPDOWN
                tab_code, tab_graph = st.tabs(["💻 Source Code", "🕸️ Graph Visualizer"])

                # --- SOURCE CODE TAB ---
                with tab_code:
                    if citation_mapping:
                        display_options = [f"{m} {k}" for m, k in citation_mapping.items()]
                        selected_display = st.selectbox("Select a reference to inspect:", display_options)

                        if selected_display:
                            selected_marker = selected_display.split(" ")[0]
                            actual_key = citation_mapping.get(selected_marker)

                            if actual_key in references:
                                st.code(
                                    references[actual_key],
                                    language="python",
                                    line_numbers=True
                                )
                    else:
                        st.caption("No direct citations in this response.")

                # --- GRAPH VISUALIZER TAB ---
                with tab_graph:
                    if citation_mapping and agent and agent.graph:
                        nodes = []
                        edges = []
                        added_nodes = set()
                        
                        # Colors for structural types
                        COLOR_CENTER = "#3b82f6" # Blue
                        COLOR_CALLER = "#10b981" # Green
                        COLOR_CALLEE = "#ef4444" # Red
                        
                        # NEW: Font configuration for high-contrast readability
                        label_font = {
                            "color": "white", 
                            "size": 14, 
                            "face": "sans-serif", 
                            "strokeWidth": 3, 
                            "strokeColor": "#0f172a" # Dark outline so text pops
                        }
                        edge_font = {
                            "color": "#e2e8f0", 
                            "size": 10, 
                            "background": "#0f172a" # Dark background for edge labels
                        }

                        for marker, node_id in citation_mapping.items():
                            if node_id in agent.graph:
                                # Add core cited node
                                if node_id not in added_nodes:
                                    label = node_id.split("::")[-1]
                                    nodes.append(Node(id=node_id, label=label, size=25, color=COLOR_CENTER, font=label_font))
                                    added_nodes.add(node_id)
                                
                                # Add Successors (What this node calls)
                                for callee in agent.graph.successors(node_id):
                                    if callee not in added_nodes:
                                        label = callee.split("::")[-1]
                                        nodes.append(Node(id=callee, label=label, size=15, color=COLOR_CALLEE, font=label_font))
                                        added_nodes.add(callee)
                                    edges.append(Edge(source=node_id, target=callee, label="CALLS", type="CURVE_SMOOTH", color="#94a3b8", font=edge_font))
                                
                                # Add Predecessors (What calls this node)
                                for caller in agent.graph.predecessors(node_id):
                                    if caller not in added_nodes:
                                        label = caller.split("::")[-1]
                                        nodes.append(Node(id=caller, label=label, size=15, color=COLOR_CALLER, font=label_font))
                                        added_nodes.add(caller)
                                    edges.append(Edge(source=caller, target=node_id, label="CALLED_BY", type="CURVE_SMOOTH", color="#94a3b8", font=edge_font))

                        if nodes:
                            config = Config(
                                width="100%",
                                height=600,
                                directed=True,
                                physics=True,
                                hierarchical=False,
                                nodeHighlightBehavior=True,
                                highlightColor="#facc15",
                                backgroundColor="#0f172a" # Explicit dark background
                            )
                            agraph(nodes=nodes, edges=edges, config=config)
                        else:
                            st.caption("No valid topological connections found for these citations.")
                    else:
                        st.caption("Graph data unavailable for current context.")
if __name__ == "__main__":
    main()
            
