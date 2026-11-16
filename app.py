import json
import os
import streamlit as st

# Function to discover any keys present in st.secrets or os.environ
def resolve_api_key(key_name: str) -> str:
    if os.getenv(key_name):
        return os.getenv(key_name).strip()

    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name]).strip()
    except Exception:
        pass

    try:
        if hasattr(st, "secrets"):
            for section in st.secrets:
                if section.lower() == key_name.lower():
                    return str(st.secrets[section]).strip()
                try:
                    sec_obj = st.secrets[section]
                    if hasattr(sec_obj, "items"):
                        for sub_k, sub_v in sec_obj.items():
                            if sub_k.lower() == key_name.lower():
                                return str(sub_v).strip()
                except Exception:
                    pass
    except Exception:
        pass

    return ""


for k in ["GROQ_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY", "LLM_PROVIDERS"]:
    found_val = resolve_api_key(k)
    if found_val:
        os.environ[k] = found_val

from config.settings import CHUNKS_DIR, PARSED_DIR
from src.chunking import (
    split_fixed_size,
    split_hierarchical,
    split_markdown_sections,
    split_recursive,
    split_semantic_sentences,
    split_token_based,
)
from src.generation.llm import build_llm
from src.indexing.embedder import build_embedder
from src.indexing.vector_store import get_client, get_collection
from src.parsing.schemas import ParsedChapter

# Page configuration
st.set_page_config(
    page_title="NCERT Science Intelligence & Chunking Architecture",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Theme CSS - Works smoothly across Light and Dark modes
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .app-header {
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 1rem;
        margin-bottom: 1.5rem;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        opacity: 0.75;
        font-weight: 400;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: rgba(128, 128, 128, 0.15);
        border: 1px solid rgba(128, 128, 128, 0.3);
        margin-right: 0.5rem;
    }
    
    /* Native container styling for passage cards */
    .passage-meta {
        font-size: 0.82rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 0.35rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing retrieval services...")
def get_rag_components():
    embedder = build_embedder()
    client = get_client()
    return embedder, client


embedder, client = get_rag_components()

# Sidebar Navigation
st.sidebar.markdown("### System Navigation")
mode = st.sidebar.radio(
    "Modules",
    [
        "RAG Assistant",
        "Chunking Architecture Analysis",
        "Repository Chunk Explorer",
    ],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **Environment Architecture**  
    - Retrieval: Dense Cosine (ChromaDB)
    - Embedding: MiniLM-L6 (384-d ONNX)
    - LLM Engine: Groq / Gemini (LCEL)
    """
)

# Optional sidebar key override
st.sidebar.markdown("---")
with st.sidebar.expander("API Key Configuration", expanded=False):
    manual_groq = st.text_input("Groq API Key (Override)", type="password", value=resolve_api_key("GROQ_API_KEY"))
    if manual_groq:
        os.environ["GROQ_API_KEY"] = manual_groq.strip()

# ==============================================================================
# MODULE 1: RAG Assistant
# ==============================================================================
if mode == "RAG Assistant":
    st.markdown(
        """
        <div class="app-header">
            <span class="badge">NCERT CURRICULUM</span>
            <div class="app-title">NCERT Science Query Synthesis System</div>
            <div class="app-subtitle">Semantic search and grounded question-answering across indexed secondary science textbooks.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    try:
        llm = build_llm()
    except Exception:
        llm = None

    if llm is None:
        st.warning("No active LLM detected. You can paste your Groq API key in the sidebar under 'API Key Configuration' or check Streamlit Secrets.")

    available_cols = [c.name for c in client.list_collections()]
    friendly_options = {}
    for c_name in available_cols:
        strat = c_name.replace("ncert_science_minilm_", "").replace("ncert_science_", "")
        friendly_options[f"{strat.capitalize()} Strategy Index ({c_name})"] = strat

    if not friendly_options:
        st.warning("No vector store collections detected. Index chunks to enable retrieval.")
        st.stop()

    c1, c2 = st.columns([3, 1])
    with c1:
        selected_display = st.selectbox("Vector Store Collection", list(friendly_options.keys()))
        selected_strategy = friendly_options[selected_display]
    with c2:
        top_k = st.slider("Passage Retrieval Depth (k)", min_value=1, max_value=8, value=3)

    query = st.text_input(
        "Enter inquiry",
        placeholder="e.g., Explain the law of conservation of mass in chemical reactions.",
    )

    if st.button("Synthesize Answer", type="primary") and query.strip():
        with st.spinner("Executing retrieval and LLM context synthesis..."):
            try:
                col = get_collection(selected_strategy)
                q_vec = embedder.embed_query(query.strip())
                res = col.query(
                    query_embeddings=[q_vec],
                    n_results=top_k,
                    include=["documents", "metadatas", "distances"],
                )

                docs = res["documents"][0] if res["documents"] else []
                metas = res["metadatas"][0] if res["metadatas"] else []
                dists = res["distances"][0] if res["distances"] else []

                if not docs:
                    st.warning("No matching passages exceeded retrieval threshold.")
                else:
                    context_blocks = []
                    for doc, meta in zip(docs, metas):
                        chapter_info = (
                            f"Class {meta.get('class_num', '')} Science - "
                            f"Chapter {meta.get('chapter_num', '')}: {meta.get('chapter_name', '')}"
                        )
                        sec = (
                            f" (Section {meta.get('section_number', '')}: {meta.get('section_title', '')})"
                            if meta.get("section_title")
                            else ""
                        )
                        context_blocks.append(f"[{chapter_info}{sec}]\n{doc}")

                    full_context = "\n\n".join(context_blocks)

                    if llm:
                        prompt = (
                            "You are an academic NCERT Science assistant. Answer the user's question with clarity, "
                            "strict factual grounding, and precision using only the provided textbook passages below. "
                            "Cite the chapter or section when referencing specific concepts.\n\n"
                            f"Context Passages:\n{full_context}\n\n"
                            f"Question: {query}\n\n"
                            "Grounded Synthesis:"
                        )

                        response = llm.invoke(prompt)

                        st.markdown("#### Synthesized Response")
                        st.markdown(response.content)
                    else:
                        st.info("LLM is currently unavailable; showing direct retrieved passages below:")

                    st.markdown("---")
                    st.markdown("#### Retrieved Source Documents")
                    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
                        similarity_pct = max(0.0, min(100.0, (1.0 - dist) * 100))
                        with st.container(border=True):
                            st.markdown(
                                f"""
                                <div class="passage-meta">
                                    <span>PASSAGE {idx} — Class {meta.get('class_num')} | Chapter {meta.get('chapter_num')}: {meta.get('chapter_name')}</span>
                                    <span>Relevance Confidence: <b>{similarity_pct:.1f}%</b> (Distance: {dist:.3f})</span>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            # Render pure markdown safely without raw unclosed HTML divs
                            st.markdown(doc)

            except Exception as e:
                st.error(f"Error executing retrieval pipeline: {e}")

# ==============================================================================
# MODULE 2: Chunking Strategy Comparison
# ==============================================================================
elif mode == "Chunking Architecture Analysis":
    st.markdown(
        """
        <div class="app-header">
            <span class="badge">EVALUATION BENCHMARK</span>
            <div class="app-title">Chunking Strategies Architecture Benchmark</div>
            <div class="app-subtitle">Comparative evaluation of text splitting topologies across structural, linguistic, and statistical metrics.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    json_files = sorted(PARSED_DIR.glob("*.json"))
    if not json_files:
        st.warning("No parsed chapters found in data/parsed.")
        st.stop()

    file_names = [f.name for f in json_files]
    selected_file = st.selectbox("Benchmark Target Chapter", file_names, index=0)

    import json
    with open(PARSED_DIR / selected_file, "r", encoding="utf-8") as f:
        chapter_data = json.load(f)
    chapter = ParsedChapter(**chapter_data)

    st.sidebar.markdown("### Hyperparameter Tuning")
    fixed_size = st.sidebar.number_input("Fixed Window (chars)", value=500, step=50)
    rec_size = st.sidebar.number_input("Recursive Window (chars)", value=800, step=50)
    token_size = st.sidebar.number_input("Token Window (BPE tokens)", value=250, step=25)
    sem_size = st.sidebar.number_input("Semantic Target (chars)", value=800, step=50)

    fixed_chunks = split_fixed_size(chapter, chunk_size=fixed_size, chunk_overlap=50)
    rec_chunks = split_recursive(chapter, chunk_size=rec_size, chunk_overlap=100)
    tok_chunks = split_token_based(chapter, chunk_size=token_size, chunk_overlap=25)
    sem_chunks = split_semantic_sentences(chapter, target_chunk_chars=sem_size)
    md_chunks = split_markdown_sections(chapter, max_chunk_size=1200)
    parents, children = split_hierarchical(chapter)

    strategies_map = {
        "Fixed Character Window": fixed_chunks,
        "Recursive Text Splitting": rec_chunks,
        "Token-Based (Tiktoken BPE)": tok_chunks,
        "Semantic Sentence Boundary": sem_chunks,
        "Markdown Header Split": md_chunks,
        "Hierarchical Parent Context": parents,
        "Hierarchical Child Granular": children,
    }

    summary_data = []
    for s_name, chunk_list in strategies_map.items():
        if not chunk_list:
            continue
        c_lens = [len(c.text) for c in chunk_list]
        w_lens = [len(c.text.split()) for c in chunk_list]
        summary_data.append({
            "Strategy Topology": s_name,
            "Chunk Count": len(chunk_list),
            "Mean Length (Chars)": round(sum(c_lens) / len(c_lens), 1),
            "Min Chars": min(c_lens),
            "Max Chars": max(c_lens),
            "Mean Word Count": round(sum(w_lens) / len(w_lens), 1),
        })

    st.markdown("#### Topology Comparison Matrix")
    st.dataframe(summary_data, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### Granular Chunk Inspector")
    chosen_strategy = st.selectbox("Inspect Strategy", list(strategies_map.keys()))
    inspected_chunks = strategies_map[chosen_strategy]

    c_slider, _ = st.columns([2, 1])
    with c_slider:
        chunk_idx = st.slider("Select Chunk Index", 1, len(inspected_chunks), 1) - 1
    target_c = inspected_chunks[chunk_idx]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Identifier", getattr(target_c, "chunk_id", getattr(target_c, "parent_id", "N/A")))
    m2.metric("Character Count", len(target_c.text))
    m3.metric("Word Count", len(target_c.text.split()))
    m4.metric("Line Count", len(target_c.text.splitlines()))

    st.markdown("**Complete Chunk Content**")
    st.text_area("Chunk Content", target_c.text, height=260, label_visibility="collapsed")

# ==============================================================================
# MODULE 3: Repository Chunk Explorer
# ==============================================================================
elif mode == "Repository Chunk Explorer":
    st.markdown(
        """
        <div class="app-header">
            <span class="badge">DATA REGISTRY</span>
            <div class="app-title">Persistent Chunk Artifact Registry</div>
            <div class="app-subtitle">Direct audit and serialization inspection of persistent JSONL datasets in the local store.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    strategy_folders = sorted([d.name for d in CHUNKS_DIR.iterdir() if d.is_dir()])
    if not strategy_folders:
        st.warning("No strategy directories found in data/chunks/.")
        st.stop()

    c1, c2 = st.columns(2)
    with c1:
        sel_strat = st.selectbox("Partition Strategy Directory", strategy_folders)
    with c2:
        folder_path = CHUNKS_DIR / sel_strat
        files = sorted([f.name for f in folder_path.glob("*.jsonl")])
        sel_file = st.selectbox("Serialized Chapter File", files) if files else None

    if not sel_file:
        st.info("No serialized records found for this strategy.")
        st.stop()

    import json
    chunks_in_file = []
    with open(folder_path / sel_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks_in_file.append(json.loads(line))

    st.markdown(f"Total Serialized Records in File: **{len(chunks_in_file)}**")
    if chunks_in_file:
        c_num = st.slider("Record Offset", 1, len(chunks_in_file), 1) - 1
        item = chunks_in_file[c_num]

        st.markdown("#### Document Metadata Attributes")
        st.json({k: v for k, v in item.items() if k != "text"})

        st.markdown("#### Payload Content")
        st.text_area("Content Payload", item.get("text", ""), height=240, label_visibility="collapsed")
