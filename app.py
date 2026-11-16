import os
import streamlit as st

# Automatically sync st.secrets to os.environ at the earliest entry point
try:
    if hasattr(st, "secrets"):
        for k, v in st.secrets.items():
            if isinstance(v, str):
                os.environ[k] = v.strip()
except Exception:
    pass

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

# Professional Executive Theme CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #0F172A;
    }
    
    .app-header {
        border-bottom: 1px solid #E2E8F0;
        padding-bottom: 1.25rem;
        margin-bottom: 1.75rem;
    }
    .app-title {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.025em;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 0.95rem;
        color: #64748B;
        font-weight: 400;
    }
    .badge {
        display: inline-block;
        padding: 0.2rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 600;
        border-radius: 4px;
        background-color: #EEF2F6;
        color: #334155;
        border: 1px solid #CBD5E1;
        margin-right: 0.5rem;
    }
    
    .metric-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 2px 0 rgba(0, 0, 0, 0.04);
    }
    .metric-label {
        font-size: 0.78rem;
        font-weight: 600;
        text-transform: uppercase;
        color: #64748B;
        letter-spacing: 0.05em;
        margin-bottom: 0.35rem;
    }
    .metric-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #0F172A;
    }
    
    .panel-box {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 6px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    
    .passage-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 3px solid #2563EB;
        border-radius: 4px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    
    .passage-meta {
        font-size: 0.78rem;
        font-weight: 600;
        color: #475569;
        margin-bottom: 0.4rem;
        display: flex;
        justify-content: space-between;
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.5rem !important;
        font-weight: 700 !important;
        color: #0F172A !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        color: #64748B !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing retrieval services...")
def get_rag_components():
    embedder = build_embedder()
    client = get_client()
    try:
        llm = build_llm()
    except Exception as e:
        llm = None
    return embedder, llm, client


embedder, llm, client = get_rag_components()

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

# ==============================================================================
# MODULE 1: RAG Assistant
# ==============================================================================
if mode == "RAG Assistant":
    st.markdown(
        """
        <div class="app-header">
            <span class="badge">PRODUCTION PIPELINE</span>
            <span class="badge">NCERT CURRICULUM</span>
            <div class="app-title">NCERT Science Query Synthesis System</div>
            <div class="app-subtitle">Semantic search and grounded question-answering across indexed secondary science textbooks.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if llm is None:
        st.error(
            "No valid API keys detected in Streamlit Secrets. "
            "Please go to Streamlit Cloud -> Settings -> Secrets and add GROQ_API_KEY."
        )
        st.info(
            "Example Secrets format:\n\n"
            "```toml\n"
            'GROQ_API_KEY = "gsk_..."\n'
            'LLM_PROVIDERS = "groq"\n'
            "```"
        )
        st.stop()

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

                    st.markdown("---")
                    st.markdown("#### Retrieved Source Documents")
                    for idx, (doc, meta, dist) in enumerate(zip(docs, metas, dists), start=1):
                        similarity_pct = max(0.0, min(100.0, (1.0 - dist) * 100))
                        st.markdown(
                            f"""
                            <div class="passage-card">
                                <div class="passage-meta">
                                    <span>PASSAGE {idx} — Class {meta.get('class_num')} | Chapter {meta.get('chapter_num')}: {meta.get('chapter_name')}</span>
                                    <span>Relevance Confidence: {similarity_pct:.1f}% (Distance: {dist:.3f})</span>
                                </div>
                                <div>{doc}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

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

    # Compute strategies
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

    # Comparison Matrix
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

    # Detailed Inspection
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
