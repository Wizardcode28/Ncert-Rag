# Multimodal NCERT Science Retrieval-Augmented Generation (RAG) System

A domain-adapted, production-grade Retrieval-Augmented Generation (RAG) pipeline designed for textbook question answering (NCERT Science, Classes 6–10/12). 

This project implements an end-to-end framework featuring **PDF layout parsing**, **comparative chunking strategies**, **hybrid dense-lexical indexing**, **two-stage retrieval with cross-encoder re-ranking**, **parent-child context expansion**, and **multi-gate hallucination guardrails**.

---

## 1. Architectural Overview & End-to-End Workflow

The system bridges raw textbook PDFs to factually grounded student tutoring answers through an asynchronous, modular pipeline:

```
[Raw Textbook PDFs]
        │
        ▼ (Document Parsing & Structure Extraction)
[Parsed JSON: Structured Chapters, Headings, Activities, Tables]
        │
        ▼ (Chunking Strategies Evaluation)
[Document Chunks: Fixed, Token, Recursive, Semantic, Markdown, Hierarchical]
        │
   ┌────┴─────────────────────────────┐
   │                                  │
   ▼ (Embedding Generation)           ▼ (Metadata & Inverted Index)
[Dense Embeddings: ONNX FastEmbed]  [Lexical Corpus / BM25 Inverted Index]
   │                                  │
   └───────────────┬──────────────────┘
                   ▼
       [Persistent Vector Store (ChromaDB)]
                   │
═══════════════════╪══════════════════════════════════════════════════════════
               RUNTIME QUERY WORKFLOW
═══════════════════╪══════════════════════════════════════════════════════════
                   │
           [Student Query]
                   │
   ┌───────────────┴───────────────┐
   ▼                               ▼
[BM25 Lexical Search]     [Dense Cosine Search]
(Class/Content Filter)    (all-MiniLM-L6-v2)
   │                               │
   └───────────────┬───────────────┘
                   ▼
 [Ensemble Hybrid Fusion (Weighted RRF, c=60)]
                   │ (Candidate Child Chunks)
                   ▼
 [Cross-Encoder Re-Ranker (ms-marco-TinyBERT / MiniLM)]
                   │
                   ▼ (Deduplicate & Map via parent_id)
 [Parent Context Expansion (Complete Conceptual Sections)]
                   │
                   ▼
      ┌─────────────────────────┐
      │  Guardrail Gate 1:      │ ──[Below REFUSAL_SCORE?]──► Refusal Response
      │  Relevance Threshold    │
      └────────────┬────────────┘
                   │
      ┌─────────────────────────┐
      │  Passage Filtering:     │ ──[Keep Passages >= MIN_PASSAGE_SCORE]
      │  Prune Weak Context     │
      └────────────┬────────────┘
                   │
                   ▼
 [Prompt Assembly: Numbered Grounded Excerpts + Guardrail Prompt]
                   │
                   ▼
 [LLM Generation: Groq / Gemini / OpenRouter Fallback Cascade]
                   │
      ┌─────────────────────────┐
      │  Guardrail Gate 2:      │ ──[Returns INSUFFICIENT_CONTEXT?]──► Refusal
      │  Model Abstention Check │
      └────────────┬────────────┘
                   │
      ┌─────────────────────────┐
      │  Guardrail Gate 3:      │ ──[No Valid Citations?]──► Retry / Refusal
      │  Citation Verification  │
      └────────────┬────────────┘
                   │
                   ▼
 [Final Structured Answer with Verified Metadata Citations & Extra Knowledge]
```

---

## 2. Document Parsing & Structure Preservation

Before chunking, raw NCERT PDFs are parsed into clean JSON representations capturing domain metadata:
- **Layout & Structure Extraction**: Strips headers, footers, and page artifacts while detecting hierarchical heading levels (`H1` to `H5`), activities, exercises, tables, and captions.
- **Content Classification**: Content blocks are classified into specific types (`concept`, `activity`, `exercise`, `table`, `summary`). This enables deterministic filtering during retrieval (e.g., preventing exercises from being retrieved as factual answers for scientific concepts).
- **Metadata Tagging**: Every section retains document ancestry (`class_num`, `subject`, `chapter_num`, `chapter_name`, `section_number`, `section_title`, and `activity_label`).

---

## 3. Chunking Strategies Implementation & Analysis

The project implements and benchmarks six distinct chunking strategies (`src/chunking/`):

### 3.1. Fixed-Character Chunking (`fixed.py`)
- **Mechanism**: Splits document text into fixed-length character windows (e.g., 500 characters) with a predefined sliding overlap (e.g., 50 characters).
- **Characteristics**: Fast and uniform; however, agnostic to syntactic structure, frequently bisecting words, chemical formulas, and sentences.

### 3.2. Recursive Character Chunking (`recursive.py`)
- **Mechanism**: Uses a hierarchical delimiter cascade (`["\n\n", "\n", ". ", " ", ""]`) to recursively break text down until chunks satisfy the target size (e.g., 500–1000 chars) with specified overlap.
- **Characteristics**: Preserves paragraph and sentence boundaries wherever possible, providing balanced general-purpose coherence.

### 3.3. Token-Based Chunking (`token_based.py`)
- **Mechanism**: Uses Byte Pair Encoding (BPE) via `tiktoken` (`cl100k_base` tokenizer) to measure chunk sizes directly in LLM sub-tokens.
- **Characteristics**: Strictly prevents chunk truncation inside embedding models and LLM context windows by enforcing exact token bounds.

### 3.4. Semantic / Sentence Chunking (`semantic.py`)
- **Mechanism**: Splits documents along linguistic sentence boundaries (using regex/NLP boundary rules) and packs sentences sequentially until hitting token/character limits without tearing mid-sentence ideas.
- **Characteristics**: Ensures grammatical completeness of propositions, reducing mid-thought truncation.

### 3.5. Markdown / Heading Chunking (`markdown.py`)
- **Mechanism**: Inspects the hierarchical heading markers (`#`, `##`, `###`) extracted during parsing and segments content strictly by textbook sections and subtopics.
- **Characteristics**: High semantic cohesion corresponding directly to textbook topic boundaries; chunk lengths vary depending on section depth.

### 3.6. Hierarchical (Parent-Child) Chunking (`hierarchical.py`) — *Production Strategy*
- **Mechanism**:
  - **Parent Chunks**: Encompass full conceptual sections (1,000–3,000 characters) capturing overarching context, explanations, activities, and derivations.
  - **Child Chunks**: Small sub-passages (200–400 characters) generated within each parent section with sliding overlaps.
  - **Relational Linkage**: Each child chunk is explicitly linked via a deterministic `parent_id`.
- **Why this solves the core RAG dilemma**:
  - Small child chunks yield sharp, focused vector embeddings that maximize retrieval precision and semantic similarity matching.
  - Feeding tiny child chunks to the LLM causes context starvation (missing surrounding qualifiers or formulas). The system maps top-ranked child chunks back to their encompassing parent chunks, providing the LLM with full context.

---

## 4. Embedding & Indexing Pipeline

### 4.1. Embedding Model & Architecture
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors) executed via **FastEmbed** on **ONNX Runtime**.
- **Efficiency**: CPU-optimized execution, zero dependency on heavy PyTorch runtimes, sub-10ms vector generation, ~80MB memory footprint.

### 4.2. Contextual Pre-Embedding Construction
To prevent vector ambiguity where identical words appear in different contexts (e.g., "respiration" in plant vs. animal biology), chunks are formatted with structured contextual headers prior to vectorization:

```text
Class {class_num} {Subject} | Chapter {chapter_num}: {chapter_name} | Section {sec_num} {sec_title} | {activity_label}
{chunk_text}
```
*Note: The metadata header is injected into the text passed to the embedding model for indexing, while the raw chunk text is preserved for retrieval output.*

### 4.3. Vector Database Indexing (ChromaDB)
- **Engine**: ChromaDB PersistentClient (`chromadb.PersistentClient`) with HNSW index using **Cosine Similarity** (`"hnsw:space": "cosine"`).
- **Strategy-Dedicated Collections**: Indexed under isolated collections (`ncert_science_minilm_<strategy_name>`) facilitating clean benchmarking and ablation testing.
- **Metadata Payloads**: Every index entry stores rich metadata payloads:
  - `parent_id`, `chunk_id`, `strategy`
  - `class_num`, `subject`, `chapter_num`, `chapter_name`
  - `section_number`, `section_title`, `activity_label`, `content_type`
- **Idempotent Updates**: Pre-indexing hooks execute deterministic chapter deletions prior to batch ingestion (`batch_size=64`), preventing duplicate vector pollution.

---

## 5. Retrieval & Search Engine Architecture

The retrieval system employs a multi-stage search and re-ranking architecture (`src/retrieval/`):

### 5.1. First-Stage Hybrid Retrieval
1. **Dense Semantic Search (`dense_retriever.py`)**:
   - Queries are converted to 384-d vectors using FastEmbed.
   - Evaluates cosine distance across ChromaDB collections with metadata pre-filtering (e.g., `class_num`, `content_type in ["concept", "activity"]`).
   - Retrieves top-$K$ candidates (e.g., $K=8$).
2. **Lexical Sparse Search (`bm25_retriever.py`)**:
   - In-memory BM25 Okapi retriever indexed over the filtered document corpus.
   - Accurately captures exact terminology, scientific nomenclatures, and chemical names (e.g., *"Plaster of Paris"*, *"CaSO4·1/2H2O"*).
   - Retrieves top-$K$ candidates (e.g., $K=8$).
3. **Ensemble Reciprocal Rank Fusion (`fusion.py`)**:
   - Combines dense and sparse candidate sets using **Weighted Reciprocal Rank Fusion (RRF)**:
     $$RRF\_Score(d \in D) = \sum_{m \in \{Dense, BM25\}} w_m \cdot \frac{1}{c + r_m(d)}$$
   - Parameterized with standard constant $c = 60$ and balanced weights $w = (0.5, 0.5)$, with deduplication via `chunk_id`.

### 5.2. Second-Stage Cross-Encoder Re-Ranking (`reranker.py`)
- **Engine**: Cross-encoder scoring via **FlashRank** (`ms-marco-TinyBERT-L-2-v2` / `ms-marco-MiniLM-L-12-v2`).
- **Mechanism**: Evaluates the full question-passage pair jointly through cross-attention to compute deep semantic relevance scores, overcoming bi-encoder limitations.
- Keeps top-$N$ ranked child chunks ($N=8$).

### 5.3. Parent Context Expansion (`search.py`)
- Child chunks are deduplicated by their `parent_id`.
- The system loads the corresponding full parent sections from disk (`load_all_parents()`).
- Yields up to `MAX_PARENTS = 5` distinct parent sections, ordered by their constituent child chunk's re-ranking score.

---

## 6. Generation Pipeline & Hallucination Guardrails

The synthesis engine (`src/generation/`) ensures high fidelity to the source curriculum and strict adherence to factual evidence:

### 6.1. Tri-Level Guardrail Architecture
1. **Gate 1 — Relevance Threshold (`check_retrieval`)**:
   - If the top re-ranker score falls below `REFUSAL_SCORE` (0.10), retrieval fails immediately without incurring an LLM call.
   - Passages scoring below `MIN_PASSAGE_SCORE` (0.30) are dropped to prune noise.
2. **Gate 2 — Model Abstention (`INSUFFICIENT_CONTEXT`)**:
   - The system prompt instructs the LLM that if the provided excerpts lack the answer, it must return the single token `INSUFFICIENT_CONTEXT`.
   - When triggered, the system safely falls back to a standardized refusal message.
3. **Gate 3 — Citation Extraction & Verification**:
   - The LLM is constrained to cite excerpts using square bracket notation (e.g., `[1]`, `[2]`).
   - The pipeline extracts all citations via regular expressions (`extract_citation_numbers`).
   - If an answer fails to provide valid citations to the supplied excerpts, an automatic retry is triggered with an explicit reminder prompt. If the retry fails, the generation is rejected.

### 6.2. Two-Tier Answer Synthesis (`prompts.py`)
Prompts format the context with numbered headers (`[1] Class 10 Science, Chapter 1...`). The LLM constructs:
- **Textbook Answer**: Synthesized strictly from cited excerpts without external knowledge injection.
- **Extra Knowledge**: Optional supplementary enrichment clearly segregated from the core curriculum answer.

### 6.3. LLM Resilience & Fallback Cascade (`llm.py`)
- Employs LangChain Expression Language (LCEL) pipelines with deterministic temperature (`temperature = 0.0`).
- Implements an automated provider failover chain:
  $$\text{Primary: Groq} \longrightarrow \text{Fallback 1: Google Gemini} \longrightarrow \text{Fallback 2: OpenRouter}$$

---

## 7. Project Structure & Core Modules

```
├── config/
│   └── settings.py          # Central hyperparameter configuration & environment loading
├── data/
│   ├── parsed/              # Structured JSON extractions from raw PDFs
│   ├── chunks/              # Chunk outputs partitioned by strategy
│   ├── vectorstore/         # Persistent ChromaDB HNSW storage
│   └── models/              # Local ONNX cross-encoder model cache
├── scripts/
│   ├── run_ingestion.py     # PDF parsing & structure extraction pipeline
│   ├── compare_chunking.py  # Statistical & quantitative comparison of chunk strategies
│   ├── export_all_chunks.py # Chunk dataset export utility
│   ├── run_indexing.py      # Batch embedding & vectorstore population
│   ├── run_search.py        # CLI retrieval & re-ranking inspection tool
│   └── run_query.py         # End-to-end interactive RAG query interface
├── src/
│   ├── parsing/             # PDF text, table, and layout parsing routines
│   ├── chunking/            # Implementations of the 6 chunking algorithms
│   │   ├── fixed.py
│   │   ├── recursive.py
│   │   ├── token_based.py
│   │   ├── semantic.py
│   │   ├── markdown.py
│   │   └── hierarchical.py
│   ├── indexing/            # FastEmbed embedding client & ChromaDB wrappers
│   ├── retrieval/           # Hybrid search, BM25, Dense, RRF fusion & FlashRank re-ranking
│   └── generation/          # LCEL chains, prompt templates, multi-gate guardrails
├── chunking.ipynb           # Exploratory data analysis & chunking distribution notebook
└── requirements.txt         # Core dependencies (chromadb, fastembed, flashrank, langchain)
```

---

## 8. Setup & Execution

### 8.1. Environment Setup
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
# Or bash: source .venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt
```

### 8.2. Environment Variables Configuration
Create a `.env` file in the root directory:
```env
LLM_PROVIDERS=groq,gemini,openrouter
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
OPENROUTER_API_KEY=your_openrouter_api_key
```

### 8.3. Pipeline Execution Lifecycle

1. **Ingestion & Layout Parsing**:
   ```bash
   python -m scripts.run_ingestion
   ```
2. **Chunk Generation & Strategy Comparison**:
   ```bash
   python -m scripts.compare_chunking
   ```
3. **Vector Store Indexing**:
   ```bash
   python -m scripts.run_indexing
   ```
4. **Interactive RAG Query Session**:
   ```bash
   # Interactive CLI session
   python -m scripts.run_query

   # Single query evaluation
   python -m scripts.run_query "Why must chemical equations be balanced?"
   ```
