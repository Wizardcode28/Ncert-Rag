import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PARSED_DIR = PROJECT_ROOT / "data" / "parsed"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
VECTORSTORE_DIR = PROJECT_ROOT / "data" / "vectorstore"
MODELS_DIR = PROJECT_ROOT / "data" / "models"

# --- Retrieval ---
RERANKER_MODEL_NAME = "ms-marco-TinyBERT-L-2-v2"
ANSWER_CONTENT_TYPES = ("concept", "activity")
BM25_TOP_K = 8
DENSE_TOP_K = 8
HYBRID_WEIGHTS = (0.5, 0.5)
RERANK_TOP_N = 8
MAX_PARENTS = 5

# --- Generation (hosted LLM APIs) ---
LLM_TEMPERATURE = 0.0
LLM_MAX_RETRIES = 1

GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "openrouter/free")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# --- Guardrails ---
REFUSAL_SCORE = 0.10
MIN_PASSAGE_SCORE = 0.30
