"""
Indexes chunks into ChromaDB with a lightweight MiniLM model (384-dim, ONNX CPU).
Supports indexing individual strategies into dedicated vector collections, or indexing all strategies.

Usage:
    # Index standard hierarchical children
    python -m scripts.run_indexing --strategy hierarchical_children

    # Index a specific strategy (e.g. recursive, semantic, markdown, fixed)
    python -m scripts.run_indexing --strategy recursive
    python -m scripts.run_indexing --strategy semantic

    # Index ALL chunking strategies into their own respective collections
    python -m scripts.run_indexing --strategy all
"""

import argparse
from pathlib import Path
from typing import List

from config.settings import CHUNKS_DIR
from src.indexing.embedder import build_embedder
from src.indexing.loader import load_chunk_file
from src.indexing.vector_store import get_client, get_collection, index_chunks

AVAILABLE_STRATEGIES = [
    "hierarchical_children",
    "hierarchical_parents",
    "recursive",
    "semantic",
    "token",
    "markdown",
    "fixed",
]

SMOKE_TEST_QUERIES = [
    "What is a balanced chemical equation?",
    "How is Plaster of Paris made from gypsum?",
    "Which metals react with cold water?",
]


def index_strategy(strategy: str, embedder):
    folder = CHUNKS_DIR / strategy
    if not folder.exists():
        print(f"[!] Folder not found: {folder}")
        return

    jsonl_files = sorted(folder.glob("*.jsonl"))
    if not jsonl_files:
        print(f"[!] No .jsonl files in {folder}")
        return

    collection = get_collection(strategy)
    print(f"\n========================================================")
    print(f"Indexing Strategy: '{strategy}' into Collection: '{collection.name}'")
    print(f"Total Chapter Files: {len(jsonl_files)}")
    print(f"========================================================")

    total_chunks = 0
    for file_path in jsonl_files:
        chunks = load_chunk_file(file_path)
        if not chunks:
            continue
        index_chunks(collection, embedder, chunks)
        total_chunks += len(chunks)
        print(f"  [+] Indexed {file_path.stem}: {len(chunks)} chunks")

    print(f"\n[OK] Finished '{strategy}'. Total Vectors in collection: {collection.count()}")

    # Smoke test query
    print(f"\n--- Smoke Test Retrieval on '{strategy}' ---")
    for q in SMOKE_TEST_QUERIES[:1]:
        q_vec = embedder.embed_query(q)
        res = collection.query(query_embeddings=[q_vec], n_results=1, include=["documents", "metadatas", "distances"])
        if res["documents"] and res["documents"][0]:
            doc_preview = res["documents"][0][0].replace("\n", " ")[:140]
            dist = res["distances"][0][0] if "distances" in res and res["distances"] else 0.0
            print(f"  Q: '{q}'")
            print(f"  Match (cosine dist: {dist:.3f}): {doc_preview}...")


def main():
    parser = argparse.ArgumentParser(description="Lightweight NCERT Vector Indexing")
    parser.add_argument(
        "--strategy",
        type=str,
        default="hierarchical_children",
        help=f"Chunking strategy to index: {AVAILABLE_STRATEGIES} or 'all'",
    )
    args = parser.parse_args()

    print("[*] Initializing lightweight MiniLM embedder (fastembed / ONNX)...")
    embedder = build_embedder()

    if args.strategy == "all":
        for strat in AVAILABLE_STRATEGIES:
            index_strategy(strat, embedder)
    else:
        if args.strategy not in AVAILABLE_STRATEGIES:
            print(f"[!] Unknown strategy '{args.strategy}'. Choose from: {AVAILABLE_STRATEGIES} or 'all'")
            return
        index_strategy(args.strategy, embedder)

    # Print summary of all existing collections
    client = get_client()
    collections = client.list_collections()
    print("\n" + "=" * 60)
    print("                CURRENT VECTOR COLLECTIONS")
    print("=" * 60)
    for col in collections:
        print(f"  - {col.name:<40} (Count: {col.count()} vectors)")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
