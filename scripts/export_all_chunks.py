"""
Exports all chunking strategies (fixed, recursive, token_based, semantic, markdown, hierarchical)
to organized JSONL files in data/chunks/ organized by strategy subfolders.

Usage:
    python -m scripts.export_all_chunks
    python -m scripts.export_all_chunks --file class10_science_ch01.json
"""

import argparse
import json
from pathlib import Path
from typing import List

from config.settings import PARSED_DIR, CHUNKS_DIR
from src.parsing.schemas import ParsedChapter
from src.chunking import (
    split_fixed_size,
    split_recursive,
    split_token_based,
    split_semantic_sentences,
    split_markdown_sections,
    split_hierarchical,
)


def write_jsonl(path: Path, items: List[any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for item in items:
            # item.model_dump_json() outputs complete metadata & text
            f.write(item.model_dump_json() + "\n")


def process_chapter(json_path: Path):
    stem = json_path.stem
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chapter = ParsedChapter(**data)

    print(f"[-] Processing {stem} (Class {chapter.class_num} Science Ch {chapter.chapter_num})...")

    # 1. Fixed-size chunking
    fixed_chunks = split_fixed_size(chapter, chunk_size=500, chunk_overlap=50)
    write_jsonl(CHUNKS_DIR / "fixed" / f"{stem}_fixed.jsonl", fixed_chunks)

    # 2. Recursive character chunking
    rec_chunks = split_recursive(chapter, chunk_size=800, chunk_overlap=100)
    write_jsonl(CHUNKS_DIR / "recursive" / f"{stem}_recursive.jsonl", rec_chunks)

    # 3. Token-based chunking
    tok_chunks = split_token_based(chapter, chunk_size=250, chunk_overlap=25)
    write_jsonl(CHUNKS_DIR / "token" / f"{stem}_token.jsonl", tok_chunks)

    # 4. Semantic / sentence chunking
    sem_chunks = split_semantic_sentences(chapter, target_chunk_chars=800)
    write_jsonl(CHUNKS_DIR / "semantic" / f"{stem}_semantic.jsonl", sem_chunks)

    # 5. Markdown section chunking
    md_chunks = split_markdown_sections(chapter, max_chunk_size=1200)
    write_jsonl(CHUNKS_DIR / "markdown" / f"{stem}_markdown.jsonl", md_chunks)

    # 6. Hierarchical (Parents & Children)
    parents, children = split_hierarchical(chapter)
    write_jsonl(CHUNKS_DIR / "hierarchical" / f"{stem}_parents.jsonl", parents)
    write_jsonl(CHUNKS_DIR / "hierarchical" / f"{stem}_children.jsonl", children)

    print(
        f"    Saved: fixed={len(fixed_chunks)}, recursive={len(rec_chunks)}, "
        f"token={len(tok_chunks)}, semantic={len(sem_chunks)}, md={len(md_chunks)}, "
        f"hierarchical=(P:{len(parents)}, C:{len(children)})"
    )


def main():
    parser = argparse.ArgumentParser(description="Export all chunking strategies to data/chunks/")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific parsed chapter file (e.g. class10_science_ch01.json). If omitted, processes all 64 chapters.",
    )
    args = parser.parse_args()

    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

    if args.file:
        target = PARSED_DIR / args.file
        if not target.exists():
            print(f"Error: {target} does not exist.")
            return
        files_to_process = [target]
    else:
        files_to_process = sorted(PARSED_DIR.glob("*.json"))

    print(f"\n[+] Starting chunk export for {len(files_to_process)} chapter(s) into {CHUNKS_DIR}\n")
    for f in files_to_process:
        process_chapter(f)

    print("\n[+] All chunking strategies exported successfully!")


if __name__ == "__main__":
    main()
