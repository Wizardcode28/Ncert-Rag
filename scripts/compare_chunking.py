"""
CLI tool to inspect, compare, and analyze chunking strategies on parsed NCERT chapters.
Outputs metrics (count, chars, words, min/max), metadata attributes, and sample chunk previews.

Usage:
    python -m scripts.compare_chunking
    python -m scripts.compare_chunking --file class10_science_ch01.json
    python -m scripts.compare_chunking --file class10_science_ch01.json --verbose
"""

import argparse
import json
from typing import Dict, Any, List

from config.settings import PARSED_DIR
from src.parsing.schemas import ParsedChapter
from src.chunking import (
    split_fixed_size,
    split_recursive,
    split_sentence_chunks,
    split_hierarchical,
)


def compute_metrics(name: str, chunks: List[Any]) -> Dict[str, Any]:
    count = len(chunks)
    if count == 0:
        return {
            "name": name,
            "count": 0,
            "avg_chars": 0,
            "min_chars": 0,
            "max_chars": 0,
            "avg_words": 0,
            "min_words": 0,
            "max_words": 0,
        }
    char_lens = [len(c.text) for c in chunks]
    word_lens = [len(c.text.split()) for c in chunks]
    return {
        "name": name,
        "count": count,
        "avg_chars": round(sum(char_lens) / count, 1),
        "min_chars": min(char_lens),
        "max_chars": max(char_lens),
        "avg_words": round(sum(word_lens) / count, 1),
        "min_words": min(word_lens),
        "max_words": max(word_lens),
    }


def print_chunk_details(label: str, chunk: Any, index: int = 0):
    cid = getattr(chunk, "chunk_id", getattr(chunk, "parent_id", f"chunk_{index}"))
    chars = len(chunk.text)
    words = len(chunk.text.split())
    lines = len(chunk.text.splitlines())
    
    print(f"\n{'='*75}")
    print(f"[{label}] Sample Chunk #{index + 1}")
    print(f"{'-'*75}")
    print(f"  • ID              : {cid}")
    if hasattr(chunk, "parent_id") and hasattr(chunk, "chunk_id"):
        print(f"  • Parent ID       : {chunk.parent_id}")
    print(f"  • Strategy        : {getattr(chunk, 'strategy', label)}")
    print(f"  • Class & Subject : Class {chunk.class_num} {chunk.subject.capitalize()} (Ch {chunk.chapter_num})")
    print(f"  • Chapter Name    : {chunk.chapter_name}")
    print(f"  • Section         : {chunk.section_number or 'N/A'} - {chunk.section_title or 'General'}")
    print(f"  • Content Type    : {chunk.content_type}")
    if getattr(chunk, "activity_label", ""):
        print(f"  • Activity        : {chunk.activity_label}")
    print(f"  • Length Metrics  : {chars} chars | {words} words | {lines} lines")
    print(f"{'-'*75}")
    print(f"  --- [COMPLETE CHUNK CONTENT START] ---\n{chunk.text}\n  --- [COMPLETE CHUNK CONTENT END] ---")
    print(f"{'='*75}")


def main():
    parser = argparse.ArgumentParser(description="NCERT Chunking Strategies Comparison & Inspection")
    parser.add_argument(
        "--file",
        type=str,
        default=None,
        help="Specific parsed json file in data/parsed (e.g. class10_science_ch01.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed metadata and inspection for top 3 chunks of each strategy",
    )
    args = parser.parse_args()

    parsed_files = sorted(PARSED_DIR.glob("*.json"))
    if not parsed_files:
        print(f"No parsed chapters found in {PARSED_DIR}.")
        return

    if args.file:
        target_path = PARSED_DIR / args.file
        if not target_path.exists():
            print(f"File {target_path} not found.")
            return
    else:
        target_path = parsed_files[0]

    print(f"\n[+] Loading Chapter File: {target_path.name}")
    with open(target_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    chapter = ParsedChapter(**data)
    print(f"[+] Loaded: Class {chapter.class_num} {chapter.subject.capitalize()} Ch {chapter.chapter_num} (Total Pages: {len(chapter.pages)})\n")

    # Execute all chunking strategies
    fixed_chunks = split_fixed_size(chapter, chunk_size=500, chunk_overlap=50)
    rec_chunks = split_recursive(chapter, chunk_size=800, chunk_overlap=100)
    sent_chunks = split_sentence_chunks(chapter, target_chunk_chars=800)
    parents, children = split_hierarchical(chapter)

    strategies = [
        ("Fixed Character (500c)", fixed_chunks),
        ("Recursive Character (800c)", rec_chunks),
        ("Sentence-Based Split", sent_chunks),
        ("Hierarchical (Parents)", parents),
        ("Hierarchical (Children)", children),
    ]

    # Display comparison table
    print("\n" + "="*88)
    print("                              CHUNKING STRATEGIES COMPARISON")
    print("="*88)
    row_fmt = "{:<28} | {:<7} | {:<10} | {:<18} | {:<15}"
    print(row_fmt.format("Strategy", "Count", "Avg Chars", "Min / Max Chars", "Avg Words (Min/Max)"))
    print("-" * 88)
    for name, chunk_list in strategies:
        m = compute_metrics(name, chunk_list)
        char_range = f"{m['min_chars']} - {m['max_chars']}"
        word_info = f"{m['avg_words']} ({m['min_words']}-{m['max_words']})"
        print(row_fmt.format(name, m["count"], m["avg_chars"], char_range, word_info))
    print("="*88 + "\n")

    # Detailed inspection of chunks & attributes
    inspect_limit = 3 if args.verbose else 1
    print(f"\n--- Detailed Chunk Attributes & Preview ({'Top 3 per strategy' if args.verbose else 'First chunk per strategy'}) ---")
    for name, chunk_list in strategies:
        if chunk_list:
            for idx in range(min(inspect_limit, len(chunk_list))):
                print_chunk_details(name, chunk_list[idx], index=idx)


if __name__ == "__main__":
    main()
