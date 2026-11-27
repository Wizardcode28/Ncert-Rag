"""
Chunking module offering multiple chunking strategies for NCERT textbooks:
- Fixed-size character chunking
- Recursive character text splitting
- Sentence-based splitting
- Hierarchical (Parent-Child) splitting
"""

from src.chunking.schemas import Chunk, ParentChunk, ChildChunk, ChunkMetadata
from src.chunking.fixed import split_fixed_size
from src.chunking.recursive import split_recursive
from src.chunking.sentence import split_sentence_chunks
from src.chunking.hierarchical import build_chunks as split_hierarchical

__all__ = [
    "Chunk",
    "ParentChunk",
    "ChildChunk",
    "ChunkMetadata",
    "split_fixed_size",
    "split_recursive",
    "split_sentence_chunks",
    "split_hierarchical",
]
