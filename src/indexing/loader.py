import json
from pathlib import Path
from typing import List, Union

from src.chunking.schemas import Chunk, ChildChunk, ParentChunk


def load_chunk_file(path: Path) -> List[Union[Chunk, ChildChunk, ParentChunk]]:
    """Load chunks from any strategy jsonl file, validating against appropriate schema."""
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if "parent_id" in data and "chunk_id" in data:
                chunks.append(ChildChunk(**data))
            elif "parent_id" in data and "chunk_id" not in data:
                chunks.append(ParentChunk(**data))
            else:
                chunks.append(Chunk(**data))
    return chunks
