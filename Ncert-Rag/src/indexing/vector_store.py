from typing import Any, Union
import chromadb
from chromadb.config import Settings
from tqdm import tqdm

from config.settings import VECTORSTORE_DIR
from src.chunking.schemas import Chunk, ChildChunk, ParentChunk

DEFAULT_COLLECTION_PREFIX = "ncert_science_minilm"


def get_client() -> chromadb.PersistentClient:
    """Returns persistent ChromaDB client."""
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(VECTORSTORE_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def get_collection(strategy_name: str = "hierarchical_children"):
    """
    Open or create a dedicated persistent Chroma collection for a specific chunking strategy.
    Collection name convention: ncert_science_minilm_<strategy_name>
    """
    client = get_client()
    collection_name = f"{DEFAULT_COLLECTION_PREFIX}_{strategy_name}"
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine", "strategy": strategy_name},
    )


def build_embedding_text(chunk: Union[Chunk, ChildChunk, ParentChunk]) -> str:
    """Context header + chunk text for high-quality embedding representation."""
    parts = [
        f"Class {chunk.class_num} {chunk.subject.title()}",
        f"Chapter {chunk.chapter_num}: {chunk.chapter_name}",
    ]
    sec_num = getattr(chunk, "section_number", "")
    sec_title = getattr(chunk, "section_title", "")
    act_label = getattr(chunk, "activity_label", "")

    if sec_num != "":
        parts.append(f"Section {sec_num} {sec_title}")
    elif sec_title != "":
        parts.append(sec_title)
    if act_label != "":
        parts.append(act_label)

    header = " | ".join(parts)
    return header + "\n" + chunk.text


def chunk_to_metadata(chunk: Union[Chunk, ChildChunk, ParentChunk]) -> dict:
    """Everything except text becomes searchable/filterable metadata."""
    meta = chunk.model_dump(exclude={"text"})
    # Ensure strategy is always populated in metadata
    if "strategy" not in meta or not meta["strategy"]:
        meta["strategy"] = getattr(chunk, "strategy", "unknown") or "unknown"
    return meta


def delete_chapter(collection, class_num: int, subject: str, chapter_num: int):
    """Remove existing chapter vectors before re-indexing."""
    try:
        collection.delete(
            where={
                "$and": [
                    {"class_num": class_num},
                    {"subject": subject},
                    {"chapter_num": chapter_num},
                ]
            }
        )
    except Exception:
        pass


def index_chunks(collection, embedder, chunks, batch_size: int = 64):
    """Embed and index a list of chunks into the designated collection."""
    if not chunks:
        return

    first = chunks[0]
    delete_chapter(collection, first.class_num, first.subject, first.chapter_num)

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        ids = [getattr(c, "chunk_id", getattr(c, "parent_id", f"chunk_{idx}")) for idx, c in enumerate(batch, start=i)]
        documents = [c.text for c in batch]
        metadatas = [chunk_to_metadata(c) for c in batch]
        embed_texts = [build_embedding_text(c) for c in batch]
        embeddings = embedder.embed_documents(embed_texts)

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
