from typing import List
from fastembed import TextEmbedding

# Lightweight (384-dimensional) embedding model using ONNX Runtime
# Fast on CPU, uses ~80MB, no heavy deep learning frameworks needed
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class MiniLMEmbeddings:
    """Fast, lightweight MiniLM embedding client using fastembed/ONNX."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        self.model_name = model_name
        self.client = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = list(self.client.embed(texts))
        return [e.tolist() for e in embeddings]

    def embed_query(self, text: str) -> List[float]:
        embeddings = list(self.client.embed([text]))
        return embeddings[0].tolist()


def build_embedder() -> MiniLMEmbeddings:
    """Build and return the fast, lightweight MiniLM embedder."""
    return MiniLMEmbeddings()
