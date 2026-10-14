"""Minimal NumPy cosine-similarity vector store."""
import numpy as np


class VectorStore:
    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder.fit(chunks)
        self.matrix = self.embedder.encode(chunks)

    def retrieve(self, query: str, k: int = 3):
        q = self.embedder.encode([query])[0]
        scores = self.matrix @ q
        idx = np.argsort(-scores)[:k]
        return [(int(i), float(scores[i]), self.chunks[i]) for i in idx]
