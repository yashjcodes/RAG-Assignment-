"""Embedding back-ends. Both return L2-normalised vectors, so dot product = cosine."""
import numpy as np


class TfidfEmbedder:
    """Offline, dependency-light embedder (scikit-learn). Lexical, not semantic."""
    name = "tfidf"

    def fit(self, chunks):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2),
                                   sublinear_tf=True)
        self.vec.fit(chunks)
        return self

    def encode(self, texts):
        return self.vec.transform(texts).toarray().astype("float32")  # already L2-normed


class MiniLMEmbedder:
    """Semantic embedder (sentence-transformers). Needs internet on first run."""
    name = "minilm"

    def fit(self, chunks):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        return self

    def encode(self, texts):
        return np.asarray(self.model.encode(texts, normalize_embeddings=True),
                          dtype="float32")


def get_embedder(name: str):
    return {"tfidf": TfidfEmbedder, "minilm": MiniLMEmbedder}[name]()
