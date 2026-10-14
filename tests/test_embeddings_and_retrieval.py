import sys
import unittest
from pathlib import Path
import numpy as np

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from embeddings import get_embedder, TfidfEmbedder
from retrieval import VectorStore


class TestEmbeddingsAndRetrieval(unittest.TestCase):
    def test_get_embedder(self):
        embedder = get_embedder("tfidf")
        self.assertIsInstance(embedder, TfidfEmbedder)
        self.assertEqual(embedder.name, "tfidf")

        with self.assertRaises(KeyError):
            get_embedder("non_existent_embedder")

    def test_tfidf_fit_and_encode(self):
        chunks = [
            "Type 2 diabetes is a chronic metabolic condition.",
            "Metformin is the first-line oral medication.",
            "Symptoms include increased thirst and frequent urination.",
        ]
        embedder = TfidfEmbedder()
        embedder.fit(chunks)
        vecs = embedder.encode(chunks)

        self.assertEqual(vecs.shape[0], 3)
        self.assertEqual(vecs.dtype, np.float32)

        # Vector should be L2-normalised (magnitude ~ 1.0)
        norms = np.linalg.norm(vecs, axis=1)
        for norm in norms:
            self.assertAlmostEqual(norm, 1.0, places=4)

    def test_vector_store_retrieval(self):
        chunks = [
            "Apples, oranges, and bananas are fruits.",
            "Metformin and sulfonylureas treat diabetes.",
            "Python, Go, and Rust are programming languages.",
        ]
        embedder = get_embedder("tfidf")
        store = VectorStore(chunks, embedder)

        # Query about diabetes
        results = store.retrieve("What drugs treat diabetes?", k=2)
        self.assertEqual(len(results), 2)

        best_chunk_id, best_score, best_text = results[0]
        self.assertEqual(best_chunk_id, 1)
        self.assertIn("Metformin", best_text)
        self.assertGreater(best_score, 0.0)

    def test_vector_store_k_exceeding_chunks(self):
        chunks = ["Chunk one", "Chunk two"]
        embedder = get_embedder("tfidf")
        store = VectorStore(chunks, embedder)

        results = store.retrieve("Chunk", k=5)
        # Should return only the available chunks
        self.assertEqual(len(results), 2)


if __name__ == "__main__":
    unittest.main()
