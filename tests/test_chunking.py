import sys
import unittest
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chunking import (
    naive_chunks,
    structure_chunks,
    _split_sections,
    _blocks,
    _split_oversized,
)


class TestChunking(unittest.TestCase):
    def test_naive_chunks_window_and_overlap(self):
        text = "a" * 1000
        # chunk_size=500, overlap=50 -> step=450
        # indices: 0..500, 450..950, 900..1000
        chunks = naive_chunks(text, chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(len(chunks[0]), 500)
        self.assertEqual(len(chunks[1]), 500)
        self.assertEqual(len(chunks[2]), 100)

    def test_naive_chunks_empty_and_whitespace(self):
        self.assertEqual(naive_chunks(""), [])
        self.assertEqual(naive_chunks("   \n\n  \t "), [])

    def test_naive_chunks_short_text(self):
        chunks = naive_chunks("Short text here", chunk_size=500, overlap=50)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], "Short text here")

    def test_split_sections_headers_and_hierarchy(self):
        doc = (
            "# Main Title\n"
            "Introductory text.\n\n"
            "## Section 1\n"
            "Section 1 details.\n\n"
            "### Sub 1.1\n"
            "Sub details.\n\n"
            "## Section 2\n"
            "Section 2 details."
        )
        sections = _split_sections(doc)
        self.assertEqual(len(sections), 4)

        # Main Title
        self.assertEqual(sections[0][0], "Main Title")
        self.assertEqual(sections[0][1], "Introductory text.")

        # Section 1
        self.assertEqual(sections[1][0], "Main Title > Section 1")
        self.assertEqual(sections[1][1], "Section 1 details.")

        # Sub 1.1
        self.assertEqual(sections[2][0], "Main Title > Section 1 > Sub 1.1")
        self.assertEqual(sections[2][1], "Sub details.")

        # Section 2 resets Sub 1.1
        self.assertEqual(sections[3][0], "Main Title > Section 2")
        self.assertEqual(sections[3][1], "Section 2 details.")

    def test_blocks_colon_gluing(self):
        body = (
            "Symptoms include:\n\n"
            "- Thirst\n"
            "- Hunger\n\n"
            "Next paragraph here."
        )
        blocks = _blocks(body)
        self.assertEqual(len(blocks), 2)
        self.assertTrue(blocks[0].startswith("Symptoms include:\n- Thirst"))
        self.assertEqual(blocks[1], "Next paragraph here.")

    def test_split_oversized(self):
        long_block = "First sentence here. Second sentence here. Third sentence here."
        limit = 30
        pieces = _split_oversized(long_block, limit=limit)
        self.assertGreater(len(pieces), 1)
        self.assertEqual(len(pieces), 3)

    def test_structure_chunks_prefix_and_packing(self):
        doc = (
            "## Overview\n"
            "This is paragraph one.\n\n"
            "This is paragraph two.\n\n"
            "## Management\n"
            "Manage with diet and exercise."
        )
        chunks = structure_chunks(doc, max_chars=100)
        self.assertGreater(len(chunks), 0)
        # Each chunk should have a breadcrumb header in brackets
        for c in chunks:
            self.assertTrue(c.startswith("["))
            self.assertIn("]\n", c)


if __name__ == "__main__":
    unittest.main()
