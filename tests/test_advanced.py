"""Unit tests for Advanced Challenge components (Ingestion, Multimodal Retrieval, Agent)."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.advanced_agent import AdvancedBM25Index, AdvancedMultimodalRetriever
from src.advanced_ingest import _chunk_text, _to_markdown_table


class AdvancedTests(unittest.TestCase):
    def test_markdown_table_conversion(self):
        raw_rows = [
            ["Model", "MMLU", "HumanEval"],
            ["Llama 2 70B", "68.9", "29.9"],
            ["GPT-3.5", "70.0", "48.1"],
        ]
        md = _to_markdown_table(raw_rows)
        self.assertIn("| Model | MMLU | HumanEval |", md)
        self.assertIn("| Llama 2 70B | 68.9 | 29.9 |", md)

    def test_chunk_text_overlap(self):
        text = "Câu 1. Câu 2. Câu 3. Câu 4. Câu 5. Câu 6. Câu 7."
        chunks = _chunk_text(text, max_chars=20, overlap_chars=5)
        self.assertTrue(len(chunks) >= 2)
        for c in chunks:
            self.assertTrue(len(c) <= 30)

    def test_advanced_bm25_modality_filter(self):
        dummy_chunks = [
            {
                "id": "c1",
                "text": "Bảng số liệu accuracy trên benchmark MMLU",
                "page": 10,
                "metadata": {"modality": "table"},
            },
            {
                "id": "c2",
                "text": "Biểu đồ so sánh accuracy và loss theo epoch",
                "page": 12,
                "metadata": {"modality": "chart"},
            },
            {
                "id": "c3",
                "text": "Mô tả văn bản về phương pháp training và accuracy",
                "page": 15,
                "metadata": {"modality": "text"},
            },
        ]
        bm25 = AdvancedBM25Index(dummy_chunks)

        # Search without filter
        res_all = bm25.search("accuracy", top_k=5)
        self.assertEqual(len(res_all), 3)

        # Search with modality=table
        res_table = bm25.search("accuracy", top_k=5, modality="table")
        self.assertEqual(len(res_table), 1)
        self.assertEqual(res_table[0]["id"], "c1")

        # Search with modality=chart
        res_chart = bm25.search("accuracy", top_k=5, modality="chart")
        self.assertEqual(len(res_chart), 1)
        self.assertEqual(res_chart[0]["id"], "c2")

        # Search with page filter
        res_page = bm25.search("accuracy", top_k=5, page=15)
        self.assertEqual(len(res_page), 1)
        self.assertEqual(res_page[0]["id"], "c3")


if __name__ == "__main__":
    unittest.main()

