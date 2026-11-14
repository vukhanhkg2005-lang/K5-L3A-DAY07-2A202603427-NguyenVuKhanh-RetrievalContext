"""Unit tests for retrieval methods in src/retrievers.py."""
import unittest
from src.retrievers import (
    tokenize,
    infer_metadata_filter,
    LexicalRetriever,
    HybridRetriever,
    RerankRetriever,
)


class MockRetriever:
    """Mock retriever returning deterministic chunk structures."""
    def __init__(self, items: list[dict]):
        self.items = items

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3) -> list[dict]:
        res = self.items
        if filters:
            res = [c for c in res if all(c.get("metadata", {}).get(k) == v for k, v in filters.items())]
        return res[:top_k]


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            {
                "id": "c1",
                "text": "Llama 2 sử dụng optimizer AdamW với beta1 0.9 và beta2 0.95 cho pretraining.",
                "source_id": "llama2_paper",
                "source": "data/llama2_paper.pdf",
                "page": 5,
                "metadata": {"section": "Pretraining", "page_number": 5}
            },
            {
                "id": "c2",
                "text": "Ghost Attention GAtt giúp duy trì chỉ dẫn hệ thống qua nhiều lượt hội thoại fine-tuning.",
                "source_id": "llama2_paper",
                "source": "data/llama2_paper.pdf",
                "page": 16,
                "metadata": {"section": "Fine-tuning", "page_number": 16}
            },
            {
                "id": "c3",
                "text": "Đánh giá an toàn safety sử dụng các bộ benchmark TruthfulQA và ToxiGen.",
                "source_id": "llama2_paper",
                "source": "data/llama2_paper.pdf",
                "page": 22,
                "metadata": {"section": "Safety", "page_number": 22}
            }
        ]

    def test_tokenize(self):
        toks = tokenize("Llama-2, Pretraining & AdamW!")
        self.assertIn("llama", toks)
        self.assertIn("2", toks)
        self.assertIn("pretraining", toks)
        self.assertIn("adamw", toks)

    def test_infer_metadata_filter(self):
        self.assertEqual(infer_metadata_filter("Siêu tham số cho AdamW pretraining"), {"section": "Pretraining"})
        self.assertEqual(infer_metadata_filter("Kỹ thuật RLHF và Reward Model"), {"section": "Fine-tuning"})
        self.assertEqual(infer_metadata_filter("Kiểm thử an toàn red teaming"), {"section": "Safety"})
        self.assertIsNone(infer_metadata_filter("Câu hỏi bình thường không chứa từ khóa"))

    def test_lexical_bm25_search_and_schema(self):
        retriever = LexicalRetriever(self.chunks)
        results = retriever.search("AdamW optimizer", top_k=2)
        self.assertEqual(len(results), 1)
        top = results[0]
        self.assertEqual(top["id"], "c1")
        self.assertGreater(top["score"], 0.0)
        # Verify schema
        for key in ("id", "text", "source_id", "source", "page", "metadata", "score", "strategy"):
            self.assertIn(key, top)

    def test_lexical_filter(self):
        retriever = LexicalRetriever(self.chunks)
        results = retriever.search("an toàn", filters={"section": "Pretraining"}, top_k=3)
        self.assertEqual(len(results), 0)
        results_safety = retriever.search("an toàn", filters={"section": "Safety"}, top_k=3)
        self.assertEqual(len(results_safety), 1)
        self.assertEqual(results_safety[0]["id"], "c3")

    def test_hybrid_rrf_and_schema(self):
        dense_mock = MockRetriever([
            {**self.chunks[0], "score": 0.85, "strategy": "dense"},
            {**self.chunks[1], "score": 0.70, "strategy": "dense"}
        ])
        lexical_mock = MockRetriever([
            {**self.chunks[1], "score": 5.2, "strategy": "bm25"},
            {**self.chunks[0], "score": 3.1, "strategy": "bm25"}
        ])
        hybrid = HybridRetriever(dense_mock, lexical_mock, fusion_mode="rrf", rrf_k=60)
        results = hybrid.search("test query", top_k=2)
        
        self.assertEqual(len(results), 2)
        # Both appear in both ranks (rank 1 and 2), score should be equal
        self.assertEqual(results[0]["strategy"], "hybrid_rrf")
        for r in results:
            self.assertIn("score", r)
            self.assertIn("metadata", r)

    def test_hybrid_score_fusion(self):
        dense_mock = MockRetriever([
            {**self.chunks[0], "score": 0.9, "strategy": "dense"},
            {**self.chunks[1], "score": 0.1, "strategy": "dense"}
        ])
        lexical_mock = MockRetriever([
            {**self.chunks[0], "score": 10.0, "strategy": "bm25"},
            {**self.chunks[1], "score": 1.0, "strategy": "bm25"}
        ])
        hybrid = HybridRetriever(dense_mock, lexical_mock, fusion_mode="score_fusion", alpha=0.5)
        results = hybrid.search("test query", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "c1")
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_rerank_retriever(self):
        base_mock = MockRetriever([
            {**self.chunks[1], "score": 0.6, "strategy": "base"},
            {**self.chunks[0], "score": 0.5, "strategy": "base"}
        ])
        reranker = RerankRetriever(base_mock)
        # Query specifically targeting c1
        results = reranker.search("AdamW beta1", top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["id"], "c1")
        self.assertEqual(results[0]["strategy"], "retrieval_rerank")


if __name__ == "__main__":
    unittest.main()

