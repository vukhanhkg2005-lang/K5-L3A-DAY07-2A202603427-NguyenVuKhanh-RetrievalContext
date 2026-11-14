"""Unit tests for ContextManager and AgentEngine in src/agent.py."""
import unittest
from src.agent import ContextManager, AgentEngine


class MockSearchRetriever:
    def __init__(self, items):
        self.items = items

    def search(self, query: str, *, filters: dict | None = None, top_k: int = 3):
        return self.items[:top_k]


class AgentSystemTests(unittest.TestCase):
    def setUp(self):
        self.sample_chunks = [
            {
                "id": "c1",
                "text": "Đoạn văn 1 chứa thông tin A.",
                "source": "doc1.pdf",
                "page": 1,
                "score": 0.95
            },
            {
                "id": "c2",
                "text": "Đoạn văn 2 chứa thông tin B.",
                "source": "doc1.pdf",
                "page": 2,
                "score": 0.85
            }
        ]

    def test_context_manager_deduplication(self):
        ctx = ContextManager(budget=1000)
        # Add c1 twice and c2 once
        added1 = ctx.add_chunks([self.sample_chunks[0], self.sample_chunks[0]])
        self.assertEqual(len(added1), 1)
        added2 = ctx.add_chunks([self.sample_chunks[1]])
        self.assertEqual(len(added2), 1)
        self.assertEqual(len(ctx.chunks_pool), 2)

    def test_context_manager_budget(self):
        # Budget fits exactly 1 chunk (~60 chars) but not both (~120 chars)
        ctx = ContextManager(budget=80)
        ctx.add_chunks(self.sample_chunks)
        packed, ids = ctx.pack()
        self.assertLessEqual(len(packed), 80)
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids, ["c1"])

        # Extreme budget < 10 chars cannot fit any chunk header
        ctx_tiny = ContextManager(budget=5)
        ctx_tiny.add_chunks(self.sample_chunks)
        packed_tiny, ids_tiny = ctx_tiny.pack()
        self.assertEqual(ids_tiny, [])

    def test_agent_engine_mock_abstention_on_empty(self):
        empty_retriever = MockSearchRetriever([])
        engine = AgentEngine(dense_retriever=empty_retriever, max_searches=2)
        res = engine.run("Câu hỏi không có dữ liệu?")
        self.assertEqual(res["status"], "abstain")
        self.assertEqual(res["citations"], [])
        self.assertIn("Không đủ bằng chứng", res["answer"])

    def test_agent_engine_mock_answered(self):
        retriever = MockSearchRetriever(self.sample_chunks)
        engine = AgentEngine(dense_retriever=retriever, max_searches=2)
        res = engine.run("Câu hỏi có dữ liệu?")
        self.assertEqual(res["status"], "answered")
        self.assertEqual(res["citations"], ["c1"])
        self.assertIn("[c1]", res["answer"])
        self.assertEqual(res["search_count"], 1)


if __name__ == "__main__":
    unittest.main()
