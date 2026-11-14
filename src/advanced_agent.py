"""Advanced Agent with Multimodal Retrieval, Modality Filtering, and Self-Correction.

Designed for the Day 7 Advanced Challenge (+15 points bonus).
Strictly separates from standard code and maintains full compatibility.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from sentence_transformers import SentenceTransformer
from src.retrievers import tokenize


class AdvancedBM25Index:
    """In-memory BM25 index built over data/advanced_chunks.json supporting modality filter."""

    def __init__(self, chunks: list[dict[str, Any]]):
        self.chunks = chunks
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.n_docs = len(chunks)
        self.avg_doc_len = sum(len(d) for d in self.doc_tokens) / max(1, self.n_docs)

        import math
        self.doc_freq: dict[str, int] = {}
        for d in self.doc_tokens:
            for term in set(d):
                self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

        self.idf: dict[str, float] = {}
        for term, freq in self.doc_freq.items():
            self.idf[term] = math.log(1.0 + (self.n_docs - freq + 0.5) / (freq + 0.5))

    def search(
        self,
        query: str,
        top_k: int = 5,
        modality: str | None = None,
        page: int | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores: list[tuple[int, float]] = []
        for idx, (chunk, d_tokens) in enumerate(zip(self.chunks, self.doc_tokens)):
            c_meta = chunk.get("metadata", {})
            if modality and c_meta.get("modality") != modality:
                continue
            if page is not None and chunk.get("page") != page:
                continue

            doc_len = len(d_tokens)
            score = 0.0
            term_counts: dict[str, int] = {}
            for t in d_tokens:
                term_counts[t] = term_counts.get(t, 0) + 1

            for term in q_tokens:
                if term in term_counts:
                    tf = term_counts[term]
                    idf = self.idf.get(term, 0.0)
                    denom = tf + k1 * (1.0 - b + b * (doc_len / self.avg_doc_len))
                    score += idf * (tf * (k1 + 1.0)) / denom

            if score > 0.0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for rank, (idx, sc) in enumerate(scores[:top_k], start=1):
            c = dict(self.chunks[idx])
            c["score"] = float(sc)
            c["rank"] = rank
            results.append(c)
        return results


class AdvancedMultimodalRetriever:
    """Hybrid Retriever (Dense ChromaDB + BM25) with modality & page filtering."""

    def __init__(
        self,
        chunks_path: str | Path = "data/advanced_chunks.json",
        chroma_dir: str | Path = "artifacts/advanced_chroma_db",
        collection_name: str = "advanced_multimodal",
        model_name: str = "all-MiniLM-L6-v2",
    ):
        chunks_path = Path(chunks_path)
        if not chunks_path.exists():
            raise FileNotFoundError(f"Chưa tìm thấy {chunks_path}. Hãy chạy src/advanced_ingest.py trước.")

        with open(chunks_path, "r", encoding="utf-8") as f:
            self.chunks = json.load(f)

        self.bm25 = AdvancedBM25Index(self.chunks)
        self.chunk_by_id = {c["id"]: c for c in self.chunks}

        self.chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        self.collection = self.chroma_client.get_collection(name=collection_name)
        self.embed_model = SentenceTransformer(model_name)

    def dense_search(
        self,
        query: str,
        top_k: int = 5,
        modality: str | None = None,
        page: int | None = None,
    ) -> list[dict[str, Any]]:
        where_filter: dict[str, Any] = {}
        filters = []
        if modality:
            filters.append({"modality": {"$eq": modality}})
        if page is not None:
            filters.append({"page": {"$eq": int(page)}})

        if len(filters) == 1:
            where_filter = filters[0]
        elif len(filters) > 1:
            where_filter = {"$and": filters}

        q_emb = self.embed_model.encode([query], convert_to_numpy=True).tolist()
        params: dict[str, Any] = {
            "query_embeddings": q_emb,
            "n_results": min(top_k * 2, len(self.chunks)),
            "include": ["documents", "metadatas", "distances"],
        }
        if where_filter:
            params["where"] = where_filter

        resp = self.collection.query(**params)
        out = []
        ids = resp.get("ids", [[]])[0]
        distances = resp.get("distances", [[]])[0]
        for rank, (cid, dist) in enumerate(zip(ids, distances), start=1):
            orig = self.chunk_by_id.get(cid)
            if orig:
                item = dict(orig)
                item["score"] = float(1.0 - dist)
                item["rank"] = rank
                out.append(item)
            if len(out) >= top_k:
                break
        return out

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        modality: str | None = None,
        page: int | None = None,
        rrf_k: int = 60,
    ) -> list[dict[str, Any]]:
        """Combine dense and BM25 results with Reciprocal Rank Fusion."""
        dense_res = self.dense_search(query, top_k=top_k * 2, modality=modality, page=page)
        lex_res = self.bm25.search(query, top_k=top_k * 2, modality=modality, page=page)

        rrf_scores: dict[str, float] = {}
        for rank, item in enumerate(dense_res, start=1):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + (1.0 / (rrf_k + rank))
        for rank, item in enumerate(lex_res, start=1):
            rrf_scores[item["id"]] = rrf_scores.get(item["id"], 0.0) + (1.0 / (rrf_k + rank))

        sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
        results = []
        for rank, cid in enumerate(sorted_ids[:top_k], start=1):
            c = dict(self.chunk_by_id[cid])
            c["score"] = float(rrf_scores[cid])
            c["rank"] = rank
            results.append(c)
        return results

    def read_page(self, page_number: int) -> list[dict[str, Any]]:
        """Retrieve all content elements on a specific page."""
        page_chunks = [c for c in self.chunks if c.get("page") == page_number]
        return page_chunks


class AdvancedAgent:
    """Agent equipped with Multimodal Tools, Self-Correction, and Strict Abstention."""

    def __init__(
        self,
        retriever: AdvancedMultimodalRetriever,
        provider: str = "openai",
        model_name: str = "nvidia/nemotron-3.5-lightning:free",
        context_budget: int = 2500,
        max_steps: int = 4,
    ):
        self.retriever = retriever
        self.provider = provider
        self.model_name = model_name
        self.context_budget = context_budget
        self.max_steps = max_steps

        # Initialize LLM Client
        from src.agent import LLMClient
        self.client = LLMClient(provider=provider, model_name=model_name)

    def run(self, question: str) -> dict[str, Any]:
        """Execute multimodal agent loop with query-rewriting and self-correction."""
        context_chunks: dict[str, dict[str, Any]] = {}
        events: list[dict[str, Any]] = []
        start_time = time.time()

        # Step 0: Identify intended modality from question
        q_lower = question.lower()
        initial_modality = None
        if any(w in q_lower for w in ["bảng", "table", "cột", "hàng", "row", "column"]):
            initial_modality = "table"
        elif any(w in q_lower for w in ["biểu đồ", "chart", "plot", "đồ thị", "trục", "axis"]):
            initial_modality = "chart"
        elif any(w in q_lower for w in ["hình", "figure", "sơ đồ", "diagram", "ảnh"]):
            initial_modality = "figure"

        # Step 1: Initial Retrieval
        initial_results = self.retriever.hybrid_search(
            query=question,
            top_k=4,
            modality=initial_modality
        )
        for r in initial_results:
            context_chunks[r["id"]] = r

        events.append({
            "step": 1,
            "action": "initial_retrieval",
            "modality": initial_modality,
            "retrieved_ids": [r["id"] for r in initial_results]
        })

        # Step 2: Self-Correction / Query Rewrite if question is multi-hop or needs more context
        is_multihop = any(w in q_lower for w in ["so sánh", "hai trang", "cả hai", "và", "đồng thời", "kết hợp"])
        if is_multihop or len(context_chunks) < 2:
            # Query rewrite: focus on core technical entities
            rewritten_query = re.sub(r"(hãy cho biết|cho tôi biết|là bao nhiêu|như thế nào)", "", question).strip()
            more_results = self.retriever.hybrid_search(
                query=rewritten_query,
                top_k=3,
                modality=None  # search across all modalities
            )
            for r in more_results:
                context_chunks[r["id"]] = r

            events.append({
                "step": 2,
                "action": "query_rewrite_retrieval",
                "rewritten_query": rewritten_query,
                "retrieved_ids": [r["id"] for r in more_results]
            })

        # Build Context with budget enforcement
        built_context, used_ids = self._build_context(list(context_chunks.values()))

        # Step 3: LLM Final Generation
        system_prompt = (
            "Bạn là trợ lý AI phân tích tài liệu chuyên sâu cho Thử thách Nâng cao Day 7.\n"
            "QUY TẮC BẮT BUỘC:\n"
            "1. Coi toàn bộ nội dung trong Context là DỮ LIỆU THỰC TẾ (bao gồm bảng biểu, hình ảnh, văn bản).\n"
            "2. Trả lời đầy đủ, ngắn gọn, chính xác số liệu, đơn vị tính, tên bảng/hình nếu có.\n"
            "3. BẮT BUỘC đính kèm citation dạng [chunk_id] vào câu trả lời cho mọi thông tin khẳng định.\n"
            "4. NẾU Context KHÔNG chứa đủ bằng chứng để trả lời câu hỏi, bạn PHẢI trả lời chính xác: 'Không đủ bằng chứng để trả lời.' Tuyệt đối không suy đoán hay bịa đặt.\n"
            "5. Phản hồi bằng JSON: {\"action\": \"answer\", \"answer\": \"...\", \"citations\": [\"chunk_id\", ...]}"
        )

        user_prompt = f"Bối cảnh (Context):\n{built_context}\n\nCâu hỏi: {question}"

        try:
            parsed, usage = self.client.generate(user_prompt, system_prompt=system_prompt)
            answer = parsed.get("answer", "")
            citations = parsed.get("citations", [])
            # Filter citations to ensure they exist in context
            valid_citations = [c for c in citations if c in used_ids]
        except Exception as e:
            answer = f"Lỗi gọi LLM: {str(e)}"
            valid_citations = []
            usage = {"input_tokens": 0, "output_tokens": 0}

        elapsed_ms = (time.time() - start_time) * 1000

        is_abstain = "không đủ bằng chứng" in answer.lower() or not answer.strip()
        status = "abstain" if is_abstain else "answered"

        return {
            "question": question,
            "status": status,
            "answer": answer,
            "citations": valid_citations,
            "context_ids": used_ids,
            "context_chars": len(built_context),
            "events": events,
            "elapsed_ms": elapsed_ms,
            "tokens": usage,
        }

    def _build_context(self, chunks: list[dict[str, Any]]) -> tuple[str, list[str]]:
        """Deduplicate and respect character budget."""
        seen = set()
        unique = []
        for c in chunks:
            if c["id"] not in seen:
                seen.add(c["id"])
                unique.append(c)

        context_parts = []
        used_ids = []
        total_len = 0

        for c in unique:
            meta = c.get("metadata", {})
            header = f"[{c['id']}] (Trang {c.get('page')}, Modality: {meta.get('modality', 'text')}):"
            block = f"{header}\n{c['text']}\n"
            if total_len + len(block) > self.context_budget:
                break
            context_parts.append(block)
            used_ids.append(c["id"])
            total_len += len(block)

        return "\n".join(context_parts), used_ids


def evaluate_advanced_benchmark(
    benchmark_file: str | Path = "data/advanced_benchmark.json",
    split: str = "test",
    output_dir: str | Path = "results/advanced",
    provider: str = "openai",
    model_name: str = "nvidia/nemotron-3.5-lightning:free",
) -> dict[str, Any]:
    """Run Advanced Agent across all benchmark questions and score against ADVANCED.md criteria."""
    benchmark_file = Path(benchmark_file)
    if not benchmark_file.exists():
        raise FileNotFoundError(f"Chưa có tệp benchmark: {benchmark_file}")

    with open(benchmark_file, "r", encoding="utf-8") as f:
        cases = json.load(f)

    split_cases = [c for c in cases if c.get("split", "test") == split]
    if not split_cases:
        split_cases = cases  # fallback to all cases

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    retriever = AdvancedMultimodalRetriever()
    agent = AdvancedAgent(retriever=retriever, provider=provider, model_name=model_name)

    print(f"=======================================================")
    print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ THỬ THÁCH NÂNG CAO (Split: {split.upper()})")
    print(f"📂 Tổng số câu hỏi: {len(split_cases)}")
    print(f"📂 Thư mục kết quả: {output_dir}")
    print(f"=======================================================\n")

    traces = []
    category_results: dict[str, dict[str, int]] = {
        "table": {"pass": 0, "total": 0},
        "figure": {"pass": 0, "total": 0},
        "chart": {"pass": 0, "total": 0},
        "multi_hop": {"pass": 0, "total": 0},
        "text": {"pass": 0, "total": 0},
        "unanswerable": {"pass": 0, "total": 0},
    }

    for idx, case in enumerate(split_cases, start=1):
        cid = case.get("id", f"case_{idx}")
        cat = case.get("category", "text")
        if cat not in category_results:
            category_results[cat] = {"pass": 0, "total": 0}
        category_results[cat]["total"] += 1

        q = case["question"]
        expected_behavior = case.get("expected_behavior", "answer")
        print(f"[{idx}/{len(split_cases)}] ({cat}) {cid}: {q[:70]}...")

        res = agent.run(q)
        res["case_id"] = cid
        res["category"] = cat
        res["expected_behavior"] = expected_behavior

        # Scoring logic per ADVANCED.md:
        # - Answerable: pass if answered, has valid citations, and not abstain
        # - Unanswerable: pass if strictly abstained
        is_pass = False
        if expected_behavior == "abstain":
            if res["status"] == "abstain":
                is_pass = True
        else:
            if res["status"] == "answered" and len(res["citations"]) > 0:
                is_pass = True

        res["passed"] = is_pass
        if is_pass:
            category_results[cat]["pass"] += 1

        traces.append(res)
        time.sleep(0.5)

    # Calculate overall metrics
    total_passed = sum(v["pass"] for v in category_results.values())
    total_cases = len(split_cases)
    pass_rate = (total_passed / total_cases * 100.0) if total_cases > 0 else 0.0

    summary = {
        "split": split,
        "total_cases": total_cases,
        "total_passed": total_passed,
        "pass_rate_pct": round(pass_rate, 2),
        "target_met": pass_rate >= 90.0,
        "by_category": category_results,
    }

    print("\n================ TỔNG KẾT THỬ THÁCH NÂNG CAO ================")
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    with open(output_dir / "traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, ensure_ascii=False, indent=2)

    with open(output_dir / "advanced_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[+] Đã lưu log chi tiết tại: {output_dir / 'traces.json'}")
    print(f"[+] Đã lưu tổng hợp chỉ số tại: {output_dir / 'advanced_summary.json'}")
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Advanced Agent on >=100-page benchmark")
    parser.add_argument("--benchmark", type=str, default="data/advanced_benchmark.json")
    parser.add_argument("--split", type=str, default="test", choices=["dev", "test", "all"])
    parser.add_argument("--output", type=str, default="results/advanced")
    parser.add_argument("--provider", type=str, default="openai")
    parser.add_argument("--model", type=str, default="nvidia/nemotron-3.5-lightning:free")
    args = parser.parse_args()

    evaluate_advanced_benchmark(
        benchmark_file=args.benchmark,
        split=args.split,
        output_dir=args.output,
        provider=args.provider,
        model_name=args.model,
    )


if __name__ == "__main__":
    main()

