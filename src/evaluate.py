"""Comprehensive Evaluation Pipeline for Day 7:
Evaluates Retrieval Methods (Dense, Lexical BM25, Hybrid, Rerank) and Agent Configurations.
Adheres strictly to BENCHMARK.md and RUBRIC.md.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

# Ensure workspace root is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrievers import (
    BaseRetriever,
    DenseRetriever,
    LexicalRetriever,
    HybridRetriever,
    RerankRetriever,
    infer_metadata_filter,
)
from src.agent import AgentEngine, ContextManager, LLMClient


def hit(chunk: dict[str, Any], gold: list[dict[str, Any]]) -> bool:
    """Check if chunk matches any alternative gold evidence quote, source, and page."""
    if not gold:
        return False
    return any(
        chunk.get("source_id") == g.get("source_id")
        and chunk.get("page") == g.get("page")
        and g.get("evidence") in chunk.get("text", "")
        for g in gold
    )


def pack_context(results: list[dict[str, Any]], budget: int = 1200) -> tuple[str, list[str]]:
    """Context packing adhering to character budget."""
    blocks, ids, seen = [], [], set()
    for c in results:
        cid = c["id"]
        if cid in seen:
            continue
        seen.add(cid)
        source = c.get("source", "doc")
        page = c.get("page", 1)
        text = c.get("text", "")
        block = f"[{cid} | {source} | page {page}]\n{text}"
        if len("\n\n".join(blocks + [block])) <= budget:
            blocks.append(block)
            ids.append(cid)
        elif not blocks:
            rem = budget - len(f"[{cid} | {source} | page {page}]\n") - 10
            if rem > 50:
                blocks.append(f"[{cid} | {source} | page {page}]\n{text[:rem]}...")
                ids.append(cid)
            break
    return "\n\n".join(blocks), ids


def evaluate_retriever(
    retriever: BaseRetriever,
    questions: list[dict[str, Any]],
    strategy_name: str,
    top_k: int = 3,
    budget: int = 1200,
    enable_filter: bool = False
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate a retriever on answerable benchmark questions."""
    traces = []
    
    # Filter answerable questions for retrieval metrics (BENCHMARK.md: exclude unanswerable from retrieval denominator)
    answerable_qs = [q for q in questions if q.get("answerable", True) is not False]
    
    for q in answerable_qs:
        qid = q["id"]
        question_text = q["question"]
        gold = q.get("gold", [])
        
        # Determine filter if enabled (query inferred, NEVER from gold)
        filters = None
        if enable_filter:
            # User provided filter or inferred from question
            filters = q.get("filter") or infer_metadata_filter(question_text)

        start = time.perf_counter()
        results = retriever.search(question_text, filters=filters, top_k=top_k)
        context, context_ids = pack_context(results, budget=budget)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        
        # Mark hits
        marked_results = [{**c, "hit": hit(c, gold)} for c in results]
        
        first_hit_rank = None
        for rank, c in enumerate(marked_results, start=1):
            if c["hit"] and first_hit_rank is None:
                first_hit_rank = rank
                
        hit1 = bool(marked_results and marked_results[0]["hit"])
        hit3 = any(c["hit"] for c in marked_results[:3])
        context_hit = any(c["hit"] and c["id"] in context_ids for c in marked_results)
        
        traces.append({
            "query_id": qid,
            "split": q.get("split"),
            "question": question_text,
            "strategy": strategy_name,
            "top_k": top_k,
            "budget_chars": budget,
            "filters_used": filters,
            "gold": gold,
            "results": marked_results,
            "context": context,
            "context_ids": context_ids,
            "context_chars": len(context),
            "elapsed_ms": elapsed_ms,
            "hit1": hit1,
            "hit3": hit3,
            "first_hit_rank": first_hit_rank,
            "mrr": round(1.0 / first_hit_rank, 4) if first_hit_rank else 0.0,
            "context_hit": context_hit
        })

    n = len(traces)
    hit1_count = sum(t["hit1"] for t in traces)
    hit3_count = sum(t["hit3"] for t in traces)
    ctx_hit_count = sum(t["context_hit"] for t in traces)
    mrr_sum = sum(t["mrr"] for t in traces)
    latencies = [t["elapsed_ms"] for t in traces] or [0]

    summary = {
        "strategy": strategy_name,
        "split": questions[0].get("split") if questions else "unknown",
        "n_answerable": n,
        "hit1": f"{hit1_count}/{n} ({hit1_count / max(1, n):.2%})",
        "hit3": f"{hit3_count}/{n} ({hit3_count / max(1, n):.2%})",
        "context_evidence_rate": f"{ctx_hit_count}/{n} ({ctx_hit_count / max(1, n):.2%})",
        "mrr_at_3": round(mrr_sum / max(1, n), 4),
        "hit1_count": hit1_count,
        "hit3_count": hit3_count,
        "context_hit_count": ctx_hit_count,
        "avg_context_chars": round(statistics.mean(t["context_chars"] for t in traces), 1) if traces else 0,
        "median_ms": round(statistics.median(latencies), 2),
        "avg_ms": round(statistics.mean(latencies), 2),
        "estimated_cost_usd": 0.0
    }
    return summary, traces


def evaluate_agent(
    agent_engine: AgentEngine,
    questions: list[dict[str, Any]],
    config_name: str
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Evaluate Agent on all benchmark questions including unanswerable cases."""
    traces = []
    
    total_tool_calls = 0
    questions_with_tools = 0
    abstentions_correct = 0
    total_unanswerable = 0
    false_abstentions = 0
    total_answerable = 0
    total_citations = 0
    valid_citations = 0
    answers_with_citations = 0
    total_answered = 0
    total_cost_usd = 0.0
    latencies = []

    for q in questions:
        qid = q["id"]
        q_text = q["question"]
        is_answerable = q.get("answerable", True)
        gold = q.get("gold", [])
        user_filter = q.get("filter")

        trace = agent_engine.run(q_text, user_filter=user_filter)
        trace["query_id"] = qid
        trace["split"] = q.get("split")
        trace["is_answerable"] = is_answerable
        trace["expected_behavior"] = q.get("expected_behavior")
        trace["gold"] = gold
        
        status = trace["status"]
        cits = trace["citations"]
        search_cnt = trace["search_count"]
        elapsed = trace["elapsed_ms"]
        cost = trace.get("cost_usd", 0.0)
        
        total_tool_calls += search_cnt
        if search_cnt > 0:
            questions_with_tools += 1
        total_cost_usd += cost
        latencies.append(elapsed)

        # Citation validation against context
        context_ids_set = set(trace.get("context_ids", []))
        cits_in_context = [c for c in cits if c in context_ids_set]
        total_citations += len(cits)
        valid_citations += len(cits_in_context)

        if status == "answered":
            total_answered += 1
            if len(cits) > 0:
                answers_with_citations += 1

        # Evaluate abstention vs answerability
        if not is_answerable:
            total_unanswerable += 1
            if status == "abstain":
                abstentions_correct += 1
                trace["abstention_correct"] = True
            else:
                trace["abstention_correct"] = False
        else:
            total_answerable += 1
            if status == "abstain":
                false_abstentions += 1
                trace["false_abstention"] = True
            else:
                trace["false_abstention"] = False

        traces.append(trace)

    n_total = len(questions)
    summary = {
        "config_name": config_name,
        "split": questions[0].get("split") if questions else "unknown",
        "total_cases": n_total,
        "tool_use_rate": f"{questions_with_tools}/{n_total} ({questions_with_tools / max(1, n_total):.2%})",
        "avg_tool_calls": round(total_tool_calls / max(1, n_total), 2),
        "citation_validity": f"{valid_citations}/{total_citations} ({valid_citations / max(1, total_citations):.2%})" if total_citations > 0 else "N/A",
        "answer_citation_coverage": f"{answers_with_citations}/{total_answered} ({answers_with_citations / max(1, total_answered):.2%})" if total_answered > 0 else "0/0",
        "abstention_accuracy": f"{abstentions_correct}/{total_unanswerable} ({abstentions_correct / max(1, total_unanswerable):.2%})" if total_unanswerable > 0 else "N/A",
        "false_abstention_rate": f"{false_abstentions}/{total_answerable} ({false_abstentions / max(1, total_answerable):.2%})" if total_answerable > 0 else "0/0",
        "median_latency_ms": round(statistics.median(latencies), 2) if latencies else 0,
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else 0,
        "total_cost_usd": round(total_cost_usd, 6)
    }
    return summary, traces


def main():
    parser = argparse.ArgumentParser(description="Day 7 Retrieval and Agent Evaluation Pipeline")
    parser.add_argument("--mode", choices=["retrieval", "agent", "all"], default="retrieval",
                        help="Chế độ đánh giá: 'retrieval' (dense, bm25, hybrid) hoặc 'agent'")
    parser.add_argument("--method", choices=["dense", "lexical", "hybrid_rrf", "hybrid_score", "rerank", "all_retrievers"],
                        default="dense", help="Phương pháp retrieval cụ thể")
    parser.add_argument("--split", choices=["dev", "test"], default="dev",
                        help="Tập đánh giá: dev hoặc test")
    parser.add_argument("--chunks", type=str, default="data/chunks.json")
    parser.add_argument("--questions", type=str, default="data/benchmark_questions.json")
    parser.add_argument("--output", type=str, default="results/eval_run")
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--budget", type=int, default=1200)
    parser.add_argument("--enable_filter", action="store_true", help="Bật metadata filter suy luận từ query")
    parser.add_argument("--provider", choices=["gemini", "openai"], default="gemini", help="LLM Provider cho agent")
    parser.add_argument("--model", type=str, default=None, help="Tên model cụ thể (VD: gemini-2.5-flash)")
    parser.add_argument("--max_searches", type=int, default=3, help="Số lần tìm kiếm tối đa cho agent")
    args = parser.parse_args()

    # Load data
    chunks_path = Path(args.chunks)
    questions_path = Path(args.questions)
    
    if not chunks_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file chunks tại: {chunks_path}")
    if not questions_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file questions tại: {questions_path}")

    all_chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    all_questions = json.loads(questions_path.read_text(encoding="utf-8"))
    
    selected_questions = [q for q in all_questions if q.get("split") == args.split]
    if not selected_questions:
        raise ValueError(f"Không có câu hỏi nào thuộc split: {args.split}")

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=======================================================")
    print(f"🚀 BẮT ĐẦU ĐÁNH GIÁ DAY 7 (Split: {args.split.upper()} | Mode: {args.mode.upper()})")
    print(f"📂 Tổng số câu hỏi: {len(selected_questions)}")
    print(f"📂 Thư mục xuất kết quả: {out_dir}")
    print(f"=======================================================\n")

    # Initialize retrievers
    dense = DenseRetriever()
    lexical = LexicalRetriever(all_chunks)
    hybrid_rrf = HybridRetriever(dense, lexical, fusion_mode="rrf", rrf_k=60)
    hybrid_score = HybridRetriever(dense, lexical, fusion_mode="score_fusion", alpha=0.6)
    rerank = RerankRetriever(hybrid_rrf)

    if args.mode in ("retrieval", "all"):
        retrievers_to_run = {}
        if args.method == "dense" or args.method == "all_retrievers":
            retrievers_to_run["Dense (Baseline)"] = (dense, False)
        if args.method == "lexical" or args.method == "all_retrievers":
            retrievers_to_run["Lexical BM25"] = (lexical, False)
        if args.method == "hybrid_rrf" or args.method == "all_retrievers":
            retrievers_to_run["Hybrid RRF (k=60)"] = (hybrid_rrf, False)
        if args.method == "hybrid_score" or args.method == "all_retrievers":
            retrievers_to_run["Hybrid Score Fusion"] = (hybrid_score, False)
        if args.method == "rerank" or args.method == "all_retrievers":
            retrievers_to_run["Hybrid + Rerank"] = (rerank, False)
            
        if args.enable_filter:
            retrievers_to_run[f"{args.method.capitalize()} + Filter"] = (
                dense if args.method == "dense" else hybrid_rrf, True
            )

        all_summaries = []
        for name, (retr, use_flt) in retrievers_to_run.items():
            print(f"[*] Đang chạy đánh giá: {name}...")
            summary, traces = evaluate_retriever(
                retr,
                selected_questions,
                strategy_name=name,
                top_k=args.top_k,
                budget=args.budget,
                enable_filter=use_flt
            )
            all_summaries.append(summary)
            
            # Save individual strategy traces
            clean_name = name.lower().replace(" ", "_").replace("+", "plus").replace("(", "").replace(")", "").replace("=", "")
            (out_dir / f"traces_{clean_name}.json").write_text(
                json.dumps(traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            print(f"  -> Hit@1: {summary['hit1']} | Hit@3: {summary['hit3']} | Context Hit: {summary['context_evidence_rate']} | MRR: {summary['mrr_at_3']}")

        (out_dir / "retrieval_summary.json").write_text(
            json.dumps(all_summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n[+] Đã lưu tổng hợp retrieval tại: {out_dir / 'retrieval_summary.json'}")

    if args.mode in ("agent", "all"):
        print(f"\n[*] Khởi tạo Agent với LLM Client ({args.provider.upper()})...")
        llm = LLMClient(provider=args.provider, model_name=args.model)
        agent_engine = AgentEngine(
            llm_client=llm,
            dense_retriever=dense,
            lexical_retriever=lexical,
            hybrid_retriever=hybrid_rrf,
            max_searches=args.max_searches,
            top_k=args.top_k,
            context_budget=args.budget
        )
        print(f"[*] Đang chạy Agent trên {len(selected_questions)} câu hỏi tập {args.split}...")
        config_label = f"Agent_{args.provider}_{args.model or 'default'}_top{args.top_k}_b{args.budget}"
        agent_summary, agent_traces = evaluate_agent(agent_engine, selected_questions, config_name=config_label)
        
        (out_dir / "agent_summary.json").write_text(
            json.dumps(agent_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        (out_dir / "traces.json").write_text(
            json.dumps(agent_traces, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n================ TỔNG KẾT ĐÁNH GIÁ AGENT ================")
        print(json.dumps(agent_summary, ensure_ascii=False, indent=2))
        print(f"[+] Đã lưu log chi tiết Agent tại: {out_dir / 'traces.json'}")


if __name__ == "__main__":
    main()

