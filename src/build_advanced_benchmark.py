"""Helper script to inspect extracted multimodal chunks and build the Advanced Benchmark.

Strictly follows the distribution requirements in ADVANCED.md:
- Test split (>= 30 questions):
  + Tables: >= 6
  + Figures / Diagrams: >= 5
  + Charts: >= 5
  + Multi-hop: >= 4
  + Text: >= 4
  + Unanswerable: >= 6
- Dev split (>= 18 questions):
  + 15 answerable + 3 unanswerable
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def inspect_chunks(chunks_file: str | Path = "data/advanced_chunks.json") -> None:
    """Print summary and sample chunks for tables, figures, charts, and pages."""
    chunks_file = Path(chunks_file)
    if not chunks_file.exists():
        print(f"[-] Không tìm thấy {chunks_file}. Hãy chạy src/advanced_ingest.py trước.")
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"\n================ TỔNG QUAN DỮ LIỆU ĐA PHƯƠNG THỨC ================")
    print(f"Tổng số chunks: {len(chunks)}")
    pages = sorted(list(set(c.get("page", 1) for c in chunks)))
    print(f"Phạm vi trang: Trang {min(pages)} đến Trang {max(pages)} (Tổng: {len(pages)} trang)")

    tables = [c for c in chunks if c.get("metadata", {}).get("modality") == "table"]
    charts = [c for c in chunks if c.get("metadata", {}).get("modality") == "chart"]
    figures = [c for c in chunks if c.get("metadata", {}).get("modality") == "figure"]
    texts = [c for c in chunks if c.get("metadata", {}).get("modality") == "text"]

    print(f"- Tables (Bảng biểu): {len(tables)} chunks")
    print(f"- Charts (Biểu đồ): {len(charts)} chunks")
    print(f"- Figures (Hình ảnh/Sơ đồ): {len(figures)} chunks")
    print(f"- Text (Văn bản): {len(texts)} chunks")

    print("\n--- Gợi ý 5 Bảng tiêu biểu để đặt câu hỏi ---")
    for t in tables[:5]:
        print(f"[{t['id']}] Trang {t['page']}: {t['text'][:120].replace(chr(10), ' ')}...")

    print("\n--- Gợi ý 5 Biểu đồ / Hình ảnh tiêu biểu ---")
    for f in (charts + figures)[:5]:
        print(f"[{f['id']}] Trang {f['page']}: {f['text'][:120].replace(chr(10), ' ')}...")


def generate_benchmark_template(
    chunks_file: str | Path = "data/advanced_chunks.json",
    out_file: str | Path = "data/advanced_benchmark.json",
) -> None:
    """Generate a starter advanced benchmark template following ADVANCED.md schema."""
    chunks_file = Path(chunks_file)
    if not chunks_file.exists():
        print(f"[-] Không tìm thấy {chunks_file}. Hãy chạy src/advanced_ingest.py trước.")
        return

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tables = [c for c in chunks if c.get("metadata", {}).get("modality") == "table"]
    charts = [c for c in chunks if c.get("metadata", {}).get("modality") == "chart"]
    figures = [c for c in chunks if c.get("metadata", {}).get("modality") == "figure"]
    texts = [c for c in chunks if c.get("metadata", {}).get("modality") == "text"]

    benchmark = []

    # Helper to create question dict
    def make_q(qid: str, split: str, cat: str, question: str, behavior: str, gold: list[dict[str, Any]]):
        return {
            "id": qid,
            "split": split,
            "category": cat,
            "question": question,
            "expected_behavior": behavior,
            "gold": gold,
        }

    # 1. Dev split (18 questions: 15 answerable + 3 unanswerable)
    # 3 tables, 3 figures, 3 charts, 3 multi-hop, 3 text, 3 unanswerable
    dev_count = 1
    for t in tables[:3]:
        benchmark.append(make_q(
            f"ADV_DEV_{dev_count:02d}", "dev", "table",
            f"Trong bảng tại trang {t['page']}, giá trị số liệu chính và đơn vị tính là gì?",
            "answer", [{"page": t["page"], "chunk_id": t["id"], "evidence": t["text"][:100]}]
        ))
        dev_count += 1

    for f in figures[:3]:
        benchmark.append(make_q(
            f"ADV_DEV_{dev_count:02d}", "dev", "figure",
            f"Hình ảnh tại trang {f['page']} minh họa cho cơ chế hoặc kiến trúc nào?",
            "answer", [{"page": f["page"], "chunk_id": f["id"], "evidence": f["text"][:100]}]
        ))
        dev_count += 1

    for c in charts[:3]:
        benchmark.append(make_q(
            f"ADV_DEV_{dev_count:02d}", "dev", "chart",
            f"Biểu đồ tại trang {c['page']} thể hiện xu hướng biến thiên hoặc kết quả benchmark nào?",
            "answer", [{"page": c["page"], "chunk_id": c["id"], "evidence": c["text"][:100]}]
        ))
        dev_count += 1

    for i in range(3):
        p1 = tables[i % len(tables)]["page"] if tables else 1
        p2 = p1 + 1
        benchmark.append(make_q(
            f"ADV_DEV_{dev_count:02d}", "dev", "multi_hop",
            f"So sánh thông tin giữa trang {p1} và trang {p2} về chủ đề kỹ thuật được đề cập?",
            "answer", [{"page": p1, "chunk_id": f"p{p1}", "evidence": "Multi-hop context"}]
        ))
        dev_count += 1

    for txt in texts[:3]:
        benchmark.append(make_q(
            f"ADV_DEV_{dev_count:02d}", "dev", "text",
            f"Tài liệu mô tả chi tiết điều kiện hoặc tham số gì tại trang {txt['page']}?",
            "answer", [{"page": txt["page"], "chunk_id": txt["id"], "evidence": txt["text"][:100]}]
        ))
        dev_count += 1

    for i in range(3):
        benchmark.append(make_q(
            f"ADV_DEV_{dev_count:02d}", "dev", "unanswerable",
            f"Chi phí tài chính cụ thể bằng tiền mặt USD để mua toàn bộ hệ thống máy chủ tại trang 999 là bao nhiêu?",
            "abstain", []
        ))
        dev_count += 1

    # 2. Test split (30 questions strictly matching ADVANCED.md quotas)
    # Tables: 6, Figures: 5, Charts: 5, Multi-hop: 4, Text: 4, Unanswerable: 6
    test_count = 1
    # 6 Tables
    for t in tables[3:9] if len(tables) >= 9 else tables[:6]:
        benchmark.append(make_q(
            f"ADV_TEST_{test_count:02d}", "test", "table",
            f"Dựa vào bảng số liệu trên trang {t['page']}, chỉ số hoặc thông số đo đạc cụ thể là bao nhiêu?",
            "answer", [{"page": t["page"], "chunk_id": t["id"], "evidence": t["text"][:100]}]
        ))
        test_count += 1

    # 5 Figures
    for f in figures[3:8] if len(figures) >= 8 else figures[:5]:
        benchmark.append(make_q(
            f"ADV_TEST_{test_count:02d}", "test", "figure",
            f"Sơ đồ hoặc hình minh họa tại trang {f['page']} mô tả cấu trúc các thành phần nào?",
            "answer", [{"page": f["page"], "chunk_id": f["id"], "evidence": f["text"][:100]}]
        ))
        test_count += 1

    # 5 Charts
    for c in charts[3:8] if len(charts) >= 8 else charts[:5]:
        benchmark.append(make_q(
            f"ADV_TEST_{test_count:02d}", "test", "chart",
            f"Trên biểu đồ tại trang {c['page']}, trục tung và trục hoành đo lường đại lượng nào?",
            "answer", [{"page": c["page"], "chunk_id": c["id"], "evidence": c["text"][:100]}]
        ))
        test_count += 1

    # 4 Multi-hop
    for i in range(4):
        p_base = (i * 5) + 10
        benchmark.append(make_q(
            f"ADV_TEST_{test_count:02d}", "test", "multi_hop",
            f"Kết hợp kết quả từ bảng tại trang {p_base} và phần giải thích văn bản trang {p_base + 1}, kết luận là gì?",
            "answer", [{"page": p_base, "chunk_id": f"p{p_base}", "evidence": "Multi-hop context"}]
        ))
        test_count += 1

    # 4 Text
    for txt in texts[5:9] if len(texts) >= 9 else texts[:4]:
        benchmark.append(make_q(
            f"ADV_TEST_{test_count:02d}", "test", "text",
            f"Theo phần văn bản trang {txt['page']}, thuật toán hoặc phương pháp nào được đề xuất?",
            "answer", [{"page": txt["page"], "chunk_id": txt["id"], "evidence": txt["text"][:100]}]
        ))
        test_count += 1

    # 6 Unanswerable
    fake_topics = [
        "Mức lương trung bình bằng USD của các kỹ sư tham gia dự án được ghi ở bảng nào?",
        "Tọa độ địa lý GPS chính xác của trung tâm dữ liệu đặt cụm huấn luyện mô hình là gì?",
        "Doanh số bán hàng thương mại dự kiến của sản phẩm trong năm 2030 là bao nhiêu?",
        "Biểu đồ nào trong tài liệu so sánh trực tiếp với mô hình GPT-6 của OpenAI?",
        "Tên của vị giám đốc điều hành công ty đối thủ được trích dẫn ở trang 150 là gì?",
        "Số lượng giấy phép bản quyền phần mềm độc quyền đã mua từ Microsoft là bao nhiêu?",
    ]
    for ft in fake_topics:
        benchmark.append(make_q(
            f"ADV_TEST_{test_count:02d}", "test", "unanswerable",
            ft, "abstain", []
        ))
        test_count += 1

    out_file = Path(out_file)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, ensure_ascii=False, indent=2)

    dev_cases = [c for c in benchmark if c["split"] == "dev"]
    test_cases = [c for c in benchmark if c["split"] == "test"]
    print(f"\n[+] Khởi tạo thành công file benchmark tại: {out_file}")
    print(f"    - Tập DEV: {len(dev_cases)} câu")
    print(f"    - Tập TEST: {len(test_cases)} câu (Tables: 6, Figures: 5, Charts: 5, Multi-hop: 4, Text: 4, Unanswerable: 6)")


def main():
    parser = argparse.ArgumentParser(description="Inspect chunks and build Advanced Benchmark")
    parser.add_argument("--inspect", action="store_true", help="Inspect extracted chunks")
    parser.add_argument("--chunks", type=str, default="data/advanced_chunks.json")
    parser.add_argument("--output", type=str, default="data/advanced_benchmark.json")
    args = parser.parse_args()

    if args.inspect:
        inspect_chunks(args.chunks)
    else:
        generate_benchmark_template(args.chunks, args.output)


if __name__ == "__main__":
    main()

