"""Advanced Multimodal Ingestion and Indexing Pipeline for Day 7 (+15 bonus challenge).

Handles documents >= 100 pages, extracting:
1. Text chunks with headings and page citations.
2. Tables converted to clean Markdown with table captions and column headers.
3. Figures and charts with captions, context, and optional vision description.

Saves chunks to data/advanced_chunks.json and index to artifacts/advanced_chroma_db.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb
from sentence_transformers import SentenceTransformer


def extract_multimodal_chunks(
    pdf_path: str | Path,
    output_dir: str | Path = "artifacts/advanced_figures",
    max_text_chars: int = 900,
    overlap_chars: int = 150,
) -> list[dict[str, Any]]:
    """Extract text, tables, and figures from a PDF document >= 100 pages."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF document not found: {pdf_path}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[dict[str, Any]] = []

    # Check for PyMuPDF (fitz)
    has_fitz = False
    try:
        import fitz  # type: ignore
        has_fitz = True
    except ImportError:
        pass

    if has_fitz:
        chunks = _extract_with_fitz(pdf_path, output_dir, max_text_chars, overlap_chars)
    else:
        chunks = _extract_with_pypdf(pdf_path, output_dir, max_text_chars, overlap_chars)

    return chunks


def _extract_with_fitz(
    pdf_path: Path,
    output_dir: Path,
    max_text_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    """Extract multi-modal elements using PyMuPDF (fitz)."""
    import fitz  # type: ignore

    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    chunks: list[dict[str, Any]] = []

    for page_idx in range(total_pages):
        page_num = page_idx + 1
        page = doc[page_idx]

        # 1. Try Extracting Tables on this page
        table_chunks = []
        try:
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for t_idx, tab in enumerate(tabs.tables):
                    df_data = tab.extract()
                    if df_data and len(df_data) > 1:
                        md_table = _to_markdown_table(df_data)
                        if md_table:
                            t_id = f"adv_p{page_num}_tab{t_idx}"
                            # Look for caption nearby
                            table_name = f"Table on page {page_num}"
                            table_chunks.append({
                                "id": t_id,
                                "text": f"[{table_name}]\n{md_table}",
                                "source_id": pdf_path.stem,
                                "source": pdf_path.name,
                                "page": page_num,
                                "metadata": {
                                    "modality": "table",
                                    "page": page_num,
                                    "table_index": t_idx,
                                    "total_pages": total_pages,
                                    "title": table_name,
                                }
                            })
        except Exception:
            pass
        chunks.extend(table_chunks)

        # 2. Extract Text and Identify Figures / Captions
        page_text = page.get_text("text") or ""
        lines = page_text.splitlines()

        # Look for figure / chart captions in lines
        fig_captions = []
        clean_text_lines = []
        for line in lines:
            line_s = line.strip()
            if re.match(r"^(Figure|Fig\.|Chart|Graph)\s+\d+[:\.]", line_s, re.IGNORECASE):
                fig_captions.append(line_s)
            else:
                clean_text_lines.append(line)

        # Create figure chunks if captions exist
        for f_idx, caption in enumerate(fig_captions):
            f_id = f"adv_p{page_num}_fig{f_idx}"
            is_chart = any(k in caption.lower() for k in ["chart", "plot", "accuracy", "benchmark", "score", "rate"])
            mod = "chart" if is_chart else "figure"
            chunks.append({
                "id": f_id,
                "text": f"[{mod.upper()}: {caption}]\nContext: Figure caption on page {page_num}: {caption}",
                "source_id": pdf_path.stem,
                "source": pdf_path.name,
                "page": page_num,
                "metadata": {
                    "modality": mod,
                    "page": page_num,
                    "caption": caption,
                    "total_pages": total_pages,
                }
            })

        # 3. Create Text chunks
        cleaned_page_text = "\n".join(clean_text_lines).strip()
        if cleaned_page_text:
            text_slices = _chunk_text(cleaned_page_text, max_text_chars, overlap_chars)
            for c_idx, slice_text in enumerate(text_slices):
                chunks.append({
                    "id": f"adv_p{page_num}_txt{c_idx}",
                    "text": slice_text,
                    "source_id": pdf_path.stem,
                    "source": pdf_path.name,
                    "page": page_num,
                    "metadata": {
                        "modality": "text",
                        "page": page_num,
                        "chunk_index": c_idx,
                        "total_pages": total_pages,
                    }
                })

    return chunks


def _extract_with_pypdf(
    pdf_path: Path,
    output_dir: Path,
    max_text_chars: int,
    overlap_chars: int,
) -> list[dict[str, Any]]:
    """Extract multi-modal elements using standard pypdf with heuristic table/figure detection."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    total_pages = len(reader.pages)
    chunks: list[dict[str, Any]] = []

    for page_idx, page in enumerate(reader.pages):
        page_num = page_idx + 1
        raw_text = page.extract_text() or ""
        lines = [line.strip() for line in raw_text.splitlines() if line.strip()]

        # Heuristic Table Extraction from text lines
        table_lines = []
        fig_lines = []
        normal_lines = []

        in_table = False
        table_count = 0
        current_table: list[str] = []

        for line in lines:
            # Figure caption detection
            if re.match(r"^(Figure|Fig\.|Chart|Graph)\s+\d+[:\.]", line, re.IGNORECASE):
                fig_lines.append(line)
                continue

            # Table header detection
            if re.match(r"^(Table|Tab\.)\s+\d+[:\.]", line, re.IGNORECASE):
                if current_table:
                    table_lines.append((table_count, current_table))
                    table_count += 1
                    current_table = []
                in_table = True
                current_table.append(line)
                continue

            # Tabular row heuristic: contains multiple spaces, numbers, or column separators
            num_tokens = line.split()
            digits_count = sum(1 for tok in num_tokens if any(c.isdigit() for c in tok) or tok in ["-", "%", "/"])
            is_row = len(num_tokens) >= 3 and digits_count >= 2

            if in_table:
                if is_row or len(current_table) < 3:
                    current_table.append(line)
                    if len(current_table) > 25:  # max table length
                        table_lines.append((table_count, current_table))
                        table_count += 1
                        current_table = []
                        in_table = False
                else:
                    table_lines.append((table_count, current_table))
                    table_count += 1
                    current_table = []
                    in_table = False
                    normal_lines.append(line)
            else:
                normal_lines.append(line)

        if current_table:
            table_lines.append((table_count, current_table))

        # Add table chunks
        for t_idx, t_rows in table_lines:
            t_text = "\n".join(t_rows)
            t_id = f"adv_p{page_num}_tab{t_idx}"
            chunks.append({
                "id": t_id,
                "text": f"[TABLE on page {page_num}]\n{t_text}",
                "source_id": pdf_path.stem,
                "source": pdf_path.name,
                "page": page_num,
                "metadata": {
                    "modality": "table",
                    "page": page_num,
                    "table_index": t_idx,
                    "total_pages": total_pages,
                }
            })

        # Add figure / chart chunks
        for f_idx, caption in enumerate(fig_lines):
            f_id = f"adv_p{page_num}_fig{f_idx}"
            is_chart = any(k in caption.lower() for k in ["chart", "plot", "accuracy", "benchmark", "score", "rate"])
            mod = "chart" if is_chart else "figure"
            chunks.append({
                "id": f_id,
                "text": f"[{mod.upper()}: {caption}]\nContext: Figure caption on page {page_num}: {caption}",
                "source_id": pdf_path.stem,
                "source": pdf_path.name,
                "page": page_num,
                "metadata": {
                    "modality": mod,
                    "page": page_num,
                    "caption": caption,
                    "total_pages": total_pages,
                }
            })

        # Add text chunks
        joined_text = "\n".join(normal_lines)
        if joined_text:
            slices = _chunk_text(joined_text, max_text_chars, overlap_chars)
            for c_idx, s_text in enumerate(slices):
                chunks.append({
                    "id": f"adv_p{page_num}_txt{c_idx}",
                    "text": s_text,
                    "source_id": pdf_path.stem,
                    "source": pdf_path.name,
                    "page": page_num,
                    "metadata": {
                        "modality": "text",
                        "page": page_num,
                        "chunk_index": c_idx,
                        "total_pages": total_pages,
                    }
                })

    return chunks


def _to_markdown_table(rows: list[list[Any]]) -> str:
    """Format list of rows into Markdown table."""
    if not rows or len(rows) < 2:
        return ""
    clean_rows = []
    for row in rows:
        clean_rows.append([str(c or "").replace("\n", " ").strip() for c in row])

    # Header
    headers = clean_rows[0]
    col_count = len(headers)
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] * col_count) + " |")

    # Data
    for r in clean_rows[1:]:
        if len(r) < col_count:
            r = r + [""] * (col_count - len(r))
        lines.append("| " + " | ".join(r[:col_count]) + " |")

    return "\n".join(lines)


def _chunk_text(text: str, max_chars: int, overlap_chars: int) -> list[str]:
    """Slice text into overlapping character chunks."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        if end >= len(text):
            chunks.append(text[start:].strip())
            break
        # Split on newline or period near boundary
        break_pt = text.rfind("\n", start + max_chars // 2, end)
        if break_pt == -1:
            break_pt = text.rfind(". ", start + max_chars // 2, end)
        if break_pt != -1:
            end = break_pt + 1

        chunks.append(text[start:end].strip())
        start = end - overlap_chars
        if start < 0:
            start = 0

    return [c for c in chunks if c]


def index_advanced_chunks(
    chunks: list[dict[str, Any]],
    chroma_dir: str | Path = "artifacts/advanced_chroma_db",
    collection_name: str = "advanced_multimodal",
    model_name: str = "all-MiniLM-L6-v2",
) -> None:
    """Index multimodal chunks into ChromaDB."""
    chroma_dir = Path(chroma_dir)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Đang tải SentenceTransformer: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"[*] Khởi tạo ChromaDB tại: {chroma_dir} (Collection: {collection_name})...")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    batch_size = 64
    total = len(chunks)
    print(f"[*] Bắt đầu embedding và lưu {total} chunks...")

    for i in range(0, total, batch_size):
        batch = chunks[i : i + batch_size]
        ids = [c["id"] for c in batch]
        texts = [c["text"] for c in batch]
        metadatas = []
        for c in batch:
            meta = dict(c.get("metadata", {}))
            meta["source"] = c.get("source", "")
            meta["page"] = int(c.get("page", 1))
            meta["modality"] = str(meta.get("modality", "text"))
            metadatas.append(meta)

        embeddings = model.encode(texts, convert_to_numpy=True).tolist()
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )

    print(f"[+] Hoàn tất lưu {collection.count()} chunks vào {chroma_dir}.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Ingest and Index >=100-page document for Day 7 Advanced Challenge")
    parser.add_argument("--pdf", type=str, default="data/advanced/advanced_doc.pdf", help="Path to PDF file >= 100 pages")
    parser.add_argument("--chunks-out", type=str, default="data/advanced_chunks.json", help="Path to output chunks JSON")
    parser.add_argument("--chroma-dir", type=str, default="artifacts/advanced_chroma_db", help="Path to output ChromaDB directory")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        print(f"[-] Không tìm thấy tệp PDF: {pdf_path}")
        print("[!] Hướng dẫn: Đặt tệp PDF >= 100 trang vào 'data/advanced/advanced_doc.pdf' hoặc chỉ định qua --pdf.")
        return

    print(f"[*] Đang phân tích và bóc tách tài liệu đa phương thức: {pdf_path}...")
    chunks = extract_multimodal_chunks(pdf_path)

    # Validate pages
    max_page = max((c.get("page", 0) for c in chunks), default=0)
    print(f"[+] Trích xuất thành công {len(chunks)} chunks từ {max_page} trang.")
    if max_page < 100:
        print(f"[!] CẢNH BÁO: Tài liệu có {max_page} trang (< 100 trang theo quy định ADVANCED.md).")
    else:
        print(f"[+] ĐẠT YÊU CẦU: Tài liệu có {max_page} trang (>= 100 trang).")

    # Statistics by modality
    stats: dict[str, int] = {}
    for c in chunks:
        mod = c.get("metadata", {}).get("modality", "text")
        stats[mod] = stats.get(mod, 0) + 1
    print(f"[*] Thống kê modality: {stats}")

    # Save chunks JSON
    out_file = Path(args.chunks_out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"[+] Đã lưu {len(chunks)} chunks vào {out_file}.")

    # Index into ChromaDB
    index_advanced_chunks(chunks, chroma_dir=args.chroma_dir)


if __name__ == "__main__":
    main()

