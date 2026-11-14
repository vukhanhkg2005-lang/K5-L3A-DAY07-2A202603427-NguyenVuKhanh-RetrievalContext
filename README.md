# Day 7 — Agentic RAG

## Hướng dẫn Cài đặt & Chạy Thực nghiệm (Student Submission - Nguyễn Vũ Khanh)

### 1. Kiến trúc Hệ thống & Mô hình Sử dụng

- **Vector Store & Embeddings:**
  - Kết nối trực tiếp từ Day 6: `artifacts/chroma_db`, collection `strategy_3_semantic` (313 chunks, 77 trang bài báo Llama 2).
  - Mô hình Embedding: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions, cosine distance).
- **Mô hình Agent (LLM):**
  - Mô hình thật: `nvidia/nemotron-3.5-lightning:free` qua OpenRouter API (hoặc tương thích với OpenAI API / Gemini 2.5 Flash).
  - Nhiệt độ (Temperature): `0.0` đảm bảo tính tái lập 100%.
- **Các phương pháp Retrieval đã triển khai trong `src/retrievers.py`:**
  1. *Dense Retriever:* Kết nối ChromaDB Day 6 qua embedding `all-MiniLM-L6-v2`.
  2. *Lexical Retriever:* BM25 cải tiến hỗ trợ bộ lọc metadata `section`.
  3. *Hybrid RRF Retriever:* Hợp nhất Dense + Lexical qua xếp hạng nghịch đảo Reciprocal Rank Fusion ($k=60$).
  4. *Hybrid Score Fusion:* Hợp nhất điểm chuẩn hóa Min-Max giữa Dense và BM25.
  5. *Cải thiện 1 (Hybrid + Filter):* Tự động suy luận metadata `section` từ query để thu hẹp không gian tìm kiếm.
  6. *Cải thiện 2 (Hybrid + Cross-Encoder Rerank):* Lấy top-10 ứng viên Hybrid rồi tái xếp hạng bằng `cross-encoder/ms-marco-MiniLM-L-6-v2` (**Cấu hình tốt nhất được chọn đóng băng cho tập Test**).
- **Thử thách nâng cao (+15 điểm bonus):**
  - Tài liệu thực tế $\ge 100$ trang: GPT-4 Technical Report (100 trang, `data/advanced/advanced_doc.pdf`).
  - Pipeline Ingestion đa phương thức (`src/advanced_ingest.py`): Bóc tách 460 chunks (19 Tables, 8 Charts, 12 Figures, 421 Text) vào ChromaDB `artifacts/advanced_chroma_db`.
  - Advanced Agent (`src/advanced_agent.py`): Hybrid Search lọc theo `modality` (`table`, `chart`, `figure`, `text`) kết hợp Query Rewriting tự sửa lỗi, đạt **100% (30/30 câu test)**.

---

### 2. Cài đặt Môi trường (Environment Setup)

Yêu cầu: **Python 3.11+**

```powershell
# 1. Tạo và kích hoạt môi trường ảo (khuyến nghị)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 2. Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt

# 3. Thiết lập biến môi trường API Key
Copy-Item .env.example .env
# Mở file .env và điền API key của bạn:
# OPENAI_API_KEY=sk-or-v1-...
```

---

### 3. Kiểm tra Unit Tests

Dự án đi kèm 29 unit tests kiểm thử toàn diện cả phần cơ bản và phần nâng cao:

```powershell
python -m unittest discover -s tests -v
```
*(Kết quả: `Ran 29 tests in 0.198s ... OK`)*

---

### 4. Lệnh chạy Thực nghiệm Phần Cơ bản (Dev & Test)

Tất cả các lệnh thực thi đều thông qua script CLI chuẩn `src/evaluate.py`:

```powershell
# --- BƯỚC 4.1: ĐÁNH GIÁ CÁC PHƯƠNG PHÁP RETRIEVAL TRÊN DEV ---
# Chạy toàn bộ 5 phương pháp retrieval trên 15 câu dev có đáp án:
python src/evaluate.py --mode retrieval --method all_retrievers --split dev --top_k 3 --budget 1200 --output results/dev_retrieval

# --- BƯỚC 4.2: THỬ NGHIỆM CÁC CẤU HÌNH CẢI THIỆN TRÊN DEV ---
# Thử nghiệm Cải thiện 1 (Hybrid RRF + Section Filter):
python src/evaluate.py --mode retrieval --method hybrid_filter --split dev --top_k 3 --budget 1200 --output results/dev_improved_filter

# Thử nghiệm Cải thiện 2 (Hybrid + Cross-Encoder Rerank - Cấu hình chốt):
python src/evaluate.py --mode retrieval --method rerank --split dev --top_k 3 --budget 1200 --output results/dev_improved_rerank

# --- BƯỚC 4.3: CHẠY AGENT MODEL THẬT TRÊN DEV (17 CÂU: 15 ANSWERABLE + 2 UNANSWERABLE) ---
python src/evaluate.py --mode agent --provider openai --model nvidia/nemotron-3.5-lightning:free --split dev --top_k 3 --budget 1500 --output results/dev_agent_nvidia

# --- BƯỚC 4.4: ĐÁNH GIÁ CẤU HÌNH ĐÃ CHỌN TRÊN TẬP TEST (ĐỐI CHIẾU VS BASELINE) ---
# Đánh giá đối chứng Baseline Dense và Hybrid+Rerank trên 7 câu test:
python src/evaluate.py --mode retrieval --method all_retrievers --split test --top_k 3 --budget 1200 --output results/test_retrieval
```

---

### 5. Lệnh chạy Thử thách Nâng cao (Tài liệu >= 100 trang & Đa phương thức)

```powershell
# --- BƯỚC 5.1: TẢI TÀI LIỆU 100 TRANG VÀO DATA/ADVANCED/ ---
New-Item -ItemType Directory -Force -Path data/advanced
Invoke-WebRequest -Uri "https://arxiv.org/pdf/2303.08774.pdf" -OutFile "data/advanced/advanced_doc.pdf"

# --- BƯỚC 5.2: BÓC TÁCH ĐA PHƯƠNG THỨC & TẠO VECTOR STORE ADVANCED ---
python src/advanced_ingest.py --pdf data/advanced/advanced_doc.pdf --chunks-out data/advanced_chunks.json --chroma-dir artifacts/advanced_chroma_db

# --- BƯỚC 5.3: KHỞI TẠO BENCHMARK NÂNG CAO (18 DEV + 30 TEST) ---
python src/populate_advanced_benchmark.py

# --- BƯỚC 5.4: ĐÁNH GIÁ AGENT ĐA PHƯƠNG THỨC TRÊN TẬP TEST NÂNG CAO ---
python src/advanced_agent.py --benchmark data/advanced_benchmark.json --split test --output results/advanced --provider openai --model nvidia/nemotron-3.5-lightning:free
```

---

### 6. Cấu trúc Thư mục Kết quả & Báo cáo Cá nhân

- **Báo cáo cá nhân chính thức:** [report/individual/2A202603427_NguyenVuKhanh.md](report/individual/2A202603427_NguyenVuKhanh.md) (điền đầy đủ cả 2 phần Cơ bản và Nâng cao).
- **Thư mục lưu trữ kết quả thực nghiệm thực tế:**
  - `results/dev_retrieval/retrieval_summary.json`: So sánh Dense Baseline, BM25, Hybrid RRF, Hybrid Score Fusion, Rerank trên tập dev.
  - `results/dev_improved_filter/`: Kết quả của Hybrid RRF + Section Filter.
  - `results/dev_improved_rerank/`: Kết quả của Hybrid + Cross-Encoder Rerank.
  - `results/dev_agent_nvidia/`: Chứa `agent_summary.json` và `traces.json` (17 câu dev với model thật).
  - `results/test_retrieval/`: Chứa `retrieval_summary.json` đối chiếu Baseline Dense vs Cấu hình đã chọn trên tập test.
  - `results/advanced/`: Chứa `advanced_summary.json` và `traces.json` (30 câu test nâng cao đạt 100%).

---

# Nội dung Đề bài Gốc (Assignment Specification)

**Bài cá nhân · Python 3.11+ · tiếp nối Day 6**

Dùng embedding và vector store từ Day 6 để xây một agent biết tìm bằng chứng, trả lời có nguồn và nhận biết khi không đủ thông tin. Bạn tự chọn kiến trúc, số tool và cách tổ chức retrieval.

> **Tùy chọn:** [Gợi ý dùng AI để lập trình](AGENT_SETUP.md) giúp bạn làm lab; phần này không bắt buộc và không tính điểm.

## Bài cần làm

![Agent tìm tài liệu trong kho Day 6 và trả lời có nguồn](assets/retrieval-pipeline.svg)

1. Kết nối dữ liệu, embedding và vector store Day 6.
2. Xây agent dùng model thật và ít nhất một khả năng truy xuất dữ liệu. Agent có thể dùng nhiều tool và nhiều lượt gọi.
3. Thử **ít nhất 3 phương pháp retrieval khác nhau**, trong đó có dense retrieval dùng vector store Day 6. Tự chọn các phương pháp còn lại; xem [gợi ý](GUIDE.md#2-thử-các-phương-pháp-retrieval).
4. Thử nhiều cấu hình, benchmark, phân tích lỗi và chọn cấu hình tốt trên **dev**; sau đó đánh giá trên **test**.

Benchmark tối thiểu có **24 câu**: dev gồm 15 câu có đáp án và 2 câu thiếu bằng chứng; test gồm 5 câu có đáp án và 2 câu thiếu bằng chứng. Xem [benchmark](BENCHMARK.md) và [rubric](RUBRIC.md).

## Thử thách nâng cao · +15 điểm

Index một tài liệu thực tế từ **100 trang** có bảng, ảnh và biểu đồ; xây agent truy xuất và trả lời các câu hỏi về nội dung đó. Hoàn thành [thử thách nâng cao](ADVANCED.md) và đạt **ít nhất 90% benchmark test** để nhận 15 điểm bonus. Phần cơ bản tối đa 85 điểm; toàn bài tối đa 100 điểm.

## Bắt đầu

1. Đọc [hướng dẫn](GUIDE.md), [rubric](RUBRIC.md) và [quy tắc](RULES.md).
2. Tạo repo cá nhân theo [cách nộp bài](SUBMISSION.md), chuẩn bị đầu ra Day 6 và model hỗ trợ tool calling.
3. Dùng hoặc thay thế [starter](starter/README.md); bạn cần kết nối model và vector store thật.
4. Chạy `python -m unittest discover -s tests -v`. Test mock chỉ kiểm tra code, không phải benchmark agent thật.

![Các bước thử nghiệm trên dev và đánh giá trên test](assets/experiment-loop.svg)

Nộp code, cách chạy, cấu hình, benchmark, trace và [báo cáo](report/individual/MSSV_HoTen.md). Thời gian ở [checkpoints](CHECKPOINTS.md) là gợi ý, có thể điều chỉnh theo lớp và phạm vi thử nghiệm.
