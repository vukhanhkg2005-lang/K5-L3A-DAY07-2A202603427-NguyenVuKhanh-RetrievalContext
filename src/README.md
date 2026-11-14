# Source Code Day 7

Thư mục chứa toàn bộ mã nguồn phục vụ bài lab Day 7 (Retrieval & Context Engineering):

- [retrievers.py](retrievers.py):
  - `DenseRetriever`: Kết nối ChromaDB Day 6 (`artifacts/chroma_db`, `strategy_3_semantic`), model `all-MiniLM-L6-v2`.
  - `LexicalRetriever`: BM25 lexical retriever tối ưu cho tiếng Việt và tiếng Anh.
  - `HybridRetriever`: Kết hợp Dense + Lexical theo RRF (Reciprocal Rank Fusion) hoặc Score Fusion có chuẩn hóa Min-Max.
  - `RerankRetriever`: Pipeline Retrieval + Reranking dựa trên độ tương đồng và token overlap.
  - `infer_metadata_filter`: Suy luận metadata filter từ câu hỏi mà không truy cập gold.
- [agent.py](agent.py):
  - `ContextManager`: Khử trùng lặp (deduplication) và nén ngữ cảnh theo ngân sách ký tự (`context_budget`).
  - `LLMClient`: Client giao tiếp với mô hình thật (hỗ trợ Google Gemini và OpenAI).
  - `AgentEngine`: Điều phối luồng tìm kiếm nhiều bước (Multi-step tool calling loop), trích xuất citation `[chunk_id]` và cơ chế từ chối (Abstention) nghiêm ngặt.
- [evaluate.py](evaluate.py):
  - Kịch bản chạy thực nghiệm và đánh giá toàn diện theo chuẩn `BENCHMARK.md` và `RUBRIC.md`.
  - Hỗ trợ so sánh các retriever và đánh giá Agent trên tập `dev` và `test`.

