"""Day 7 Retrieval System: Dense, Lexical (BM25), Hybrid (RRF & Score Fusion), and Rerank."""
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any, Protocol


def tokenize(text: str) -> list[str]:
    """Tokenize text into lowercased unicode NFC normalized tokens."""
    if not text:
        return []
    normalized = unicodedata.normalize("NFC", text).casefold()
    return re.findall(r"\w+", normalized)


def infer_metadata_filter(query: str) -> dict[str, str] | None:
    """Infer section metadata filter purely from query semantics (NEVER from gold)."""
    q_lower = query.lower()
    
    # Pretraining signals
    pretrain_signals = [
        "pretrain", "tiền huấn luyện", "adamw", "grouped-query", "gqa", 
        "carbon", "tco2eq", "bpe", "sentencepiece", "vocabulary size", 
        "từ vựng", "trillion tokens", "nghìn tỷ token"
    ]
    if any(sig in q_lower for sig in pretrain_signals):
        return {"section": "Pretraining"}
    
    # Fine-tuning signals
    finetune_signals = [
        "fine-tuning", "sft", "rlhf", "reward model", "mô hình phần thưởng",
        "ppo", "proximal policy", "rejection sampling", "ghost attention", "gatt",
        "preference annotation", "chú thích"
    ]
    if any(sig in q_lower for sig in finetune_signals):
        return {"section": "Fine-tuning"}
        
    # Safety signals
    safety_signals = [
        "safety", "an toàn", "red team", "red teaming", "truthfulqa", 
        "toxigen", "bold", "rủi ro", "độc hại", "adversarial", 
        "context distillation", "preprompt"
    ]
    if any(sig in q_lower for sig in safety_signals):
        return {"section": "Safety"}
        
    return None


class BaseRetriever(Protocol):
    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        """Retrieve top_k chunks matching query and optional metadata filters."""
        ...


class DenseRetriever:
    """Dense retriever connecting to Day 6 ChromaDB collection with all-MiniLM-L6-v2."""
    def __init__(
        self,
        chroma_path: str = "artifacts/chroma_db",
        collection_name: str = "strategy_3_semantic",
        embedding_model_name: str = "all-MiniLM-L6-v2"
    ):
        import chromadb
        from sentence_transformers import SentenceTransformer

        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        
        self.client = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.client.get_collection(name=collection_name)
        self.embedder = SentenceTransformer(embedding_model_name)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        if not query.strip():
            return []
            
        q_vec = self.embedder.encode([query], normalize_embeddings=True).tolist()
        
        # Build Chroma where clause
        where_clause = None
        if filters:
            if len(filters) == 1:
                where_clause = filters
            else:
                where_clause = {"$and": [{k: v} for k, v in filters.items()]}
                
        res = self.collection.query(
            query_embeddings=q_vec,
            n_results=top_k,
            where=where_clause
        )
        
        results = []
        if not res or not res.get("ids") or not res["ids"][0]:
            return []
            
        for rank in range(len(res["ids"][0])):
            cid = res["ids"][0][rank]
            doc = res["documents"][0][rank]
            meta = res["metadatas"][0][rank]
            dist = res["distances"][0][rank]
            
            # Map cosine distance [0, 2] to similarity score [0, 1]
            sim_score = round(1.0 - (dist / 2.0), 4)
            
            results.append({
                "id": cid,
                "text": doc,
                "source_id": meta.get("doc_id", "llama2_paper"),
                "source": meta.get("source", "data/llama2_paper.pdf"),
                "page": meta.get("page_number", meta.get("page", 1)),
                "metadata": meta,
                "score": sim_score,
                "strategy": "dense"
            })
            
        # Ensure unique IDs and top_k limit
        unique_results = []
        seen = set()
        for r in results:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique_results.append(r)
        return unique_results[:top_k]


class LexicalRetriever:
    """BM25 lexical retriever for Vietnamese + English corpus."""
    def __init__(self, chunks: list[dict[str, Any]], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.N = len(chunks)
        
        self.doc_tokens = [tokenize(c["text"]) for c in chunks]
        self.doc_lens = [len(toks) for toks in self.doc_tokens]
        self.avgdl = sum(self.doc_lens) / max(1, self.N)
        
        # Calculate document frequency
        df: Counter[str] = Counter()
        for toks in self.doc_tokens:
            df.update(set(toks))
            
        # BM25 IDF
        self.idf = {
            term: math.log(1.0 + (self.N - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        
        # Precompute term frequencies per document
        self.doc_tfs = [Counter(toks) for toks in self.doc_tokens]

    @classmethod
    def from_json(cls, json_path: str = "data/chunks.json", **kwargs):
        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        return cls(data, **kwargs)

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
            
        scores = []
        for idx, (chunk, tf, doc_len) in enumerate(zip(self.chunks, self.doc_tfs, self.doc_lens)):
            # Apply metadata filters if specified
            if filters:
                meta = chunk.get("metadata", {})
                match = True
                for fk, fv in filters.items():
                    val = meta.get(fk, chunk.get(fk))
                    if val != fv and str(val).lower() != str(fv).lower():
                        match = False
                        break
                if not match:
                    continue
                    
            score = 0.0
            for term in q_tokens:
                if term in tf:
                    t_freq = tf[term]
                    idf = self.idf.get(term, 0.0)
                    denom = t_freq + self.k1 * (1.0 - self.b + self.b * (doc_len / max(1e-6, self.avgdl)))
                    score += idf * (t_freq * (self.k1 + 1.0)) / denom
                    
            if score > 0.0:
                scores.append((idx, score))
                
        scores.sort(key=lambda x: -x[1])
        
        results = []
        for idx, score in scores[:top_k]:
            c = self.chunks[idx]
            results.append({
                "id": c["id"],
                "text": c["text"],
                "source_id": c.get("source_id", "llama2_paper"),
                "source": c.get("source", "data/llama2_paper.pdf"),
                "page": c.get("page", c.get("metadata", {}).get("page_number", 1)),
                "metadata": c.get("metadata", {}),
                "score": round(score, 4),
                "strategy": "bm25"
            })
        return results


class HybridRetriever:
    """Hybrid Retriever combining Dense and Lexical with Reciprocal Rank Fusion (RRF) or Normalized Score Fusion."""
    def __init__(
        self,
        dense_retriever: BaseRetriever,
        lexical_retriever: BaseRetriever,
        fusion_mode: str = "rrf",
        rrf_k: int = 60,
        alpha: float = 0.6
    ):
        self.dense = dense_retriever
        self.lexical = lexical_retriever
        self.fusion_mode = fusion_mode  # 'rrf' or 'score_fusion'
        self.rrf_k = rrf_k
        self.alpha = alpha  # weight for dense in score_fusion

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        candidate_k = max(top_k * 3, 10)
        dense_results = self.dense.search(query, filters=filters, top_k=candidate_k)
        lexical_results = self.lexical.search(query, filters=filters, top_k=candidate_k)
        
        all_chunks: dict[str, dict[str, Any]] = {}
        for r in dense_results + lexical_results:
            if r["id"] not in all_chunks:
                all_chunks[r["id"]] = r

        fused_scores: dict[str, float] = {}
        
        if self.fusion_mode == "rrf":
            # Reciprocal Rank Fusion
            for rank, r in enumerate(dense_results, start=1):
                fused_scores[r["id"]] = fused_scores.get(r["id"], 0.0) + 1.0 / (self.rrf_k + rank)
            for rank, r in enumerate(lexical_results, start=1):
                fused_scores[r["id"]] = fused_scores.get(r["id"], 0.0) + 1.0 / (self.rrf_k + rank)
        else:
            # Min-Max Normalized Score Fusion
            dense_scores = [r["score"] for r in dense_results] or [0.0]
            lex_scores = [r["score"] for r in lexical_results] or [0.0]
            
            d_min, d_max = min(dense_scores), max(dense_scores)
            l_min, l_max = min(lex_scores), max(lex_scores)
            
            def norm(s, s_min, s_max):
                return (s - s_min) / max(1e-6, s_max - s_min) if s_max > s_min else 0.5
                
            for r in dense_results:
                cid = r["id"]
                norm_d = norm(r["score"], d_min, d_max)
                fused_scores[cid] = fused_scores.get(cid, 0.0) + self.alpha * norm_d
                
            for r in lexical_results:
                cid = r["id"]
                norm_l = norm(r["score"], l_min, l_max)
                fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 - self.alpha) * norm_l

        sorted_ids = sorted(fused_scores.keys(), key=lambda cid: (-fused_scores[cid], cid))[:top_k]
        
        results = []
        for cid in sorted_ids:
            chunk = dict(all_chunks[cid])
            chunk["score"] = round(fused_scores[cid], 5)
            chunk["strategy"] = f"hybrid_{self.fusion_mode}"
            results.append(chunk)
        return results


class RerankRetriever:
    """Retrieval + Rerank Pipeline: Retrieves candidates via BaseRetriever and reranks using exact token and phrase overlap."""
    def __init__(self, base_retriever: BaseRetriever, candidate_multiplier: int = 3):
        self.base_retriever = base_retriever
        self.candidate_multiplier = candidate_multiplier

    def search(self, query: str, *, filters: dict[str, Any] | None = None, top_k: int = 3) -> list[dict[str, Any]]:
        candidate_k = max(top_k * self.candidate_multiplier, 10)
        candidates = self.base_retriever.search(query, filters=filters, top_k=candidate_k)
        if not candidates:
            return []
            
        q_tokens = tokenize(query)
        q_set = set(q_tokens)
        
        reranked = []
        for c in candidates:
            c_tokens = tokenize(c["text"])
            c_set = set(c_tokens)
            
            # 1. Base retrieval score
            base_score = c.get("score", 0.0)
            
            # 2. Token overlap (Jaccard-like)
            overlap = len(q_set & c_set) / max(1, len(q_set))
            
            # 3. Exact phrase match bonus
            phrase_bonus = 0.0
            if len(query.strip()) >= 5 and query.lower() in c["text"].lower():
                phrase_bonus = 0.5
            elif any(w in c["text"].lower() for w in query.lower().split() if len(w) > 4):
                phrase_bonus = 0.1
                
            # Composite rerank score
            rerank_score = round(0.5 * base_score + 0.3 * overlap + 0.2 * phrase_bonus, 4)
            
            item = dict(c)
            item["score"] = rerank_score
            item["strategy"] = "retrieval_rerank"
            reranked.append(item)
            
        reranked.sort(key=lambda x: (-x["score"], x["id"]))
        return reranked[:top_k]

