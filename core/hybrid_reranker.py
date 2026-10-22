"""
Hybrid Retrieval & Dynamic Reranking Engine.
Implements:
1. Lexical BM25 Scoring with query term filtering & bigram phrase boost.
2. Normalized Hybrid Score Fusion (Dense Semantic + Sparse BM25).
3. Reciprocal Rank Fusion (RRF) for rank-level combination.
4. Bounded Context Window Expansion / Adjacent Chunk Stitching (reunites split QA blocks).
5. Grounding & Query Relevance Verification (detects when requested topics are absent).
"""
import math
import re
from typing import List, Dict, Any, Tuple, Optional, Set
import numpy as np

STOPWORDS: Set[str] = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by", "of",
    "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "should", "would", "may", "might", "must", "shall",
    "what", "when", "where", "who", "whom", "which", "why", "how", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "its", "tell", "give", "show", "list", "about",
    "question", "questions", "me", "please", "can", "you", "pdf", "file", "document"
}


class HybridReranker:
    """
    Enterprise-grade Hybrid Retrieval and Dynamic Reranking Engine.
    Works directly with decrypted chunks in client-side secure memory.
    """

    @staticmethod
    def extract_salient_terms(query: str) -> List[str]:
        words = re.findall(r"[a-zA-Z0-9]+", query.lower())
        salient = [w for w in words if w not in STOPWORDS and len(w) > 1]
        return salient if salient else words

    @classmethod
    def check_corpus_relevance(cls, query: str, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check whether the salient query terms or phrases actually exist in the retrieved corpus.
        Helps prevent hallucinating answers for completely absent sections (e.g. career objective).
        """
        q_terms = cls.extract_salient_terms(query)
        if not q_terms:
            return {"has_relevant_terms": True, "matched_terms": [], "max_overlap": 0}

        corpus_text = " ".join([d.get("text", "") for d in documents]).lower()
        corpus_words = set(re.findall(r"[a-zA-Z0-9]+", corpus_text))

        matched = [w for w in q_terms if w in corpus_words]

        # Check bigrams for exact phrases
        q_bigrams = [f"{q_terms[i]} {q_terms[i+1]}" for i in range(len(q_terms) - 1)]
        matched_bigrams = [bg for bg in q_bigrams if bg in corpus_text]

        has_relevant = len(matched) > 0 or len(matched_bigrams) > 0
        return {
            "has_relevant_terms": has_relevant,
            "matched_terms": matched,
            "matched_bigrams": matched_bigrams,
            "salient_terms": q_terms,
            "overlap_ratio": len(matched) / len(q_terms) if q_terms else 1.0,
        }

    @staticmethod
    def simple_stem(w: str) -> str:
        """Morphological suffix stemming to match singular/plural and verb forms."""
        w = w.lower()
        if len(w) > 3 and w.endswith("ies"):
            return w[:-3] + "y"
        if len(w) > 4 and w.endswith("ing"):
            return w[:-3]
        if len(w) > 4 and w.endswith("es"):
            return w[:-2]
        if len(w) > 3 and w.endswith("ed"):
            return w[:-2]
        if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            return w[:-1]
        return w

    @classmethod
    def compute_bm25_scores(
        cls,
        query: str,
        documents: List[Dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75
    ) -> List[float]:
        """
        Compute BM25 scores with morphological stemming, saturation, length normalization,
        and section header & bigram phrase boosts.
        """
        q_terms = cls.extract_salient_terms(query)
        if not q_terms or not documents:
            return [0.0] * len(documents)

        q_stems = [cls.simple_stem(w) for w in q_terms]

        doc_raw_tokens = [re.findall(r"[a-zA-Z0-9]+", d.get("text", "").lower()) for d in documents]
        doc_stems = [[cls.simple_stem(w) for w in tokens] for tokens in doc_raw_tokens]
        doc_lens = [len(tokens) for tokens in doc_stems]
        avgdl = sum(doc_lens) / len(documents) if documents else 1.0
        n_docs = len(documents)

        # Compute document frequencies
        df: Dict[str, int] = {}
        for tokens in doc_stems:
            for w in set(tokens):
                df[w] = df.get(w, 0) + 1

        # Compute IDF
        idf: Dict[str, float] = {}
        for w in q_stems:
            freq = df.get(w, 0)
            idf[w] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

        q_bigrams = [f"{q_terms[i]} {q_terms[i+1]}" for i in range(len(q_terms) - 1)]

        scores: List[float] = []
        for i, (d, tokens, stems) in enumerate(zip(documents, doc_raw_tokens, doc_stems)):
            text_lower = d.get("text", "").lower()
            tf: Dict[str, int] = {}
            for w in stems:
                tf[w] = tf.get(w, 0) + 1

            bm25 = 0.0
            dl = doc_lens[i]
            for w in q_stems:
                if w in tf:
                    term_tf = tf[w]
                    num = idf[w] * term_tf * (k1 + 1.0)
                    den = term_tf + k1 * (1.0 - b + b * (dl / avgdl))
                    bm25 += num / den

            # Exact phrase match bonus
            phrase_bonus = 0.0
            for bg in q_bigrams:
                if bg in text_lower:
                    phrase_bonus += 3.5

            # Section header boost: if a query term appears in the first 15 tokens
            header_bonus = 0.0
            for w in q_stems:
                if w in stems[:15]:
                    header_bonus += 3.0

            scores.append(round(bm25 + phrase_bonus + header_bonus, 4))

        return scores

    @classmethod
    def hybrid_rerank(
        cls,
        documents: List[Dict[str, Any]],
        dense_scores: List[float],
        bm25_scores: List[float],
        alpha: float = 0.65,
    ) -> List[Dict[str, Any]]:
        """
        Normalized Hybrid Scoring:
        Final_Score = alpha * BM25_norm + (1 - alpha) * Dense_norm
        Prioritizes exact keyword and phrase matches while maintaining semantic coverage.
        """
        if not documents:
            return []

        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        min_dense = min(dense_scores) if dense_scores else 0.0
        max_dense = max(dense_scores) if dense_scores else 1.0
        dense_range = (max_dense - min_dense) if (max_dense - min_dense) > 1e-6 else 1.0

        scored_docs = []
        for i, doc in enumerate(documents):
            bm = bm25_scores[i]
            dn = dense_scores[i]

            bm_norm = (bm / max_bm25) if max_bm25 > 0 else 0.0
            dn_norm = (dn - min_dense) / dense_range

            if max_bm25 > 0 and bm > 0:
                final_score = alpha * bm_norm + (1.0 - alpha) * dn_norm
            elif max_bm25 > 0:
                # Query has keywords present elsewhere in the doc, but this chunk matched NONE of them
                final_score = (1.0 - alpha) * dn_norm * 0.35
            else:
                # No keywords in query, pure semantic ranking
                final_score = dn_norm

            d_copy = dict(doc)
            d_copy["score"] = round(float(final_score), 5)
            d_copy["bm25_score"] = round(float(bm), 3)
            d_copy["dense_score"] = round(float(dn), 3)
            scored_docs.append(d_copy)

        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs

    @classmethod
    def rrf_fuse(
        cls,
        documents: List[Dict[str, Any]],
        dense_scores: List[float],
        bm25_scores: List[float],
        k: int = 20,
        dense_weight: float = 0.8,
        bm25_weight: float = 2.0,
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF):
        RRF_score(d) = (dense_weight / (k + rank_dense)) + (bm25_weight / (k + rank_bm25))
        """
        n = len(documents)
        if n == 0:
            return []

        dense_ranked_indices = sorted(range(n), key=lambda i: dense_scores[i], reverse=True)
        bm25_ranked_indices = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)

        dense_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(dense_ranked_indices)}
        bm25_ranks = {doc_idx: rank + 1 for rank, doc_idx in enumerate(bm25_ranked_indices)}

        fused_docs = []
        for i, doc in enumerate(documents):
            d_rank = dense_ranks[i]
            b_rank = bm25_ranks[i]
            rrf_score = (dense_weight / (k + d_rank)) + (bm25_weight / (k + b_rank))

            doc_copy = dict(doc)
            doc_copy["dense_score"] = round(float(dense_scores[i]), 4)
            doc_copy["bm25_score"] = round(float(bm25_scores[i]), 4)
            doc_copy["score"] = round(float(rrf_score), 6)
            doc_copy["dense_rank"] = d_rank
            doc_copy["bm25_rank"] = b_rank
            fused_docs.append(doc_copy)

        fused_docs.sort(key=lambda x: x["score"], reverse=True)
        return fused_docs

    @classmethod
    def expand_and_stitch_context(
        cls,
        top_candidates: List[Dict[str, Any]],
        all_candidates_by_index: Optional[Dict[int, Dict[str, Any]]] = None,
        max_stitched_chars: int = 1200,
    ) -> List[Dict[str, Any]]:
        """
        Bounded Context Window Expansion / Adjacent Chunk Stitching:
        Expands top ranked chunks by stitching adjacent successor chunk (c_idx + 1)
        if present, deduplicating overlapping content.
        Keeps individual topic blocks focused (bounded by max_stitched_chars).
        """
        if not top_candidates:
            return []

        # If index map not provided, build from candidates
        if all_candidates_by_index is None:
            all_candidates_by_index = {
                int(d.get("chunk_index", 0)): d
                for d in top_candidates
            }

        stitched_results: List[Dict[str, Any]] = []
        seen_indices: Set[int] = set()

        for doc in top_candidates:
            c_idx = int(doc.get("chunk_index", 0))
            if c_idx in seen_indices:
                continue

            merged = dict(doc)
            cur_text = merged.get("text", "")

            # Look ahead for immediate successor chunk
            nxt = all_candidates_by_index.get(c_idx + 1)
            if nxt and (len(cur_text) < max_stitched_chars):
                nxt_text = nxt.get("text", "")
                merged_flag = False
                for ov_len in range(min(120, len(cur_text), len(nxt_text)), 15, -1):
                    suffix = cur_text[-ov_len:]
                    if nxt_text.startswith(suffix):
                        cur_text = cur_text + nxt_text[ov_len:]
                        merged_flag = True
                        break
                if not merged_flag:
                    cur_text = cur_text + "\n" + nxt_text
                seen_indices.add(c_idx + 1)

            merged["text"] = cur_text
            seen_indices.add(c_idx)
            stitched_results.append(merged)

        return stitched_results

    @classmethod
    def execute_advanced_retrieval_pipeline(
        cls,
        query: str,
        documents: List[Dict[str, Any]],
        alpha: float = 0.60,
        enable_rrf: bool = True,
        enable_cross_encoder: bool = True,
        top_k: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        End-to-End Fine-Tuned Retrieval & Reranking Pipeline:
        1. Multi-factor BM25 scoring with stemming & phrase bonuses.
        2. Rank fusion via RRF (Reciprocal Rank Fusion) or Normalized Hybrid Scoring.
        3. Cross-Encoder deep (query, chunk) interaction reranking on candidate pool.
        """
        if not documents:
            return []

        bm25_scores = cls.compute_bm25_scores(query, documents)
        dense_scores = [float(item.get("score") or 0.5) for item in documents]

        # Stage 1: Hybrid / RRF Ranking
        if enable_rrf:
            stage1_candidates = cls.rrf_fuse(
                documents=documents,
                dense_scores=dense_scores,
                bm25_scores=bm25_scores,
                k=20,
                dense_weight=0.8,
                bm25_weight=2.0,
            )
        else:
            stage1_candidates = cls.hybrid_rerank(
                documents=documents,
                dense_scores=dense_scores,
                bm25_scores=bm25_scores,
                alpha=alpha,
            )

        # Stage 2: Cross-Encoder Reranking on Top Candidates
        if enable_cross_encoder and stage1_candidates:
            from core.cross_encoder_reranker import CrossEncoderReranker
            rerank_pool = stage1_candidates[:max(top_k * 2, 20)]
            final_ranked = CrossEncoderReranker.rerank(
                query=query,
                documents=rerank_pool,
                top_k=top_k,
                text_key="text",
            )
            # Retain remainder if pool was smaller
            if len(stage1_candidates) > len(final_ranked):
                seen_ids = set(d.get("chunk_id") for d in final_ranked if d.get("chunk_id"))
                for rem in stage1_candidates:
                    cid = rem.get("chunk_id")
                    if cid and cid not in seen_ids:
                        final_ranked.append(rem)
                        seen_ids.add(cid)
            return final_ranked

        return stage1_candidates[:top_k]



