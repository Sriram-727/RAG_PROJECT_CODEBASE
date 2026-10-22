"""
Cross-Encoder Reranker for High-Precision RAG Retrieval.
Implements:
1. Joint (query, document) deep semantic scoring via SentenceTransformers CrossEncoder.
2. Graceful fast-path heuristic cross-scoring when offline or during zero-dependency testing.
3. Batch scoring with score normalization.
"""
from __future__ import annotations

import logging
import math
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("CrossEncoderReranker")


class HeuristicCrossScorer:
    """
    High-performance fallback cross-scorer that simulates cross-attention interaction:
    evaluates term overlap, exact phrase alignment, header proximity, and lexical coverage.
    """
    STOPWORDS = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by", "of",
        "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "should", "would", "may", "might", "must", "shall",
        "what", "when", "where", "who", "whom", "which", "why", "how", "this", "that", "these",
        "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "their", "our", "its", "tell", "give", "show", "list", "about"
    }

    @classmethod
    def score_pair(cls, query: str, text: str) -> float:
        q_tokens = [w for w in re.findall(r"[a-zA-Z0-9]+", query.lower()) if w not in cls.STOPWORDS and len(w) > 1]
        if not q_tokens:
            q_tokens = re.findall(r"[a-zA-Z0-9]+", query.lower())
        if not q_tokens or not text:
            return 0.1

        t_lower = text.lower()
        t_tokens = re.findall(r"[a-zA-Z0-9]+", t_lower)
        if not t_tokens:
            return 0.1

        # 1. Term coverage ratio
        matched_tokens = [w for w in q_tokens if w in t_tokens]
        coverage = len(matched_tokens) / len(q_tokens) if q_tokens else 0.0

        # 2. Bigram exact phrase match bonus
        bigram_bonus = 0.0
        for i in range(len(q_tokens) - 1):
            bg = f"{q_tokens[i]} {q_tokens[i+1]}"
            if bg in t_lower:
                bigram_bonus += 0.25

        # 3. Header prominence (if term appears in first 20 tokens)
        header_bonus = 0.0
        lead_tokens = set(t_tokens[:25])
        for w in q_tokens:
            if w in lead_tokens:
                header_bonus += 0.15

        # 4. Term density in matched regions
        tf_count = sum(t_tokens.count(w) for w in q_tokens)
        density_factor = min(0.2, (tf_count / (len(t_tokens) + 1)) * 3.0)

        raw_score = (coverage * 0.5) + bigram_bonus + header_bonus + density_factor
        # Sigmoid compression to [0, 1] range
        norm_score = 1.0 / (1.0 + math.exp(-3.0 * (raw_score - 0.5)))
        return round(float(norm_score), 5)


class CrossEncoderReranker:
    """
    Reranks candidate document chunks against user query using deep cross-attention.
    Operates on top-K retrieved candidates from the first-stage hybrid retrieval.
    """
    _model: Optional[Any] = None
    _model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    _load_attempted: bool = False

    @classmethod
    def get_model(cls) -> Optional[Any]:
        if cls._model is None and not cls._load_attempted:
            cls._load_attempted = True
            force_mock = os.environ.get("USE_MOCK_CROSS_ENCODER", "").lower() in ("true", "1", "yes")
            if not force_mock:
                try:
                    from sentence_transformers import CrossEncoder
                    logger.info(f"Attempting to load CrossEncoder '{cls._model_name}'...")
                    try:
                        cls._model = CrossEncoder(cls._model_name, max_length=512, local_files_only=True)
                        logger.info("CrossEncoder loaded successfully from local cache.")
                    except Exception as le:
                        logger.info(f"Local CrossEncoder weights not pre-cached ({le}). Using fast offline scorer.")
                        cls._model = None
                except Exception as e:
                    logger.warning(f"Could not load CrossEncoder model ({e}). Using HeuristicCrossScorer fallback.")
                    cls._model = None
            else:
                logger.info("Using fast heuristic CrossScorer.")
                cls._model = None
        return cls._model

    @classmethod
    def rerank(
        cls,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 10,
        text_key: str = "text",
    ) -> List[Dict[str, Any]]:
        """
        Rerank documents by computing deep relevance score for each (query, document) pair.
        """
        if not documents:
            return []

        model = cls.get_model()
        scored_docs: List[Dict[str, Any]] = []

        if model is not None:
            pairs: List[Tuple[str, str]] = [
                (query, str(doc.get(text_key, ""))[:512]) for doc in documents
            ]
            try:
                raw_scores = model.predict(pairs, show_progress_bar=False)
                # Convert logits to probability via sigmoid if needed
                for doc, score in zip(documents, raw_scores):
                    prob_score = 1.0 / (1.0 + math.exp(-float(score))) if isinstance(score, (int, float)) else 0.5
                    doc_copy = dict(doc)
                    doc_copy["cross_encoder_score"] = round(float(prob_score), 5)
                    # Blend with base score for stability
                    base_score = float(doc.get("score", 0.5))
                    doc_copy["score"] = round(0.7 * prob_score + 0.3 * base_score, 5)
                    scored_docs.append(doc_copy)
            except Exception as pe:
                logger.warning(f"CrossEncoder inference failed ({pe}). Falling back to heuristic scorer.")
                model = None

        if model is None:
            for doc in documents:
                text = str(doc.get(text_key, ""))
                h_score = HeuristicCrossScorer.score_pair(query, text)
                base_score = float(doc.get("score", 0.5))
                doc_copy = dict(doc)
                doc_copy["cross_encoder_score"] = round(h_score, 5)
                # Composite score prioritizing cross-interaction
                doc_copy["score"] = round(0.65 * h_score + 0.35 * base_score, 5)
                scored_docs.append(doc_copy)

        scored_docs.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return scored_docs[:top_k]
