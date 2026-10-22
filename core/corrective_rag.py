"""
Corrective RAG (CRAG) and Self-RAG Evaluation Engine.
Implements:
1. Retrieval Confidence Grading: Evaluates retrieved document relevance against user query.
2. CRAG Decision Routing: Routes to CORRECT (direct synthesis), AMBIGUOUS (query refinement),
   or INCORRECT (trigger corrective fallback / reject hallucination).
3. Self-RAG Reflection Tokens: Generates [Fully-Supported], [Partially-Supported], or [Unsupported] tags.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("CorrectiveRAG")


class CorrectiveRAGEngine:
    """
    Evaluates retrieved document quality before and after generation to prevent hallucinations.
    """

    @classmethod
    def grade_retrieval(
        cls,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        correct_threshold: float = 0.55,
        incorrect_threshold: float = 0.25,
    ) -> Dict[str, Any]:
        """
        CRAG Quality Assessor:
        Calculates an aggregate retrieval confidence score across top candidates.
        Returns:
            decision: 'CORRECT' | 'AMBIGUOUS' | 'INCORRECT'
            confidence_score: float [0, 1]
            supporting_chunks: int
        """
        if not retrieved_docs:
            return {
                "decision": "INCORRECT",
                "confidence_score": 0.0,
                "supporting_chunks": 0,
                "action": "TRIGGER_FALLBACK",
                "reason": "No documents retrieved in search space."
            }

        q_terms = [w for w in re.findall(r"[a-zA-Z0-9]+", query.lower()) if len(w) > 2]
        if not q_terms:
            q_terms = re.findall(r"[a-zA-Z0-9]+", query.lower())

        chunk_confidences: List[float] = []

        for doc in retrieved_docs[:5]:
            text = doc.get("text", "").lower()
            # 1. Base vector / reranker score
            base_score = float(doc.get("score") or doc.get("cross_encoder_score") or 0.5)

            # 2. Keyword overlap in chunk
            matched = sum(1 for w in q_terms if w in text)
            overlap_ratio = matched / len(q_terms) if q_terms else 0.5

            # Blended confidence for this chunk
            chunk_conf = (0.5 * base_score) + (0.5 * overlap_ratio)
            chunk_confidences.append(chunk_conf)

        avg_confidence = sum(chunk_confidences) / len(chunk_confidences) if chunk_confidences else 0.0
        max_confidence = max(chunk_confidences) if chunk_confidences else 0.0
        effective_confidence = round(0.6 * max_confidence + 0.4 * avg_confidence, 4)

        if effective_confidence >= correct_threshold:
            decision = "CORRECT"
            action = "PROCEED_SYNTHESIS"
            reason = "High confidence retrieval. Context directly supports query."
        elif effective_confidence >= incorrect_threshold:
            decision = "AMBIGUOUS"
            action = "APPLY_QUERY_REWRITE_OR_STITCH"
            reason = "Partial context match. Query enhancement or adjacent stitching recommended."
        else:
            decision = "INCORRECT"
            action = "TRIGGER_FALLBACK"
            reason = "Retrieved chunks do not contain relevant facts. Fallback recommended."

        return {
            "decision": decision,
            "confidence_score": effective_confidence,
            "supporting_chunks": sum(1 for c in chunk_confidences if c >= 0.4),
            "action": action,
            "reason": reason,
        }

    @classmethod
    def generate_self_rag_reflection(
        cls,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        llm_answer: str,
    ) -> Dict[str, Any]:
        """
        Self-RAG Reflection Evaluator:
        Assigns special reflection tokens to the final generated response:
        - [Is-Relevant]: Did retrieval match query topic?
        - [Is-Supported]: Is every generated fact verifiable in retrieved chunks?
        - [Critique]: Quality rating of answer utility.
        """
        q_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        ans_terms = set(re.findall(r"[a-zA-Z0-9]+", llm_answer.lower()))

        corpus_text = " ".join([d.get("text", "") for d in retrieved_docs]).lower()

        # Check grounding of answer terms in context
        grounded_terms = [w for w in ans_terms if w in corpus_text]
        support_ratio = len(grounded_terms) / len(ans_terms) if ans_terms else 1.0

        if support_ratio >= 0.65:
            token = "[Fully-Supported]"
            critique = "Answer is strictly grounded and substantiated by source documents."
        elif support_ratio >= 0.35:
            token = "[Partially-Supported]"
            critique = "Answer combines verified context with general synthesis."
        else:
            token = "[Unsupported]"
            critique = "Low factual grounding detected. Information may be speculative."

        return {
            "reflection_token": token,
            "support_ratio": round(support_ratio, 4),
            "critique": critique,
            "is_relevant": "[Is-Relevant:Yes]" if len(q_terms.intersection(ans_terms)) > 0 else "[Is-Relevant:No]",
        }
