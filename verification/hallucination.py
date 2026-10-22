import logging
import numpy as np
from typing import List, Dict, Any, Optional
from core.embedding_service import EmbeddingService

logger = logging.getLogger("HallucinationChecker")

class HallucinationChecker:
    """Evaluates context grounding and flags unsupported outputs."""
    @classmethod
    def calculate_grounding(cls, response: str, docs: List[Dict[str, Any]]) -> float:
        if not response or not docs:
            return 0.0
        
        corpus = " ".join([d.get("text", "") for d in docs])
        if not corpus.strip():
            return 0.0

        try:
            # Semantic similarity using the embedding service if neural model active
            if EmbeddingService.get_model() != "MOCK_MODEL":
                response_vec = EmbeddingService.embed(response)
                context_vec = EmbeddingService.embed(corpus)
                
                # Compute cosine similarity
                resp_norm = response_vec / (np.linalg.norm(response_vec) + 1e-12)
                ctx_norm = context_vec / (np.linalg.norm(context_vec) + 1e-12)
                cosine_sim = float(np.dot(resp_norm, ctx_norm))
                
                # Map [-1, 1] to [0, 1] for grounding score
                grounding_score = max(0.0, (cosine_sim + 1.0) / 2.0)
                return grounding_score
            else:
                # Fast, deterministic lexical and salient token grounding for CPU/mock mode
                return cls._compute_lexical_grounding(response, corpus)
        except Exception as e:
            logger.error(f"Failed to calculate semantic grounding: {e}")
            return 0.0

    @classmethod
    def _compute_lexical_grounding(cls, response: str, corpus: str) -> float:
        """Measure what fraction of non-stopword salient response tokens are supported by corpus."""
        import re
        resp_words = [w for w in re.findall(r"[a-zA-Z0-9]+", response.lower()) if len(w) > 1]
        corp_words = set(re.findall(r"[a-zA-Z0-9]+", corpus.lower()))
        if not resp_words or not corp_words:
            return 0.0

        stopwords = {
            "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "of", "with",
            "by", "is", "are", "was", "were", "be", "been", "it", "its", "this", "that",
            "these", "those", "must", "should", "can", "as", "from", "into", "all", "our"
        }
        salient_resp = [w for w in resp_words if w not in stopwords]
        if not salient_resp:
            salient_resp = resp_words

        matched = sum(1 for w in salient_resp if w in corp_words)
        overlap_ratio = matched / len(salient_resp)

        if overlap_ratio == 0:
            return 0.0
        # Map overlap to score: 20%+ overlap passes 0.35 threshold; 75%+ maps to > 0.78
        score = 0.20 + 0.78 * overlap_ratio
        return round(min(1.0, score), 4)

    @classmethod
    def evaluate(
        cls,
        integrity_passed: bool,
        completeness_passed: bool,
        retrieved_docs: List[Dict[str, Any]],
        llm_response: Optional[str] = None,
        min_grounding_score: float = 0.35
    ) -> Dict[str, Any]:
        if not integrity_passed:
            return {
                "flagged": True,
                "type": "INTEGRITY_CHECK_FAILURE",
                "reason": "HALLUCINATION_RISK: Cryptographic Merkle tree verification failed."
            }

        if not completeness_passed:
            return {
                "flagged": True,
                "type": "IVF_BOUNDARY_INCOMPLETENESS",
                "reason": "HALLUCINATION_RISK: Search failed to probe mandatory IVF Voronoi clusters."
            }

        if not retrieved_docs:
            return {
                "flagged": True,
                "type": "EMPTY_RETRIEVAL",
                "reason": "HALLUCINATION_RISK: Zero context chunks retrieved from vector storage."
            }

        if llm_response:
            score = cls.calculate_grounding(llm_response, retrieved_docs)
            if score < min_grounding_score:
                return {
                    "flagged": True,
                    "type": "UNGROUNDED_ANSWER",
                    "reason": f"HALLUCINATION_RISK: Low context grounding score ({score:.2f} < {min_grounding_score}).",
                    "grounding_score": round(score, 4)
                }
            return {
                "flagged": False,
                "type": "NONE",
                "reason": "Verification passed: integrity, IVF boundary completeness, and grounding satisfied.",
                "grounding_score": round(score, 4)
            }

        return {"flagged": False, "type": "NONE", "reason": "Structural verification passed."}