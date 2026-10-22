"""
Post-Retrieval Context Optimizer.
Implements:
1. Lost-in-the-Middle Mitigation: Reorders top-ranked candidates into a U-shaped distribution
   (placing the highest-relevance evidence at the start and end of the context window).
2. Context Compression / Sentence Pruning: Removes low-relevance filler sentences to save prompt tokens
   and eliminate distraction.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("ContextOptimizer")


class ContextOptimizer:
    """
    Optimizes retrieved context chunks before passing them to the generator LLM.
    """

    @classmethod
    def reorder_lost_in_the_middle(
        cls, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Lost-in-the-Middle Mitigation (U-Shaped Attention Distribution).
        Reorders sorted documents so highest-relevance documents appear at
        the beginning and end of the prompt context window, where LLM attention is strongest.
        Order: [Doc 0, Doc 2, Doc 4, ..., Doc 5, Doc 3, Doc 1]
        """
        if len(documents) <= 2:
            return documents

        reordered: List[Dict[str, Any]] = [None] * len(documents)  # type: ignore
        left = 0
        right = len(documents) - 1

        for i, doc in enumerate(documents):
            if i % 2 == 0:
                reordered[left] = doc
                left += 1
            else:
                reordered[right] = doc
                right -= 1

        return [d for d in reordered if d is not None]

    @classmethod
    def compress_chunk_content(
        cls,
        text: str,
        query: str,
        min_sentence_chars: int = 15,
        relevance_threshold: float = 0.15,
    ) -> str:
        """
        Context Compression: Prunes irrelevant, redundant, or filler sentences.
        Preserves section titles, bullet lines, and sentences containing query keywords.
        """
        if len(text) < 200:
            return text

        q_terms = set(re.findall(r"[a-zA-Z0-9]+", query.lower()))
        lines = text.splitlines()
        retained_lines: List[str] = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # Keep short section headers or bullet markers intact
            if stripped.endswith(":") or re.match(r"^[-*•●■\d+.]\s+", stripped):
                retained_lines.append(line)
                continue

            # Tokenize line
            line_terms = set(re.findall(r"[a-zA-Z0-9]+", stripped.lower()))
            if not line_terms:
                continue

            # Calculate query term overlap
            overlap = len(q_terms.intersection(line_terms))
            overlap_ratio = overlap / len(q_terms) if q_terms else 0.5

            # Keep sentence if it contains salient terms or passes threshold
            if overlap > 0 or overlap_ratio >= relevance_threshold:
                retained_lines.append(line)

        compressed = "\n".join(retained_lines).strip()
        return compressed if len(compressed) >= 50 else text

    @classmethod
    def optimize_context_for_llm(
        cls,
        documents: List[Dict[str, Any]],
        query: str,
        enable_lost_in_the_middle: bool = True,
        enable_compression: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Full post-retrieval context refinement pipeline:
        1. Context compression (optional token pruning)
        2. U-shaped Lost-in-the-Middle reordering
        """
        if not documents:
            return []

        processed_docs = []
        for doc in documents:
            d_copy = dict(doc)
            if enable_compression:
                d_copy["text"] = cls.compress_chunk_content(d_copy.get("text", ""), query)
            processed_docs.append(d_copy)

        if enable_lost_in_the_middle:
            processed_docs = cls.reorder_lost_in_the_middle(processed_docs)

        return processed_docs
