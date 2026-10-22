"""
Query Transformation & Pre-Retrieval Enhancement Engine.
Implements:
1. Multi-Query Expansion: Generates lexical and semantic variations of ambiguous queries.
2. Hypothetical Document Embeddings (HyDE): Generates a pseudo-answer document for answer-to-answer vector search.
3. Sub-Query Decomposition: Splits multi-part questions into targeted atomic queries.
"""
from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("QueryTransformation")

SYNONYM_MAP: Dict[str, List[str]] = {
    "projects": ["personal projects", "portfolio", "technical projects", "systems developed", "software engineering projects"],
    "vpn": ["virtual private network", "remote access", "vpn credentials", "vpn policy"],
    "rotate": ["rotation frequency", "credential expiration", "rotation schedule", "change password"],
    "security": ["cybersecurity", "security policies", "access control", "compliance", "protocols"],
    "encryption": ["aes-256", "cryptographic standards", "ciphers", "data protection"],
    "experience": ["work history", "employment", "professional experience", "roles"],
    "education": ["degrees", "university", "academic background", "certifications"],
    "policy": ["compliance rules", "enterprise guidelines", "mandatory policy", "governance"],
}


class QueryTransformer:
    """
    Transforms user queries before vector and keyword retrieval to bridge
    the vocabulary gap between short user questions and detailed document passages.
    """

    @classmethod
    def expand_query(cls, query: str, max_variations: int = 3) -> List[str]:
        """
        Generate semantic variations and reformulated queries for Multi-Query Retrieval.
        Ensures diverse lexical and semantic coverage across Qdrant and BM25.
        """
        clean_q = query.strip()
        variations: List[str] = [clean_q]
        words = re.findall(r"[a-zA-Z0-9]+", clean_q.lower())

        # Check for synonyms and expansions
        expanded_phrases = []
        for w in words:
            if w in SYNONYM_MAP:
                expanded_phrases.extend(SYNONYM_MAP[w][:2])

        if expanded_phrases:
            variations.append(f"{clean_q} {' '.join(expanded_phrases[:3])}")

        # Inverted / Statement variation for HyDE-like matching
        if clean_q.lower().startswith(("what is", "what are", "how often", "how do", "list", "tell me")):
            stripped_q = re.sub(r"^(?:what\s+is|what\s+are|how\s+often|how\s+do|list|tell\s+me\s+about)\s+", "", clean_q, flags=re.I)
            variations.append(f"document policy regarding {stripped_q.rstrip('?')}")

        return list(dict.fromkeys(variations))[:max_variations]

    @classmethod
    def generate_hyde_document(cls, query: str) -> str:
        """
        Hypothetical Document Embeddings (HyDE).
        Generates a synthetic hypothetical answer passage matching the query structure.
        Embedding the hypothetical answer achieves higher vector alignment with real document answers.
        """
        q_lower = query.lower()
        if "vpn" in q_lower or "rotat" in q_lower:
            return (
                f"Policy Specification: Regarding '{query}', all enterprise credentials, "
                f"tokens, and encryption keys must be rotated according to standard schedules. "
                f"Access policies mandate periodic expiration and re-authentication for all users."
            )
        elif "project" in q_lower or "resume" in q_lower:
            return (
                f"Candidate Portfolio Overview: The candidate has engineered multiple software projects, "
                f"architecting high-throughput distributed systems, stream processing engines, and backend services."
            )
        elif "encrypt" in q_lower or "security" in q_lower:
            return (
                f"Information Security Standards: All enterprise data at rest and in transit is protected "
                f"using AES-256-GCM encryption, secure nonces, and cryptographic access control policies."
            )
        else:
            return (
                f"Official documentation details regarding {query}. "
                f"The system provides complete specifications, requirements, and operational guidelines."
            )

    @classmethod
    def decompose_subqueries(cls, query: str) -> List[str]:
        """
        Sub-Query Decomposition.
        Detects compound, multi-part, or comparative queries and splits them into atomic searches.
        Example: 'What are the VPN policies and how are passwords stored?' -> ['VPN policies', 'how passwords stored']
        """
        split_patterns = [
            r"\s+and\s+(?:what|how|where|who|when)\s+",
            r"\s+as\s+well\s+as\s+",
            r"\s+compare\s+(?:it\s+)?(?:with|to)\s+",
            r";\s*",
        ]
        segments = [query]
        for pat in split_patterns:
            new_segs = []
            for seg in segments:
                parts = re.split(pat, seg, flags=re.I)
                new_segs.extend([p.strip() for p in parts if len(p.strip()) > 3])
            segments = new_segs

        return segments if len(segments) > 1 else [query]
