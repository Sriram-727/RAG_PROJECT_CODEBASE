"""
Hierarchical (Parent-Child) Chunking Engine.
Implements:
1. Two-Tier Segmentation: Larger Parent Chunks (context) + Smaller Child Chunks (search resolution).
2. Parent Context Expansion: Resolves retrieved child hits to parent chunks before LLM synthesis.
3. Deduplication: Ensures multiple child matches from the same parent do not duplicate context.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger("HierarchicalChunker")


@dataclass
class ChildChunk:
    child_id: str
    parent_id: str
    text: str
    child_index: int
    char_start: int
    char_end: int


@dataclass
class ParentChunk:
    parent_id: str
    text: str
    parent_index: int
    children: List[ChildChunk] = field(default_factory=list)


class HierarchicalChunker:
    """
    Splits text into larger Parent chunks (for LLM context)
    and granular Child chunks (for high-precision vector search).
    """

    @classmethod
    def create_hierarchical_chunks(
        cls,
        text: str,
        doc_id: str,
        parent_size: int = 900,
        parent_overlap: int = 100,
        child_size: int = 250,
        child_overlap: int = 40,
        
    ) -> Tuple[List[ParentChunk], List[ChildChunk]]:
        """
        Produce two-tier parent and child chunks with bidirectional linking.
        """
        clean_text = text.strip()
        if not clean_text:
            return [], []

        # Step 1: Create Parent Chunks
        parent_chunks: List[ParentChunk] = []
        all_children: List[ChildChunk] = []

        start = 0
        p_idx = 0
        while start < len(clean_text):
            end = start + parent_size
            if end < len(clean_text):
                # Prefer splitting at paragraph or newline boundary
                break_pt = clean_text.rfind("\n\n", start + parent_size // 2, end)
                if break_pt == -1:
                    break_pt = clean_text.rfind("\n", start + parent_size // 2, end)
                if break_pt == -1:
                    break_pt = clean_text.rfind(". ", start + parent_size // 2, end)
                    if break_pt != -1:
                        break_pt += 1
                if break_pt != -1 and break_pt > start:
                    end = break_pt

            p_text = clean_text[start:end].strip()
            if p_text:
                pid = f"p_{doc_id}_{p_idx}"
                parent_obj = ParentChunk(parent_id=pid, text=p_text, parent_index=p_idx)

                # Step 2: Subdivide Parent into Child Chunks
                c_start = 0
                c_idx = 0
                while c_start < len(p_text):
                    c_end = c_start + child_size
                    if c_end < len(p_text):
                        c_break = p_text.rfind(". ", c_start + child_size // 2, c_end)
                        if c_break != -1:
                            c_end = c_break + 1
                        else:
                            c_break = p_text.rfind(" ", c_start + child_size // 2, c_end)
                            if c_break != -1 and c_break > c_start:
                                c_end = c_break

                    c_text = p_text[c_start:c_end].strip()
                    if c_text:
                        cid = f"c_{doc_id}_{p_idx}_{c_idx}"
                        child_obj = ChildChunk(
                            child_id=cid,
                            parent_id=pid,
                            text=c_text,
                            child_index=c_idx,
                            char_start=c_start,
                            char_end=c_end,
                        )
                        parent_obj.children.append(child_obj)
                        all_children.append(child_obj)
                        c_idx += 1

                    c_start = max(c_start + 1, c_end - child_overlap)
                    if c_start >= len(p_text):
                        break

                parent_chunks.append(parent_obj)
                p_idx += 1

            start = max(start + 1, end - parent_overlap)
            if start >= len(clean_text):
                break

        logger.info(
            f"Hierarchical Chunking: {len(clean_text)} chars -> "
            f"{len(parent_chunks)} parent chunks, {len(all_children)} child chunks."
        )
        return parent_chunks, all_children

    @classmethod
    def resolve_child_to_parents(
        cls,
        retrieved_child_hits: List[Dict[str, Any]],
        parent_map: Dict[str, str],
        max_parents: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Maps child search hits back to broader parent chunks.
        Prevents LLM from receiving fragmented context while preserving child similarity scores.
        """
        seen_parents: Set[str] = set()
        expanded_docs: List[Dict[str, Any]] = []

        for hit in retrieved_child_hits:
            pid = hit.get("parent_id") or hit.get("metadata", {}).get("parent_id")
            if pid and pid in parent_map:
                if pid not in seen_parents:
                    seen_parents.add(pid)
                    hit_copy = dict(hit)
                    hit_copy["text"] = parent_map[pid]
                    hit_copy["is_expanded_parent"] = True
                    hit_copy["parent_id"] = pid
                    expanded_docs.append(hit_copy)
            else:
                # Fall back to original chunk if parent not indexed
                expanded_docs.append(hit)

            if len(expanded_docs) >= max_parents:
                break

        return expanded_docs
