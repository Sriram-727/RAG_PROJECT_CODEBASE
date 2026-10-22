"""Leaf models for User MMR (chunk leaves) and Cloud MMR (user root leaves)."""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field
from crypto.hashing import hash_chunk_leaf


class ChunkLeaf(BaseModel):
    """Leaf representation of an encrypted chunk in the User MMR."""
    leaf_index: int
    leaf_hash: str
    chunk_id: str
    doc_id: str
    user_id: str
    org_id: str
    chunk_index: int
    content_hash: str
    point_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_chunk_record(cls, leaf_index: int, chunk: Any) -> ChunkLeaf:
        """Construct ChunkLeaf and compute its canonical hash."""
        computed_hash = hash_chunk_leaf(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            user_id=chunk.user_id,
            org_id=chunk.org_id,
            chunk_index=chunk.chunk_index,
            content_hash=chunk.chunk_hash,
        )
        return cls(
            leaf_index=leaf_index,
            leaf_hash=computed_hash,
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            user_id=chunk.user_id,
            org_id=chunk.org_id,
            chunk_index=chunk.chunk_index,
            content_hash=chunk.chunk_hash,
            point_id=getattr(chunk, "point_id", None),
            metadata=getattr(chunk, "metadata", {}).model_dump()
            if hasattr(getattr(chunk, "metadata", None), "model_dump")
            else {},
        )


class UserRootLeaf(BaseModel):
    """Leaf representation of a User MMR Root in the Cloud MMR."""
    user_index: int
    user_id: str
    user_root_hash: str
    chunk_count: int
    bagged_peak: str
