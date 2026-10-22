"""Pydantic data schemas for Qdrant payload, document chunks, and organization datasets."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata associated with an individual document chunk."""
    filename: Optional[str] = None
    mime_type: Optional[str] = "text/plain"
    tokens: Optional[int] = None
    parent_doc_id: Optional[str] = None
    created_at: float = Field(default_factory=time.time)
    custom_attributes: Dict[str, Any] = Field(default_factory=dict)


class EncryptedChunkRecord(BaseModel):
    """
    Representation of an encrypted document chunk stored in Qdrant.
    Includes ciphertext, nonce, cryptographic hash, vector embedding, and ownership metadata.
    """
    point_id: str
    chunk_id: str
    doc_id: str
    org_id: str
    user_id: str
    chunk_index: int
    encrypted_content: str  # Base64 encoded ciphertext (AES-256-GCM)
    nonce: str              # Base64 encoded 12-byte IV/nonce
    chunk_hash: str         # SHA-256 hex of original plaintext
    metadata: ChunkMetadata = Field(default_factory=ChunkMetadata)
    vector: Optional[List[float]] = None
    leaf_hash: Optional[str] = None
    proof_path: Optional[Any] = None
    enterprise_root: Optional[str] = None
    cluster_id: Optional[str] = None

    def to_qdrant_payload(self) -> Dict[str, Any]:
        """Convert record to Qdrant payload dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "org_id": self.org_id,
            "user_id": self.user_id,
            "chunk_index": self.chunk_index,
            "encrypted_content": self.encrypted_content,
            "nonce": self.nonce,
            "chunk_hash": self.chunk_hash,
            "metadata": self.metadata.model_dump(),
            "leaf_hash": self.leaf_hash,
            "proof_path": self.proof_path,
            "enterprise_root": self.enterprise_root,
            "cluster_id": str(self.cluster_id) if self.cluster_id is not None else "0",
        }

    @classmethod
    def from_qdrant_record(
        cls,
        point_id: str,
        payload: Dict[str, Any],
        vector: Optional[List[float]] = None,
    ) -> EncryptedChunkRecord:
        """Construct record from Qdrant scroll point result."""
        meta_dict = payload.get("metadata", {})
        metadata = ChunkMetadata(**meta_dict) if isinstance(meta_dict, dict) else ChunkMetadata()
        return cls(
            point_id=str(point_id),
            chunk_id=str(payload.get("chunk_id", "")),
            doc_id=str(payload.get("doc_id", "")),
            org_id=str(payload.get("org_id", "")),
            user_id=str(payload.get("user_id", "")),
            chunk_index=int(payload.get("chunk_index", 0)),
            encrypted_content=str(payload.get("encrypted_content", "")),
            nonce=str(payload.get("nonce", "")),
            chunk_hash=str(payload.get("chunk_hash", "")),
            metadata=metadata,
            vector=vector,
            leaf_hash=payload.get("leaf_hash"),
            proof_path=payload.get("proof_path"),
            enterprise_root=payload.get("enterprise_root"),
            cluster_id=str(payload.get("cluster_id", "0")),
        )


class DocumentRecord(BaseModel):
    """Group of chunks belonging to a single document."""
    doc_id: str
    user_id: str
    org_id: str
    chunks: List[EncryptedChunkRecord] = Field(default_factory=list)


class OrganizationDataset(BaseModel):
    """Complete dataset extracted for an organization."""
    org_id: str
    users: Dict[str, List[EncryptedChunkRecord]] = Field(default_factory=dict)
    total_chunks: int = 0
    total_users: int = 0
    total_docs: int = 0
