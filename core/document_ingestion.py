"""
Document Ingestion and Preprocessing Engine
Implements:
1. Zero-dependency text & PDF extraction (using pypdf)
2. Sliding-window chunking (500 characters, 50 overlap)
3. Cryptographic AES-256-GCM encryption & SHA-256 hashing
4. Dense embedding generation (384-dim)
5. Voronoi coarse-quantizer cluster assignment
6. Two-tier Merkle Mountain Range (MMR) tree generation
7. Secure indexing into Qdrant vector database
"""
from __future__ import annotations

import io
import logging
import re
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from config.settings import get_settings
from core.embedding_service import EmbeddingService
from crypto.encryption import encrypt_aes_gcm
from crypto.hashing import sha256_hex
from database.qdrant_repository import QdrantRepository
from database.schemas import ChunkMetadata, EncryptedChunkRecord
from mmr.cloud_mmr import CloudMMR
from mmr.user_mmr import UserMMR
from verification.boundary_validation import IVFBoundaryValidator

logger = logging.getLogger("DocumentIngestionEngine")


def extract_text_from_pdf(pdf_source: Union[str, bytes, io.BytesIO]) -> str:
    """Extract full plaintext from a PDF file using pypdf."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise RuntimeError("pypdf is not installed. Please run: pip install pypdf")

    reader = PdfReader(pdf_source)
    extracted_text = ""
    for page_idx, page in enumerate(reader.pages):
        page_text = page.extract_text()
        if page_text:
            extracted_text += page_text + "\n"

    logger.info(f"Extracted {len(extracted_text)} characters from {len(reader.pages)} PDF pages.")
    return extracted_text.strip()


SECTION_PATTERNS = [
    r"^#{1,6}\s+",
    r"^[A-Z][A-Z\s&/,()\-]{2,}:?\s*$",
    r"^\d+\.\s+[A-Z]",
    r"^(?:Section|Chapter|Policy|Requirement|Standard|Overview|Objective|Experience|Education|Projects?)\b",
    r"^[-*•●■]\s+",
]


def classify_chunk_section(chunk_text: str) -> str:
    """Classify chunk content domain for metadata filtering and access routing."""
    text_lower = chunk_text.lower()
    categories = {
        "security": ["security", "encryption", "access control", "authentication", "firewall", "tls", "aes", "vpn", "mfa"],
        "policy": ["policy", "compliance", "regulation", "standard", "requirement", "audit", "governance"],
        "projects": ["project", "projects", "architecture", "system", "engine", "implemented", "kafka", "flink", "docker"],
        "experience": ["experience", "employment", "role", "engineer", "responsibilities", "developer", "lead"],
        "education": ["education", "degree", "university", "college", "bachelor", "master", "gpa", "b.tech", "m.s."],
        "financial": ["revenue", "budget", "cost", "expense", "profit", "financial", "pricing", "invoices"],
    }
    for cat, kws in categories.items():
        if any(kw in text_lower for kw in kws):
            return cat
    return "general"


def chunk_text(
    text: str,
    chunk_size: int = 500,
    overlap: int = 50,
    use_semantic: bool = True
) -> List[str]:
    """
    Split document text into overlapping chunks.
    When use_semantic is True, groups lines by structural section boundaries
    (e.g., Markdown headers, capitalized section headers, bullet lists)
    before breaking into bounded chunks, keeping topics and QA pairs intact.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    if not use_semantic:
        return _sliding_window_chunks(clean_text, chunk_size, overlap)

    lines = clean_text.splitlines()
    sections: List[str] = []
    current_sec: List[str] = []

    for line in lines:
        stripped = line.strip()
        is_header = any(re.match(pat, stripped) for pat in SECTION_PATTERNS)
        if is_header and current_sec:
            sections.append("\n".join(current_sec).strip())
            current_sec = [line]
        else:
            current_sec.append(line)

    if current_sec:
        sections.append("\n".join(current_sec).strip())

    chunks: List[str] = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        if len(sec) <= chunk_size:
            chunks.append(sec)
        else:
            # Segment section into sentence or newline bounded chunks
            chunks.extend(_sliding_window_chunks(sec, chunk_size, overlap))

    # Merge consecutive tiny fragments with neighbor to prevent orphaned header fragments
    merged_chunks: List[str] = []
    buffer = ""
    for chk in chunks:
        if buffer and (len(buffer) + len(chk) + 1 <= chunk_size):
            buffer = buffer + "\n" + chk
        else:
            if buffer:
                merged_chunks.append(buffer.strip())
            buffer = chk
    if buffer:
        merged_chunks.append(buffer.strip())

    result = [c for c in merged_chunks if c]
    logger.info(
        f"Semantic split: {len(clean_text)} chars into {len(result)} chunks "
        f"(max_size={chunk_size}, overlap={overlap})."
    )
    return result


def _sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            break_pt = text.rfind("\n", start + chunk_size // 2, end)
            if break_pt == -1:
                break_pt = text.rfind(". ", start + chunk_size // 2, end)
                if break_pt != -1:
                    break_pt += 1
            if break_pt == -1:
                break_pt = text.rfind(" ", start + chunk_size // 2, end)
            if break_pt != -1 and break_pt > start:
                end = break_pt

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = max(start + 1, end - overlap)
        if start >= len(text):
            break
    return chunks


def assign_cluster_id(vector: np.ndarray, validator: IVFBoundaryValidator) -> int:
    """Find closest IVF centroid (Voronoi cell) for a given chunk vector."""
    if validator.centroids is None or len(validator.centroids) == 0:
        return 0

    vec = np.asarray(vector, dtype=np.float32)
    if vec.ndim == 1:
        vec = vec.reshape(1, -1)

    vec_norm = vec / (np.linalg.norm(vec, axis=1, keepdims=True) + 1e-12)
    centroid_norms = validator.centroids / (np.linalg.norm(validator.centroids, axis=1, keepdims=True) + 1e-12)
    cosine_sim = centroid_norms @ vec_norm.T
    distances = (1.0 - cosine_sim).flatten()
    return int(np.argmin(distances))


class DocumentIngestionEngine:
    """Coordinates preprocessing, cryptographic sealing, MMR tree building, and Qdrant ingestion."""

    @classmethod
    def process_and_ingest(
        cls,
        text: str,
        doc_id: str,
        user_id: str,
        org_id: str,
        validator: IVFBoundaryValidator,
        qdrant_repo: Optional[QdrantRepository] = None,
        encryption_key: Optional[str] = None,
        chunk_size: int = 500,
        overlap: int = 50,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Complete end-to-end ingestion pipeline:
        1. Sliding-window chunking
        2. Vector embedding
        3. AES-256-GCM encryption & SHA-256 hashing
        4. IVF Voronoi cluster mapping
        5. Two-tier MMR tree construction & proof generation
        6. Qdrant storage
        """
        settings = get_settings()
        key = encryption_key or settings.DEFAULT_ENCRYPTION_KEY
        repo = qdrant_repo or QdrantRepository()

        # Step 1: Chunking
        raw_chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        if not raw_chunks:
            raise ValueError("Document content is empty or contains no readable text.")

        # Step 2: Dense Embeddings
        vectors = EmbeddingService.embed_batch(raw_chunks)

        # Step 3: Prepare Encrypted Chunk Records
        chunk_records: List[EncryptedChunkRecord] = []
        chunk_details: List[Dict[str, Any]] = []

        for idx, (chunk_str, vec) in enumerate(zip(raw_chunks, vectors)):
            c_hash = sha256_hex(chunk_str)
            encrypted = encrypt_aes_gcm(chunk_str, key)
            c_id = f"chk_{user_id}_{doc_id}_{idx}"
            cluster_id = assign_cluster_id(vec, validator)

            rec = EncryptedChunkRecord(
                point_id=str(uuid.uuid4()),
                chunk_id=c_id,
                doc_id=doc_id,
                org_id=org_id,
                user_id=user_id,
                chunk_index=idx,
                encrypted_content=encrypted.ciphertext_b64,
                nonce=encrypted.nonce_b64,
                chunk_hash=c_hash,
                metadata=ChunkMetadata(
                    filename=filename or f"{doc_id}.txt",
                    tokens=len(chunk_str.split()),
                    parent_doc_id=doc_id,
                    custom_attributes={
                        "acl": [user_id, f"org_admin_{org_id}"],
                        "section_type": classify_chunk_section(chunk_str),
                        "chunk_chars": len(chunk_str),
                    }
                ),
                vector=vec.tolist(),
                cluster_id=str(cluster_id),
            )
            chunk_records.append(rec)

        # Step 4: Construct Two-Tier Merkle Mountain Range (MMR)
        cloud_mmr = CloudMMR(org_id=org_id)
        cloud_mmr.build_from_chunks(chunk_records)

        # Step 5: Attach cryptographic inclusion proofs to each chunk
        for rec in chunk_records:
            proof = cloud_mmr.get_two_tier_proof(user_id=user_id, chunk_id=rec.chunk_id)
            if proof is not None:
                rec.proof_path = proof.model_dump()
                rec.leaf_hash = proof.leaf_hash
                rec.enterprise_root = cloud_mmr.cloud_root

            chunk_details.append({
                "chunk_id": rec.chunk_id,
                "chunk_index": rec.chunk_index,
                "cluster_id": rec.cluster_id,
                "preview": raw_chunks[rec.chunk_index][:80] + ("..." if len(raw_chunks[rec.chunk_index]) > 80 else ""),
                "content_hash": rec.chunk_hash[:16] + "...",
                "leaf_hash": rec.leaf_hash[:16] + "..." if rec.leaf_hash else None,
            })

        # Step 6: Ingest into Qdrant Vector Storage
        upserted_count = repo.upsert_chunks(org_id=org_id, chunks=chunk_records)

        logger.info(
            f"Successfully ingested document '{doc_id}' ({upserted_count} chunks) for user '{user_id}' in org '{org_id}'."
        )

        user_root = cloud_mmr.user_mmrs[user_id].user_root if user_id in cloud_mmr.user_mmrs else ""

        return {
            "status": "SUCCESS",
            "message": f"Document '{doc_id}' successfully chunked, encrypted, committed to Two-Tier MMR, and indexed into Qdrant.",
            "doc_id": doc_id,
            "filename": filename or f"{doc_id}.txt",
            "user_id": user_id,
            "org_id": org_id,
            "total_chunks": len(chunk_records),
            "user_mmr_root": user_root,
            "cloud_enterprise_root": cloud_mmr.cloud_root,
            "qdrant_points_upserted": upserted_count,
            "chunks": chunk_details,
        }
