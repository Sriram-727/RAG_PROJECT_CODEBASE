"""Cryptographic Merkle Mountain Range (MMR) inclusion proofs and verification logic."""
from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from crypto.hashing import (
    hash_chunk_leaf,
    hash_mmr_node,
    bag_peaks_hash,
    hash_user_root,
    hash_cloud_root,
)


class ProofStep(BaseModel):
    """A single step along the tree path towards a peak."""
    sibling_hash: str
    direction: Literal["left", "right"]  # Position of the sibling relative to the current node


class UserProof(BaseModel):
    """Proof of inclusion of a chunk leaf within a User MMR."""
    user_id: str
    leaf_index: int
    leaf_hash: str
    peak_index: int
    peak_hash: str
    sibling_path: List[ProofStep] = Field(default_factory=list)
    user_peaks: List[str] = Field(default_factory=list)
    user_bagged_peak: str
    user_root_hash: str
    chunk_count: int


class CloudProof(BaseModel):
    """Proof of inclusion of a User MMR Root within the Organization Cloud MMR."""
    org_id: str
    user_id: str
    user_index: int
    user_root_hash: str
    cloud_peak_index: int
    cloud_peak_hash: str
    cloud_sibling_path: List[ProofStep] = Field(default_factory=list)
    cloud_peaks: List[str] = Field(default_factory=list)
    cloud_bagged_peak: str
    cloud_root_hash: str
    user_count: int
    total_chunks: int


class TwoTierInclusionProof(BaseModel):
    """
    Complete end-to-end cryptographic proof connecting a document chunk
    to the organization's Cloud MMR Root.
    """
    chunk_id: str
    doc_id: str
    user_id: str
    org_id: str
    chunk_index: int
    chunk_hash: str
    leaf_hash: str
    user_proof: UserProof
    cloud_proof: CloudProof
    expected_cloud_root: str


def verify_user_proof(
    leaf_hash: str,
    proof: UserProof,
) -> bool:
    """Verify that a leaf hash belongs to the User MMR Root."""
    if proof.leaf_hash != leaf_hash:
        return False

    # 1. Ascend sibling path to compute the peak hash
    current = leaf_hash
    for step in proof.sibling_path:
        if step.direction == "left":
            current = hash_mmr_node(step.sibling_hash, current)
        else:
            current = hash_mmr_node(current, step.sibling_hash)

    if current != proof.peak_hash:
        return False

    # 2. Check peak existence in user_peaks
    if proof.peak_index >= len(proof.user_peaks):
        return False
    if proof.user_peaks[proof.peak_index] != current:
        return False

    # 3. Bag peaks and verify User MMR Root
    bagged = bag_peaks_hash(proof.user_peaks)
    if bagged != proof.user_bagged_peak:
        return False

    computed_user_root = hash_user_root(
        user_id=proof.user_id,
        user_bagged_peak=bagged,
        chunk_count=proof.chunk_count,
    )
    return computed_user_root == proof.user_root_hash


def verify_cloud_proof(
    user_root_hash: str,
    proof: CloudProof,
) -> bool:
    """Verify that a User MMR Root belongs to the Organization Cloud MMR Root."""
    if proof.user_root_hash != user_root_hash:
        return False

    # 1. Ascend sibling path to compute cloud peak hash
    current = user_root_hash
    for step in proof.cloud_sibling_path:
        if step.direction == "left":
            current = hash_mmr_node(step.sibling_hash, current)
        else:
            current = hash_mmr_node(current, step.sibling_hash)

    if current != proof.cloud_peak_hash:
        return False

    # 2. Check peak existence in cloud_peaks
    if proof.cloud_peak_index >= len(proof.cloud_peaks):
        return False
    if proof.cloud_peaks[proof.cloud_peak_index] != current:
        return False

    # 3. Bag peaks and verify Cloud MMR Root
    bagged = bag_peaks_hash(proof.cloud_peaks)
    if bagged != proof.cloud_bagged_peak:
        return False

    computed_cloud_root = hash_cloud_root(
        org_id=proof.org_id,
        cloud_bagged_peak=bagged,
        user_count=proof.user_count,
        total_chunks=proof.total_chunks,
    )
    return computed_cloud_root == proof.cloud_root_hash


def verify_two_tier_proof(
    proof: TwoTierInclusionProof,
    expected_cloud_root: Optional[str] = None,
) -> bool:
    """
    Verify complete two-tier cryptographic proof from chunk attributes to Cloud MMR Root.
    """
    target_cloud_root = expected_cloud_root or proof.expected_cloud_root

    # 1. Verify chunk leaf hash consistency
    recomputed_leaf_hash = hash_chunk_leaf(
        chunk_id=proof.chunk_id,
        doc_id=proof.doc_id,
        user_id=proof.user_id,
        org_id=proof.org_id,
        chunk_index=proof.chunk_index,
        content_hash=proof.chunk_hash,
    )
    if recomputed_leaf_hash != proof.leaf_hash:
        return False

    # 2. Verify User MMR tier
    if not verify_user_proof(recomputed_leaf_hash, proof.user_proof):
        return False

    # 3. Verify Cloud MMR tier
    if not verify_cloud_proof(proof.user_proof.user_root_hash, proof.cloud_proof):
        return False

    # 4. Compare with expected Cloud Root
    return proof.cloud_proof.cloud_root_hash == target_cloud_root
