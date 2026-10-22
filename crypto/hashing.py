"""Deterministic and domain-separated SHA-256 cryptographic hashing."""
from __future__ import annotations

import hashlib
from typing import Sequence

# Domain Separation Prefixes (NIST / RFC standard practice to prevent length extension & collision)
DOMAIN_LEAF = b"\x00"
DOMAIN_INTERNAL_NODE = b"\x01"
DOMAIN_BAG_PEAKS = b"\x02"
DOMAIN_USER_ROOT = b"\x03"
DOMAIN_CLOUD_ROOT = b"\x04"


def sha256_hex(data: bytes | str) -> str:
    """Compute standard SHA-256 hex digest."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_chunk_leaf(
    chunk_id: str,
    doc_id: str,
    user_id: str,
    org_id: str,
    chunk_index: int,
    content_hash: str,
) -> str:
    """
    Compute domain-separated leaf hash for a document chunk.
    Leaf = SHA256(0x00 || org_id || ':' || user_id || ':' || doc_id || ':' || chunk_id || ':' || chunk_index || ':' || content_hash)
    """
    hasher = hashlib.sha256()
    hasher.update(DOMAIN_LEAF)
    canonical_repr = (
        f"{org_id}:{user_id}:{doc_id}:{chunk_id}:{chunk_index}:{content_hash}"
    )
    hasher.update(canonical_repr.encode("utf-8"))
    return hasher.hexdigest()


def hash_mmr_node(left_hash: str, right_hash: str) -> str:
    """
    Compute domain-separated internal MMR node hash from left and right children.
    Node = SHA256(0x01 || left_hash_bytes || right_hash_bytes)
    """
    hasher = hashlib.sha256()
    hasher.update(DOMAIN_INTERNAL_NODE)
    hasher.update(bytes.fromhex(left_hash))
    hasher.update(bytes.fromhex(right_hash))
    return hasher.hexdigest()


def bag_peaks_hash(peak_hashes: Sequence[str]) -> str:
    """
    Bag a list of MMR peak hashes from right to left (standard MMR peak bagging).
    If empty, returns empty SHA256 hash.
    If single peak, returns that peak.
    If multiple peaks [P0, P1, P2]:
      acc = P2
      acc = hash_mmr_node(P1, acc)
      acc = hash_mmr_node(P0, acc)
    """
    if not peak_hashes:
        return sha256_hex(b"")
    if len(peak_hashes) == 1:
        return peak_hashes[0]

    # Bag from right to left
    accumulated = peak_hashes[-1]
    for peak in reversed(peak_hashes[:-1]):
        accumulated = hash_mmr_node(peak, accumulated)
    return accumulated


def hash_user_root(user_id: str, user_bagged_peak: str, chunk_count: int) -> str:
    """
    Compute domain-separated User MMR Root.
    UserRoot = SHA256(0x03 || user_id || ':' || chunk_count || ':' || user_bagged_peak)
    """
    hasher = hashlib.sha256()
    hasher.update(DOMAIN_USER_ROOT)
    canonical = f"{user_id}:{chunk_count}:{user_bagged_peak}"
    hasher.update(canonical.encode("utf-8"))
    return hasher.hexdigest()


def hash_cloud_root(org_id: str, cloud_bagged_peak: str, user_count: int, total_chunks: int) -> str:
    """
    Compute domain-separated Cloud / Organization MMR Root.
    CloudRoot = SHA256(0x04 || org_id || ':' || user_count || ':' || total_chunks || ':' || cloud_bagged_peak)
    """
    hasher = hashlib.sha256()
    hasher.update(DOMAIN_CLOUD_ROOT)
    canonical = f"{org_id}:{user_count}:{total_chunks}:{cloud_bagged_peak}"
    hasher.update(canonical.encode("utf-8"))
    return hasher.hexdigest()
