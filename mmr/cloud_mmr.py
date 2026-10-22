"""Cloud-Side Merkle Mountain Range (Cloud MMR) over entire organization dataset."""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from database.schemas import EncryptedChunkRecord, OrganizationDataset
from crypto.hashing import hash_cloud_root
from mmr.core import PureMMR
from mmr.leaf import UserRootLeaf
from mmr.proof import (
    CloudProof,
    ProofStep,
    TwoTierInclusionProof,
    verify_two_tier_proof,
)
from mmr.user_mmr import UserMMR

logger = logging.getLogger(__name__)


class CloudMMR:
    """
    Constructs the Organization / Cloud-Side Merkle Mountain Range over all members' User-MMR Roots.
    Calculates the canonical Cloud MMR Root and generates two-tier cryptographic proofs.
    """

    def __init__(self, org_id: str) -> None:
        self.org_id = org_id
        self.user_mmrs: Dict[str, UserMMR] = {}
        self.user_leaves: List[UserRootLeaf] = []
        self._user_id_map: Dict[str, int] = {}
        self.pure_cloud_mmr = PureMMR()
        self._total_chunks: int = 0
        self._cached_cloud_root: Optional[str] = None

    @property
    def total_chunks(self) -> int:
        """Total chunks across all users in the organization."""
        return self._total_chunks

    @property
    def user_count(self) -> int:
        """Total users in the organization."""
        return len(self.user_leaves)

    @property
    def cloud_root(self) -> str:
        """Domain-separated Cloud / Organization MMR Root."""
        if self._cached_cloud_root is None:
            self._recalculate_cloud_root()
        return self._cached_cloud_root or ""

    @property
    def bagged_peak(self) -> str:
        """Bagged peak hash of the Cloud MMR."""
        return self.pure_cloud_mmr.get_bagged_root()

    def _recalculate_cloud_root(self) -> None:
        """Recalculate Cloud MMR Root."""
        bagged = self.pure_cloud_mmr.get_bagged_root()
        self._cached_cloud_root = hash_cloud_root(
            org_id=self.org_id,
            cloud_bagged_peak=bagged,
            user_count=len(self.user_leaves),
            total_chunks=self._total_chunks,
        )

    def build_from_dataset(self, dataset: OrganizationDataset) -> CloudMMR:
        """
        Build Cloud MMR from an OrganizationDataset.
        Deterministically processes users in alphabetical order of user_id.
        """
        self.user_mmrs.clear()
        self.user_leaves.clear()
        self._user_id_map.clear()
        self.pure_cloud_mmr = PureMMR()
        self._total_chunks = 0
        self._cached_cloud_root = None

        # Sort user IDs deterministically
        sorted_user_ids = sorted(dataset.users.keys())

        # Step 1: Build each User MMR
        for user_id in sorted_user_ids:
            chunks = dataset.users[user_id]
            user_mmr = UserMMR(user_id=user_id, org_id=self.org_id)
            user_mmr.build_from_chunks(chunks)
            self.user_mmrs[user_id] = user_mmr
            self._total_chunks += len(chunks)

        # Step 2: Build Cloud MMR using User MMR Roots as leaves
        for idx, user_id in enumerate(sorted_user_ids):
            user_mmr = self.user_mmrs[user_id]
            user_leaf = UserRootLeaf(
                user_index=idx,
                user_id=user_id,
                user_root_hash=user_mmr.user_root,
                chunk_count=user_mmr.chunk_count,
                bagged_peak=user_mmr.bagged_peak,
            )
            self.user_leaves.append(user_leaf)
            self._user_id_map[user_id] = idx

            # Append user root to Cloud MMR
            self.pure_cloud_mmr.append(
                user_leaf.user_root_hash,
                label=f"Cloud-UserRoot-{user_id}",
            )

        self._recalculate_cloud_root()
        logger.info(
            f"Built Cloud MMR for org '{self.org_id}': users={len(sorted_user_ids)}, "
            f"chunks={self._total_chunks}, root={self.cloud_root}"
        )
        return self

    def build_from_chunks(self, chunks: List[EncryptedChunkRecord]) -> CloudMMR:
        """Convenience method to construct Cloud MMR directly from a list of chunk records."""
        users_map: Dict[str, List[EncryptedChunkRecord]] = {}
        for chunk in chunks:
            if chunk.user_id not in users_map:
                users_map[chunk.user_id] = []
            users_map[chunk.user_id].append(chunk)

        dataset = OrganizationDataset(
            org_id=self.org_id,
            users=users_map,
            total_chunks=len(chunks),
            total_users=len(users_map),
        )
        return self.build_from_dataset(dataset)

    def get_user_proof(self, user_id: str) -> Optional[CloudProof]:
        """Generate CloudProof for a specific User MMR Root."""
        if user_id not in self._user_id_map:
            return None

        user_idx = self._user_id_map[user_id]
        user_leaf = self.user_leaves[user_idx]

        peak_info = self.pure_cloud_mmr.get_peak_containing_leaf(user_idx)
        if peak_info is None:
            return None

        cloud_peak_idx, cloud_peak = peak_info
        raw_path = self.pure_cloud_mmr.get_sibling_path_to_peak(user_idx)
        sibling_path = [
            ProofStep(sibling_hash=shash, direction=sdir)  # type: ignore[arg-type]
            for shash, sdir in raw_path
        ]

        return CloudProof(
            org_id=self.org_id,
            user_id=user_id,
            user_index=user_idx,
            user_root_hash=user_leaf.user_root_hash,
            cloud_peak_index=cloud_peak_idx,
            cloud_peak_hash=cloud_peak.hash,
            cloud_sibling_path=sibling_path,
            cloud_peaks=self.pure_cloud_mmr.get_peak_hashes(),
            cloud_bagged_peak=self.pure_cloud_mmr.get_bagged_root(),
            cloud_root_hash=self.cloud_root,
            user_count=len(self.user_leaves),
            total_chunks=self._total_chunks,
        )

    def get_two_tier_proof(self, user_id: str, chunk_id: str) -> Optional[TwoTierInclusionProof]:
        """
        Generate end-to-end TwoTierInclusionProof for a given chunk.
        Combines User-level MMR proof and Cloud-level MMR proof.
        """
        if user_id not in self.user_mmrs:
            return None

        user_mmr = self.user_mmrs[user_id]
        user_proof = user_mmr.get_proof_for_chunk(chunk_id)
        if user_proof is None:
            return None

        cloud_proof = self.get_user_proof(user_id)
        if cloud_proof is None:
            return None

        chunk_leaf = user_mmr.chunk_leaves[user_proof.leaf_index]

        return TwoTierInclusionProof(
            chunk_id=chunk_leaf.chunk_id,
            doc_id=chunk_leaf.doc_id,
            user_id=chunk_leaf.user_id,
            org_id=chunk_leaf.org_id,
            chunk_index=chunk_leaf.chunk_index,
            chunk_hash=chunk_leaf.content_hash,
            leaf_hash=chunk_leaf.leaf_hash,
            user_proof=user_proof,
            cloud_proof=cloud_proof,
            expected_cloud_root=self.cloud_root,
        )

    def export_full_hierarchy(self) -> Dict[str, Any]:
        """Export the full organizational MMR hierarchy as a structured dictionary."""
        return {
            "org_id": self.org_id,
            "cloud_root": self.cloud_root,
            "bagged_peak": self.bagged_peak,
            "user_count": len(self.user_leaves),
            "total_chunks": self._total_chunks,
            "cloud_peaks": self.pure_cloud_mmr.get_peak_hashes(),
            "cloud_tree_graph": self.pure_cloud_mmr.export_graph(),
            "user_roots": {
                u_id: u_mmr.export_dict() for u_id, u_mmr in self.user_mmrs.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> CloudMMR:
        """Reconstruct a CloudMMR instance from its exported hierarchy dictionary."""
        org_id = data["org_id"]
        cloud_mmr = cls(org_id=org_id)
        user_roots_dict = data.get("user_roots", {})
        sorted_user_ids = sorted(user_roots_dict.keys())

        for idx, u_id in enumerate(sorted_user_ids):
            u_data = user_roots_dict[u_id]
            u_mmr = UserMMR.from_dict(u_data)
            cloud_mmr.user_mmrs[u_id] = u_mmr
            cloud_mmr._total_chunks += u_mmr.chunk_count

            user_leaf = UserRootLeaf(
                user_index=idx,
                user_id=u_id,
                user_root_hash=u_mmr.user_root,
                chunk_count=u_mmr.chunk_count,
                bagged_peak=u_mmr.bagged_peak,
            )
            cloud_mmr.user_leaves.append(user_leaf)
            cloud_mmr._user_id_map[u_id] = idx
            cloud_mmr.pure_cloud_mmr.append(
                user_leaf.user_root_hash,
                label=f"Cloud-UserRoot-{u_id}",
            )

        cloud_mmr._recalculate_cloud_root()
        return cloud_mmr

