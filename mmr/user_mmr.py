"""User-level Merkle Mountain Range (User MMR) builder and proof generator."""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from database.schemas import EncryptedChunkRecord
from crypto.hashing import hash_user_root
from mmr.core import PureMMR
from mmr.leaf import ChunkLeaf
from mmr.proof import ProofStep, UserProof


class UserMMR:
    """
    Constructs and manages the Merkle Mountain Range for a single user's document chunks.
    Produces the User MMR Root.
    """

    def __init__(self, user_id: str, org_id: str) -> None:
        self.user_id = user_id
        self.org_id = org_id
        self.pure_mmr = PureMMR()
        self.chunk_leaves: List[ChunkLeaf] = []
        self._chunk_id_map: Dict[str, int] = {}
        self._built: bool = False
        self._cached_user_root: Optional[str] = None

    @property
    def chunk_count(self) -> int:
        """Total number of chunks in this User MMR."""
        return len(self.chunk_leaves)

    @property
    def user_root(self) -> str:
        """Domain-separated User MMR Root."""
        if self._cached_user_root is None:
            self._recalculate_root()
        return self._cached_user_root or ""

    @property
    def bagged_peak(self) -> str:
        """Bagged peak hash across user's subtrees."""
        return self.pure_mmr.get_bagged_root()

    def _recalculate_root(self) -> None:
        """Recalculate User MMR Root."""
        bagged = self.pure_mmr.get_bagged_root()
        self._cached_user_root = hash_user_root(
            user_id=self.user_id,
            user_bagged_peak=bagged,
            chunk_count=len(self.chunk_leaves),
        )

    def build_from_chunks(self, chunks: List[EncryptedChunkRecord]) -> UserMMR:
        """
        Build User MMR from a list of chunks.
        Sorts chunks deterministically by (doc_id, chunk_index, chunk_id).
        """
        self.pure_mmr = PureMMR()
        self.chunk_leaves.clear()
        self._chunk_id_map.clear()
        self._cached_user_root = None

        # Sort deterministically
        sorted_chunks = sorted(chunks, key=lambda c: (c.doc_id, c.chunk_index, c.chunk_id))

        for idx, chunk in enumerate(sorted_chunks):
            leaf = ChunkLeaf.from_chunk_record(leaf_index=idx, chunk=chunk)
            self.chunk_leaves.append(leaf)
            self._chunk_id_map[chunk.chunk_id] = idx
            self.pure_mmr.append(leaf.leaf_hash, label=f"User-{self.user_id}-Chunk-{chunk.chunk_id}")

        self._built = True
        self._recalculate_root()
        return self

    def get_proof_for_chunk(self, chunk_id: str) -> Optional[UserProof]:
        """Generate UserProof for the specified chunk_id."""
        if chunk_id not in self._chunk_id_map:
            return None

        leaf_idx = self._chunk_id_map[chunk_id]
        leaf = self.chunk_leaves[leaf_idx]

        peak_info = self.pure_mmr.get_peak_containing_leaf(leaf_idx)
        if peak_info is None:
            return None

        peak_idx, peak = peak_info
        raw_path = self.pure_mmr.get_sibling_path_to_peak(leaf_idx)
        sibling_path = [
            ProofStep(sibling_hash=shash, direction=sdir)  # type: ignore[arg-type]
            for shash, sdir in raw_path
        ]

        return UserProof(
            user_id=self.user_id,
            leaf_index=leaf_idx,
            leaf_hash=leaf.leaf_hash,
            peak_index=peak_idx,
            peak_hash=peak.hash,
            sibling_path=sibling_path,
            user_peaks=self.pure_mmr.get_peak_hashes(),
            user_bagged_peak=self.pure_mmr.get_bagged_root(),
            user_root_hash=self.user_root,
            chunk_count=len(self.chunk_leaves),
        )

    def export_dict(self) -> Dict[str, Any]:
        """Serialize User MMR to dictionary."""
        return {
            "user_id": self.user_id,
            "org_id": self.org_id,
            "chunk_count": len(self.chunk_leaves),
            "user_root": self.user_root,
            "bagged_peak": self.bagged_peak,
            "peaks": self.pure_mmr.get_peak_hashes(),
            "leaves": [l.model_dump() for l in self.chunk_leaves],
            "tree_graph": self.pure_mmr.export_graph(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserMMR:
        """Reconstruct a UserMMR instance from its exported dictionary."""
        user_mmr = cls(user_id=data["user_id"], org_id=data["org_id"])
        leaves_data = data.get("leaves", [])
        for leaf_dict in leaves_data:
            leaf = ChunkLeaf(**leaf_dict)
            user_mmr.chunk_leaves.append(leaf)
            user_mmr._chunk_id_map[leaf.chunk_id] = leaf.leaf_index
            user_mmr.pure_mmr.append(
                leaf.leaf_hash,
                label=f"User-{user_mmr.user_id}-Chunk-{leaf.chunk_id}",
            )
        user_mmr._built = True
        user_mmr._recalculate_root()
        return user_mmr

