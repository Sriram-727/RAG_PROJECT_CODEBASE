"""Core Merkle Mountain Range (MMR) data structure and peak management algorithms."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field
from crypto.hashing import hash_mmr_node, bag_peaks_hash


class MMRNode(BaseModel):
    """Represents a node in the Merkle Mountain Range."""
    node_id: int
    hash: str
    height: int
    is_leaf: bool
    leaf_index: Optional[int] = None
    left_child_id: Optional[int] = None
    right_child_id: Optional[int] = None
    parent_id: Optional[int] = None
    label: Optional[str] = None


class Peak(BaseModel):
    """Represents an MMR peak (root of a perfect binary subtree)."""
    node_id: int
    height: int
    hash: str
    leaf_count: int  # 2^height
    leaf_offset: int  # starting leaf index


class PureMMR:
    """
    Pure append-only Merkle Mountain Range implementation.
    Manages leaves, internal nodes, peaks, and sibling proof pathways.
    """

    def __init__(self) -> None:
        self.nodes: Dict[int, MMRNode] = {}
        self.leaves: List[str] = []
        self._peaks: List[Peak] = []
        self._next_node_id: int = 1

    @property
    def leaf_count(self) -> int:
        """Total number of leaves in the MMR."""
        return len(self.leaves)

    @property
    def peak_count(self) -> int:
        """Number of active peaks."""
        return len(self._peaks)

    def get_peak_hashes(self) -> List[str]:
        """Return list of peak hashes from left (highest) to right (lowest)."""
        return [p.hash for p in self._peaks]

    def get_bagged_root(self) -> str:
        """Compute the bagged peak root."""
        return bag_peaks_hash(self.get_peak_hashes())

    def append(self, leaf_hash: str, label: Optional[str] = None) -> int:
        """
        Append a new leaf hash to the MMR.
        Merges peaks of equal height into higher-level internal nodes.
        Returns the leaf index (0-based).
        """
        leaf_index = len(self.leaves)
        self.leaves.append(leaf_hash)

        # Create leaf node (height 0)
        leaf_node_id = self._next_node_id
        self._next_node_id += 1

        leaf_node = MMRNode(
            node_id=leaf_node_id,
            hash=leaf_hash,
            height=0,
            is_leaf=True,
            leaf_index=leaf_index,
            label=label or f"Leaf-{leaf_index}",
        )
        self.nodes[leaf_node_id] = leaf_node

        current_peak = Peak(
            node_id=leaf_node_id,
            height=0,
            hash=leaf_hash,
            leaf_count=1,
            leaf_offset=leaf_index,
        )

        # Merge peaks with identical heights
        while self._peaks and self._peaks[-1].height == current_peak.height:
            prev_peak = self._peaks.pop()

            parent_node_id = self._next_node_id
            self._next_node_id += 1

            parent_hash = hash_mmr_node(prev_peak.hash, current_peak.hash)
            parent_height = current_peak.height + 1

            parent_node = MMRNode(
                node_id=parent_node_id,
                hash=parent_hash,
                height=parent_height,
                is_leaf=False,
                left_child_id=prev_peak.node_id,
                right_child_id=current_peak.node_id,
                label=f"Node-{parent_node_id}-H{parent_height}",
            )
            self.nodes[parent_node_id] = parent_node

            # Set parent references
            self.nodes[prev_peak.node_id].parent_id = parent_node_id
            self.nodes[current_peak.node_id].parent_id = parent_node_id

            current_peak = Peak(
                node_id=parent_node_id,
                height=parent_height,
                hash=parent_hash,
                leaf_count=prev_peak.leaf_count + current_peak.leaf_count,
                leaf_offset=prev_peak.leaf_offset,
            )

        self._peaks.append(current_peak)
        return leaf_index

    def get_leaf_node_id(self, leaf_index: int) -> Optional[int]:
        """Find the node_id for a given leaf index."""
        for nid, node in self.nodes.items():
            if node.is_leaf and node.leaf_index == leaf_index:
                return nid
        return None

    def get_peak_containing_leaf(self, leaf_index: int) -> Optional[Tuple[int, Peak]]:
        """Return (peak_index, Peak) containing the specified leaf index."""
        for idx, peak in enumerate(self._peaks):
            if peak.leaf_offset <= leaf_index < peak.leaf_offset + peak.leaf_count:
                return idx, peak
        return None

    def get_sibling_path_to_peak(self, leaf_index: int) -> List[Tuple[str, str]]:
        """
        Compute the list of (sibling_hash, direction) steps from the leaf to its peak.
        direction: 'left' if sibling is on the left, 'right' if sibling is on the right.
        """
        node_id = self.get_leaf_node_id(leaf_index)
        if node_id is None:
            return []

        path: List[Tuple[str, str]] = []
        current_node = self.nodes[node_id]

        while current_node.parent_id is not None:
            parent = self.nodes[current_node.parent_id]
            if parent.left_child_id == current_node.node_id:
                # Sibling is right child
                sibling = self.nodes[parent.right_child_id]
                path.append((sibling.hash, "right"))
            else:
                # Sibling is left child
                sibling = self.nodes[parent.left_child_id]
                path.append((sibling.hash, "left"))
            current_node = parent

        return path

    def export_graph(self) -> Dict[str, Any]:
        """Export full graph structure for serialization and visualization."""
        return {
            "total_leaves": len(self.leaves),
            "peaks": [p.model_dump() for p in self._peaks],
            "nodes": {str(k): v.model_dump() for k, v in self.nodes.items()},
            "bagged_root": self.get_bagged_root(),
        }
