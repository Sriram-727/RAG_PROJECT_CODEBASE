"""
clustering.py

Two things live here:

1. A minimal K-Means implementation that partitions masked embeddings
   into Voronoi cells (per your abstract: "organized into Voronoi-like
   partitions using K-Means clustering").

2. ClusterRegistry -- commits every cluster's centroid + spread into an
   MMR, exactly like documents are committed. This is what lets a
   client verify *which clusters exist and how big they are* without
   trusting the cloud's word for it -- the same Merkle authentication
   pattern used for documents, applied to the index structure itself.
   ("each cluster is authenticated using a Merkle Mountain Range" --
   from your abstract.)
"""

from __future__ import annotations
import random
from dataclasses import dataclass, field
from vector_utils import euclidean, mean_vector, vector_hash
from crypto_utils import sha256_hex
from mmr import BaggedMMR, Proof


@dataclass
class ClusterStats:
    cluster_id: int
    centroid: list
    radius: float  # max distance from centroid to any assigned point -- the cluster's "spread"
    member_indices: list = field(default_factory=list)


def kmeans(points: list, k: int, iterations: int = 20, seed: int = 42) -> list:
    """Bare-bones K-Means. Returns list[ClusterStats]. Deterministic given `seed`."""
    rng = random.Random(seed)
    k = min(k, len(points))
    centroids = [list(p) for p in rng.sample(points, k)]

    assignments = [0] * len(points)
    for _ in range(iterations):
        # assign step
        for i, p in enumerate(points):
            dists = [euclidean(p, c) for c in centroids]
            assignments[i] = dists.index(min(dists))
        # update step
        for ci in range(k):
            members = [points[i] for i in range(len(points)) if assignments[i] == ci]
            if members:
                centroids[ci] = mean_vector(members)

    clusters = []
    for ci in range(k):
        member_idx = [i for i in range(len(points)) if assignments[i] == ci]
        radius = max((euclidean(points[i], centroids[ci]) for i in member_idx), default=0.0)
        clusters.append(ClusterStats(cluster_id=ci, centroid=centroids[ci],
                                      radius=radius, member_indices=member_idx))
    return clusters


def cluster_leaf_hash(cluster: ClusterStats) -> str:
    payload = f"{cluster.cluster_id}|{vector_hash(cluster.centroid)}|{round(cluster.radius, 6)}"
    return sha256_hex(payload.encode("utf-8"))


class ClusterRegistry:
    """Merkle-committed set of clusters for one organization's index.
    Rebuilt whenever the cloud re-clusters (e.g. after enough new
    uploads); the resulting `root()` is cached client-side just like
    the Enterprise Root."""

    def __init__(self, clusters: list):
        self.clusters = {c.cluster_id: c for c in clusters}
        self._mmr = BaggedMMR()
        self._leaf_index = {}
        # deterministic order: by cluster_id
        for cid in sorted(self.clusters):
            idx = self._mmr.append(cluster_leaf_hash(self.clusters[cid]))
            self._leaf_index[cid] = idx

    def root(self) -> str:
        return self._mmr.root()

    def proof_for(self, cluster_id: int) -> Proof:
        return self._mmr.proof_for(self._leaf_index[cluster_id])

    def get(self, cluster_id: int) -> ClusterStats:
        return self.clusters[cluster_id]

    def all_ids(self) -> list:
        return sorted(self.clusters)
