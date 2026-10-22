"""
cloud_index.py

Stands in for the untrusted cloud's vector store (Stage 2 in your flow
doc). Holds masked vectors + which org/doc they belong to, clusters
them via K-Means, and serves searches restricted to whichever clusters
the client asked for.

Includes an `honest` flag on search() purely to DEMO detection: when
False, it simulates a cloud that silently skips a cluster it should
have searched (e.g. to save compute, or to hide a document) -- exactly
the behavior Dynamic Boundary Validation is meant to catch.
"""

from __future__ import annotations
from clustering import kmeans, ClusterRegistry


class CloudIndex:
    def __init__(self):
        self.points: list = []
        self.point_meta: list = []  # (owner_id, doc_id)
        self.registry: ClusterRegistry | None = None
        self.point_cluster: dict = {}

    def add_document(self, owner_id: str, doc_id: str, vector: list):
        self.points.append(vector)
        self.point_meta.append((owner_id, doc_id))

    def build_clusters(self, k: int = 3) -> ClusterRegistry:
        clusters = kmeans(self.points, k=k)
        self.registry = ClusterRegistry(clusters)
        self.point_cluster = {}
        for c in clusters:
            for idx in c.member_indices:
                self.point_cluster[idx] = c.cluster_id
        return self.registry

    def search(self, clusters_to_search: list, honest: bool = True):
        """Returns (results, clusters_actually_searched).
        If honest=False and more than one cluster was asked for, the
        LAST one is silently dropped -- simulating a cost-cutting or
        result-hiding cloud."""
        searched = list(clusters_to_search)
        if not honest and len(searched) > 1:
            searched = searched[:-1]  # drop the last required cluster

        results = [
            (owner_id, doc_id)
            for idx, (owner_id, doc_id) in enumerate(self.point_meta)
            if self.point_cluster.get(idx) in searched
        ]
        return results, searched
