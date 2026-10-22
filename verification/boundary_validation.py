import os
import logging
import numpy as np
from typing import List, Dict, Any, Tuple
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("IVFBoundaryValidator")

class IVFBoundaryValidator:
    """
    Inverted File Index (IVF) Dynamic Boundary Validator.
    Implements Voronoi coarse-quantizer partitioning and multi-probe margin expansion.
    """
    def __init__(self, margin_factor: float = 1.5, centroids_path: str = "ivf_centroids.npy", radii_path: str = "ivf_radii.npy"):
        self.margin_factor = margin_factor
        self.centroids_path = centroids_path
        self.radii_path = radii_path
        self.centroids: np.ndarray = None
        self.radii: np.ndarray = None
        self.inverted_lists: Dict[int, List[int]] = {}
        self.load_centroids()

    def build_ivf_index(self, embeddings: np.ndarray, doc_ids: List[int], n_lists: int = 6) -> np.ndarray:
        logger.info(f"Training IVF coarse quantizer with {n_lists} lists over {len(embeddings)} vectors...")
        kmeans = KMeans(n_clusters=n_lists, random_state=42, n_init=10, max_iter=300)
        labels = kmeans.fit_predict(embeddings)
        
        self.centroids = kmeans.cluster_centers_
        np.save(self.centroids_path, self.centroids)

        self.inverted_lists = {cid: [] for cid in range(n_lists)}
        for idx, cluster_id in enumerate(labels):
            self.inverted_lists[int(cluster_id)].append(doc_ids[idx])

        # Compute and save radius (max cosine distance from centroid) for each cluster
        radii = np.zeros(n_lists, dtype=np.float32)
        embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12)
        centroid_norms = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)
        for cid in range(n_lists):
            cluster_points = embeddings_norm[labels == cid]
            if len(cluster_points) > 0:
                cosine_sim = cluster_points @ centroid_norms[cid]
                dists = 1.0 - cosine_sim
                radii[cid] = float(np.max(dists))
            else:
                radii[cid] = 0.0
        self.radii = radii
        np.save(self.radii_path, self.radii)

        logger.info(f"Persisted IVF coarse quantizer matrix to '{self.centroids_path}' and radii to '{self.radii_path}'.")
        return labels

    def load_centroids(self) -> bool:
        if os.path.exists(self.centroids_path) and os.path.exists(self.radii_path):
            try:
                self.centroids = np.load(self.centroids_path)
                self.radii = np.load(self.radii_path)
                logger.info(f"Loaded IVF centroids and radii from disk.")
                return True
            except Exception as e:
                logger.error(f"Failed loading centroids/radii: {e}")
        return False

    def required_clusters(self, query_vector: np.ndarray) -> Tuple[List[int], List[float]]:
        if self.centroids is None:
            raise ValueError("IVF centroids are uninitialized. Build or load index first.")

        query_vec = np.asarray(query_vector, dtype=np.float32)
        if query_vec.ndim == 1:
            query_vec = query_vec.reshape(1, -1)

        # Cosine distance, consistent with the Qdrant collection's Distance.COSINE config.
        query_norm = query_vec / (np.linalg.norm(query_vec, axis=1, keepdims=True) + 1e-12)
        centroid_norms = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-12)
        cosine_sim = centroid_norms @ query_norm.T
        distances = (1.0 - cosine_sim).flatten()
        min_dist = float(np.min(distances))

        if min_dist == 0.0:
            closest_idx = int(np.argmin(distances))
            return [closest_idx], distances.tolist()

        closest_idx = int(np.argmin(distances))
        nearest_radius = float(self.radii[closest_idx])
        margin = self.margin_factor * nearest_radius

        required_list_ids = [closest_idx]
        for idx, dist in enumerate(distances):
            if idx != closest_idx and (dist - min_dist) < margin:
                required_list_ids.append(int(idx))

        return required_list_ids, distances.tolist()

    def verify_query_boundary(self, query_vector: np.ndarray, retrieved_cluster_ids: List[int]) -> Dict[str, Any]:
        required_ids, distances = self.required_clusters(query_vector)
        retrieved_set = set(retrieved_cluster_ids)
        required_set = set(required_ids)
        
        is_complete = required_set.issubset(retrieved_set)
        missing_clusters = list(required_set - retrieved_set)
        coverage = len(retrieved_set.intersection(required_set)) / float(len(required_set)) if required_set else 1.0

        return {
            "is_complete": is_complete,
            "coverage_ratio": round(coverage, 4),
            "required_clusters": required_ids,
            "retrieved_clusters": list(retrieved_set),
            "missing_clusters": missing_clusters,
            "centroid_distances": [round(d, 4) for d in distances]
        }