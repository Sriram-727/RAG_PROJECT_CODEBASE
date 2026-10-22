import logging
import numpy as np
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, PointStruct, Filter, 
    FieldCondition, MatchValue, PayloadSchemaType
)

logger = logging.getLogger("QdrantServiceManager")

class QdrantServiceManager:
    """Manages vector storage and IVF cluster-filtered query matching."""
    def __init__(self, in_memory: bool = True, host: str = "localhost", port: int = 6333):
        if in_memory:
            self.client = QdrantClient(":memory:")
            logger.info("Initialized local in-memory Qdrant engine.")
        else:
            self.client = QdrantClient(host=host, port=port)
            logger.info(f"Connected to Qdrant instance at {host}:{port}.")

    def initialize_collection(self, collection_name: str, vector_dim: int):
        collections = [c.name for c in self.client.get_collections().collections]
        if collection_name in collections:
            self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
        )

        try:
            self.client.create_payload_index(
                collection_name=collection_name,
                field_name="cluster_id",
                field_schema=PayloadSchemaType.KEYWORD
            )
            logger.info(f"Initialized collection '{collection_name}' with indexed key 'cluster_id'.")
        except Exception as e:
            logger.warning(f"Payload index creation note: {e}")

    def seed_ivf_dataset(self, collection_name: str, embeddings: np.ndarray, cluster_labels: np.ndarray, doc_texts: List[str] = None):
        points = [
            PointStruct(
                id=i + 1,
                vector=embeddings[i].tolist(),
                payload={
                    "chunk_id": f"chk_{i+1:04d}",
                    "cluster_id": str(cluster_labels[i]),
                    "text": doc_texts[i] if doc_texts else f"Enterprise security document chunk #{i+1} assigned to IVF cluster {cluster_labels[i]}.",
                    "org_id": "org_enterprise_01"
                }
            )
            for i in range(len(embeddings))
        ]
        self.client.upsert(collection_name=collection_name, points=points)
        logger.info(f"Upserted {len(points)} vector points into Qdrant collection '{collection_name}'.")

    def multi_probe_ivf_search(
        self, 
        collection_name: str, 
        query_vector: List[float], 
        required_cluster_ids: List[int],
        top_k: int = 2
    ) -> List[Dict[str, Any]]:
        retrieved_docs = []
        for cluster_id in required_cluster_ids:
            query_filter = Filter(
                must=[FieldCondition(key="cluster_id", match=MatchValue(value=str(cluster_id)))]
            )
            
            # Compatible with both modern (query_points) and older (search) qdrant-client versions
            if hasattr(self.client, "query_points"):
                response = self.client.query_points(
                    collection_name=collection_name,
                    query=query_vector,
                    query_filter=query_filter,
                    limit=top_k
                )
                hits = response.points
            elif hasattr(self.client, "search"):
                hits = self.client.search(
                    collection_name=collection_name,
                    query_vector=query_vector,
                    query_filter=query_filter,
                    limit=top_k
                )
            else:
                hits = []

            for hit in hits:
                payload = dict(hit.payload) if hit.payload else {}
                payload["score"] = hit.score
                retrieved_docs.append(payload)

        return retrieved_docs