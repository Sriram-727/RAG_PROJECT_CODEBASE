import sys
import os
sys.path.insert(0, ".")

from database.qdrant_repository import QdrantRepository
from core.embedding_service import EmbeddingService
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings
from verification.boundary_validation import IVFBoundaryValidator
from core.hybrid_reranker import HybridReranker
from core.llm_service import LLMService

validator = IVFBoundaryValidator(centroids_path='ivf_centroids.npy', radii_path='ivf_radii.npy')
validator.load_centroids()
repo = QdrantRepository()
settings = get_settings()

test_queries = [
    'what is multithreading in java give solution for this question',
    'tell the project listed in the resume',
    'give company information',
    'what was the system design question',
    'tell about the career objective'
]

# Fetch all chunks for org to test full candidate pool hybrid retrieval
all_encrypted = repo.extract_all_org_chunks('org_enterprise_01')
decrypted_docs = []
for c in all_encrypted:
    text = decrypt_aes_gcm(c.encrypted_content, c.nonce, settings.DEFAULT_ENCRYPTION_KEY)
    decrypted_docs.append({
        "text": text,
        "chunk_id": c.point_id,
        "doc_id": c.doc_id,
        "chunk_index": c.chunk_index,
        "cluster_id": c.cluster_id
    })

print(f"Decrypted {len(decrypted_docs)} chunks from Qdrant.\n")

for q in test_queries:
    print("=" * 80)
    print(f"QUERY: {q}")
    
    # 1. Compute Dense Scores
    q_vec = EmbeddingService.embed(q)
    # Cosine similarity for each doc
    dense_scores = []
    for d in decrypted_docs:
        # compute similarity using embedding
        d_vec = EmbeddingService.embed(d["text"][:200]) # quick preview embed
        dot = float(q_vec.dot(d_vec))
        dense_scores.append(dot)
        
    # 2. Compute BM25 Scores
    bm25_scores = HybridReranker.compute_bm25_scores(q, decrypted_docs)
    
    # 3. RRF Fusion
    fused = HybridReranker.rrf_fuse(decrypted_docs, dense_scores, bm25_scores, k=60, dense_weight=1.0, bm25_weight=1.5)
    
    # 4. Context Window Expansion (Stitching)
    stitched = HybridReranker.stitch_adjacent_chunks(fused)
    
    print(f"Top 3 Reranked Chunks:")
    for rank, item in enumerate(stitched[:3]):
        print(f"  #{rank+1} RRF={item.get('score'):.6f} BM25={item.get('bm25_score'):.2f} Dense={item.get('dense_score'):.2f} ChunkIdx={item.get('chunk_index')}")
        snippet = item['text'].replace('\n', ' ')[:140]
        print(f"      {snippet}...")
        
    # 5. LLM Synthesis
    ans = LLMService.generate_response(q, stitched[:4])
    print(f"\nSYNTHESIZED ANSWER:\n{ans}\n")
