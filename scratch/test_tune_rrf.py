import sys
sys.path.insert(0, ".")

from core.hybrid_reranker import HybridReranker
from database.qdrant_repository import QdrantRepository
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings
from core.embedding_service import EmbeddingService

repo = QdrantRepository()
settings = get_settings()
chunks = repo.extract_all_org_chunks('org_enterprise_01')
decrypted = [
    {
        'text': decrypt_aes_gcm(c.encrypted_content, c.nonce, settings.DEFAULT_ENCRYPTION_KEY),
        'chunk_index': c.chunk_index,
        'doc_id': c.doc_id,
        'cluster_id': c.cluster_id,
        'chunk_id': c.point_id
    }
    for c in chunks
]

queries = [
    'what is multithreading in java give solution for this question',
    'tell the project listed in the resume',
    'give company information',
    'what was the system design question'
]

for q in queries:
    q_vec = EmbeddingService.embed(q)
    dense_scores = [float(q_vec.dot(EmbeddingService.embed(d['text'][:250]))) for d in decrypted]
    bm25_scores = HybridReranker.compute_bm25_scores(q, decrypted)
    
    # RRF with k=20, dense_weight=0.6, bm25_weight=2.5
    fused = HybridReranker.rrf_fuse(decrypted, dense_scores, bm25_scores, k=20, dense_weight=0.6, bm25_weight=2.5)
    stitched = HybridReranker.stitch_adjacent_chunks(fused)
    
    print("=" * 70)
    print(f"QUERY: {q}")
    for rank, it in enumerate(stitched[:3]):
        c_idx = it.get('chunk_index')
        score = it.get('score')
        bm = it.get('bm25_score')
        dn = it.get('dense_score')
        txt = it['text'].replace('\n', ' ')[:90]
        print(f"  #{rank+1} ChunkIdx={c_idx} Score={score:.5f} (BM25={bm}, Dense={dn})")
        print(f"      {txt}...")
