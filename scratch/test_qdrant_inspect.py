from database.qdrant_repository import QdrantRepository
from core.embedding_service import EmbeddingService
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings
from verification.boundary_validation import IVFBoundaryValidator

validator = IVFBoundaryValidator(centroids_path='ivf_centroids.npy', radii_path='ivf_radii.npy')
validator.load_centroids()
repo = QdrantRepository()
settings = get_settings()

test_queries = [
    'tell the project listed in the resume',
    'what is multithreading in java give solution for this question',
    'what was the system design question',
    'give company information',
    'tell about the career objective'
]

for q in test_queries:
    vec = EmbeddingService.embed(q)
    req_clusters, _ = validator.required_clusters(vec)
    docs = repo.multi_probe_ivf_search('org_enterprise_01', vec.tolist(), req_clusters, top_k=4)
    print('='*70)
    print(f'QUERY: {q}')
    print(f'Total docs returned: {len(docs)}')
    for i, d in enumerate(docs[:6]):
        txt = decrypt_aes_gcm(d['encrypted_content'], d['nonce'], settings.DEFAULT_ENCRYPTION_KEY)
        c_idx = d.get('chunk_index')
        score = d.get('score', 0.0)
        print(f"  #{i+1} score={score:.4f} chunk_index={c_idx}: {repr(txt[:90])}")
