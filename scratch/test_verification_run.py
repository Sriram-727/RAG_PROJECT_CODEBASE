import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from verification.boundary_validation import IVFBoundaryValidator
from core.document_ingestion import DocumentIngestionEngine
from database.qdrant_repository import QdrantRepository
from mmr.proof import TwoTierInclusionProof, verify_two_tier_proof

def test_verification():
    print("=== STARTING MMR & BOUNDARY VERIFICATION AUDIT ===")
    
    # 1. Setup validator
    validator = IVFBoundaryValidator(margin_factor=1.5, centroids_path='ivf_centroids.npy', radii_path='ivf_radii.npy')
    loaded = validator.load_centroids()
    print(f"1. IVF Centroids loaded: {loaded}")

    # 2. Ingest test doc
    repo = QdrantRepository()
    doc_text = "Enterprise Directive: Authentication keys and TLS certs must be rotated every 30 days."
    res = DocumentIngestionEngine.process_and_ingest(
        text=doc_text,
        doc_id="doc_verify_test_01",
        user_id="usr_admin_01",
        org_id="org_enterprise_01",
        validator=validator,
        qdrant_repo=repo,
        filename="verify_test.txt"
    )
    print(f"2. Ingestion Status: {res['status']} | Chunks: {res['total_chunks']}")
    print(f"   Cloud Enterprise Root: {res['cloud_enterprise_root']}")

    # 3. Retrieve from Qdrant
    retrieved = repo.search_entire_documents("org_enterprise_01")
    print(f"3. Retrieved {len(retrieved)} chunk(s) from Qdrant")

    # 4. Check MMR Proof on retrieved doc
    found_proof = False
    for doc in retrieved:
        proof_data = doc.get("proof_path")
        if proof_data:
            found_proof = True
            proof = TwoTierInclusionProof.model_validate(proof_data)
            is_valid = verify_two_tier_proof(proof)
            print(f"4a. Untampered MMR Proof validation: {is_valid} (Expected: True)")
            
            # Test tampering detection: modify chunk_hash
            proof_tampered = TwoTierInclusionProof.model_validate(proof_data)
            proof_tampered.chunk_hash = "tampered_hash_0123456789abcdef0123456789abcdef"
            is_tampered_valid = verify_two_tier_proof(proof_tampered)
            print(f"4b. Tampered chunk_hash MMR Proof validation: {is_tampered_valid} (Expected: False)")

            # Test tampering expected cloud root
            is_wrong_root_valid = verify_two_tier_proof(proof, expected_cloud_root="0000000000000000000000000000000000000000000000000000000000000000")
            print(f"4c. Forged Cloud Root validation: {is_wrong_root_valid} (Expected: False)")
            break

    print(f"   Proof found in retrieved doc payload: {found_proof}")

    # 5. Check Boundary Validation
    dummy_q = np.random.randn(384).astype(np.float32)
    req_clusters, dists = validator.required_clusters(dummy_q)
    print(f"5a. Query required Voronoi clusters: {req_clusters}")

    eval_full = validator.verify_query_boundary(dummy_q, req_clusters)
    print(f"5b. Full cluster probe: is_complete = {eval_full['is_complete']} | coverage = {eval_full['coverage_ratio']}")

    if len(req_clusters) > 1:
        eval_missing = validator.verify_query_boundary(dummy_q, req_clusters[:-1])
        print(f"5c. Missing cluster probe: is_complete = {eval_missing['is_complete']} | coverage = {eval_missing['coverage_ratio']} | missing = {eval_missing['missing_clusters']}")

    print("=== AUDIT COMPLETE ===")

if __name__ == "__main__":
    test_verification()
