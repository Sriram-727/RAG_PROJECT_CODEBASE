import os
import sys
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from verification.boundary_validation import IVFBoundaryValidator
from core.document_ingestion import assign_cluster_id

def run_tests():
    print("=" * 60)
    print("TESTING IVF BOUNDARY VALIDATION MODULE")
    print("=" * 60)

    # Test 1: Instantiation & loading existing centroids/radii
    centroids_path = os.path.join(PROJECT_ROOT, "ivf_centroids.npy")
    radii_path = os.path.join(PROJECT_ROOT, "ivf_radii.npy")
    validator = IVFBoundaryValidator(
        margin_factor=1.5,
        centroids_path=centroids_path,
        radii_path=radii_path
    )
    loaded = validator.load_centroids()
    print(f"Test 1 - Load disk centroids: {loaded}")
    assert loaded is True, "Failed to load centroids from disk"
    assert validator.centroids is not None, "Centroids array is None"
    assert validator.radii is not None, "Radii array is None"
    assert len(validator.centroids) == 6, f"Expected 6 centroids, got {len(validator.centroids)}"
    assert len(validator.radii) == 6, f"Expected 6 radii, got {len(validator.radii)}"
    print("   [PASS] Centroids & radii loaded successfully.")

    # Test 2: required_clusters
    query_vec = np.random.randn(384).astype(np.float32)
    req_clusters, dists = validator.required_clusters(query_vec)
    print(f"Test 2 - Required clusters for random query: {req_clusters}")
    print(f"         Centroid distances: {[round(d, 4) for d in dists]}")
    assert isinstance(req_clusters, list) and len(req_clusters) > 0, "No clusters returned"
    assert all(isinstance(c, int) for c in req_clusters), "Clusters must be integers"
    assert len(dists) == len(validator.centroids), "Distance count mismatch"
    print("   [PASS] required_clusters works properly.")

    # Test 3: verify_query_boundary with 100% complete clusters
    eval_complete = validator.verify_query_boundary(query_vec, req_clusters)
    print(f"Test 3 - Complete probe: {eval_complete}")
    assert eval_complete["is_complete"] is True, "Expected is_complete == True"
    assert eval_complete["coverage_ratio"] == 1.0, "Expected coverage_ratio == 1.0"
    assert len(eval_complete["missing_clusters"]) == 0, "Expected no missing clusters"
    print("   [PASS] Complete boundary verification passed.")

    # Test 4: verify_query_boundary with incomplete clusters (omission attack)
    if len(req_clusters) > 1:
        incomplete_clusters = req_clusters[:-1]
        eval_incomplete = validator.verify_query_boundary(query_vec, incomplete_clusters)
        print(f"Test 4 - Incomplete probe: {eval_incomplete}")
        assert eval_incomplete["is_complete"] is False, "Expected is_complete == False"
        assert eval_incomplete["coverage_ratio"] < 1.0, "Expected coverage_ratio < 1.0"
        assert len(eval_incomplete["missing_clusters"]) > 0, "Expected missing clusters"
        print("   [PASS] Incomplete boundary correctly flagged as incomplete.")
    else:
        print("   [SKIP] Only 1 cluster required for this query, skipping multi-probe omission check.")

    # Test 5: Integration with chunk cluster assignment (assign_cluster_id)
    chunk_vec = np.random.randn(384).astype(np.float32)
    cid = assign_cluster_id(chunk_vec, validator)
    print(f"Test 5 - Chunk Voronoi cell assignment: cluster_id = {cid}")
    assert isinstance(cid, int) and 0 <= cid < len(validator.centroids)
    print("   [PASS] assign_cluster_id works properly.")

    # Test 6: In-memory build_ivf_index training
    test_centroids_path = os.path.join(PROJECT_ROOT, "scratch", "tmp_centroids.npy")
    test_radii_path = os.path.join(PROJECT_ROOT, "scratch", "tmp_radii.npy")
    temp_validator = IVFBoundaryValidator(
        margin_factor=1.5,
        centroids_path=test_centroids_path,
        radii_path=test_radii_path
    )
    dummy_data = np.random.randn(50, 384).astype(np.float32)
    labels = temp_validator.build_ivf_index(dummy_data, doc_ids=list(range(50)), n_lists=4)
    print(f"Test 6 - build_ivf_index labels count: {len(labels)}, unique clusters: {len(np.unique(labels))}")
    assert len(temp_validator.centroids) == 4
    assert len(temp_validator.radii) == 4
    print("   [PASS] build_ivf_index works properly.")

    # Cleanup temp files
    if os.path.exists(test_centroids_path): os.remove(test_centroids_path)
    if os.path.exists(test_radii_path): os.remove(test_radii_path)

    print("\n" + "=" * 60)
    print("ALL 6 TESTS PASSED! BOUNDARY VALIDATION MODULE IS FULLY FUNCTIONAL.")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
