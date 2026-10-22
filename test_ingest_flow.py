import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi.testclient import TestClient
from main import app

def test_full_ingestion_and_query_flow():
    client = TestClient(app)

    print("\n--- 1. Testing Health Check ---")
    res = client.get("/health")
    assert res.status_code == 200, res.text
    print("Health check OK:", res.json())

    print("\n--- 2. Ingesting Plaintext Document ---")
    doc_payload = {
        "text": (
            "ENTERPRISE CLUSTER SECURITY DIRECTIVE:\n"
            "All Kubernetes master nodes must enforce mutual TLS (mTLS) with certificates signed by the internal Vault CA. "
            "Pod network policies must default to deny-all ingress traffic. "
            "Administrative SSH access to bastion hosts is restricted to dedicated jumpboxes with hardware YubiKeys."
        ),
        "doc_id": "doc_k8s_security_01",
        "user_id": "usr_admin_01",
        "org_id": "org_enterprise_01",
        "chunk_size": 500,
        "overlap": 50,
        "filename": "k8s_security.txt"
    }

    ingest_res = client.post("/api/v1/ingest-text", json=doc_payload)
    assert ingest_res.status_code == 200, ingest_res.text
    ingest_data = ingest_res.json()
    print("Ingestion Success!")
    print(f"Total Chunks:           {ingest_data['total_chunks']}")
    print(f"User MMR Root:          {ingest_data['user_mmr_root']}")
    print(f"Cloud Enterprise Root:  {ingest_data['cloud_enterprise_root']}")
    print(f"Qdrant Points Upserted: {ingest_data['qdrant_points_upserted']}")

    print("\n--- 3. Querying Ingested Document through 5-Gate Engine ---")
    query_payload = {
        "query_text": "What certificates must Kubernetes master nodes enforce for mutual TLS?",
        "user_id": "usr_admin_01",
        "org_id": "org_enterprise_01",
        "simulate_missing_cluster": False,
        "simulate_integrity_failure": False
    }

    query_res = client.post("/api/v1/query-verify", json=query_payload)
    assert query_res.status_code == 200, query_res.text
    q_data = query_res.json()

    print(f"Verification Status:     {q_data['status']}")
    print(f"ACL Verified:            {q_data['acl_verified']}")
    print(f"Integrity Verified:      {q_data['integrity_verified']}")
    print(f"Boundary Verified:       {q_data['ivf_boundary_verification']['is_complete']}")
    print(f"Decryption Verified:     {q_data['decryption_verified']}")
    print(f"Hallucination Flagged:   {q_data['hallucination_report']['flagged']}")
    print(f"Grounding Score:         {q_data['hallucination_report'].get('grounding_score')}")
    print(f"Retrieved Chunks Count:  {q_data['retrieved_chunks_count']}")
    print(f"\nVerified Answer:\n  {q_data['answer']}")
    print("\n✅ End-to-End Ingest & Query Verification Passed!\n")

if __name__ == "__main__":
    test_full_ingestion_and_query_flow()
