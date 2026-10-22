"""Utility script to inspect actual points and payloads stored in Qdrant."""
import sys
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
from database.qdrant_repository import QdrantRepository
from database.qdrant_client import QdrantClientManager
from core.document_ingestion import DocumentIngestionEngine
from verification.boundary_validation import IVFBoundaryValidator
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings

validator = IVFBoundaryValidator(centroids_path="ivf_centroids.npy", radii_path="ivf_radii.npy")
validator.load_centroids()
repo = QdrantRepository()

# Sample document text
doc_text = (
    "ENTERPRISE CLOUD SECURITY DIRECTIVE 2026: All employee accounts accessing production Kubernetes clusters "
    "and AWS infrastructure must enforce hardware security keys (FIDO2/WebAuthn). "
    "All VPN security credentials and TLS private certificates must be rotated every 30 days automatically."
)

res = DocumentIngestionEngine.process_and_ingest(
    text=doc_text,
    doc_id="doc_security_demo",
    user_id="usr_admin_01",
    org_id="org_enterprise_01",
    validator=validator,
    qdrant_repo=repo,
    chunk_size=500,
    overlap=50,
    filename="cloud_security_policy.pdf"
)

client = QdrantClientManager.get_client()
collection_name = repo.get_collection_name("org_enterprise_01")
scroll_result, _ = client.scroll(
    collection_name=collection_name,
    limit=2,
    with_payload=True,
    with_vectors=True
)

settings = get_settings()

for idx, point in enumerate(scroll_result):
    payload = point.payload
    ciphertext = payload.get("encrypted_content", "")
    nonce = payload.get("nonce", "")
    decrypted = decrypt_aes_gcm(ciphertext, nonce, settings.DEFAULT_ENCRYPTION_KEY)
    
    print("\n" + "=" * 80)
    print(f" 📍 RAW POINT IN QDRANT #{idx+1} [Point ID: {point.id}]")
    print("=" * 80)
    print(f"Collection:         {collection_name}")
    print(f"Vector Dimensions:  {len(point.vector)} (First 5 values: {point.vector[:5]})")
    print(f"Chunk ID:           {payload.get('chunk_id')}")
    print(f"Document ID:        {payload.get('doc_id')}")
    print(f"Owner User ID:      {payload.get('user_id')}")
    print(f"Tenant Org ID:      {payload.get('org_id')}")
    print(f"IVF Cluster ID:     {payload.get('cluster_id')}")
    print(f"SHA-256 Hash:       {payload.get('chunk_hash')}")
    print(f"MMR Leaf Hash:      {payload.get('leaf_hash')}")
    print(f"Enterprise Root:    {payload.get('enterprise_root')}")
    print(f"Metadata:           {payload.get('metadata')}")
    print("-" * 80)
    print("🔐 Encrypted Content (Stored in Qdrant - Zero Plaintext Exposure):")
    print(f"   Nonce (IV):       {nonce}")
    print(f"   Ciphertext (B64): {ciphertext}")
    print("-" * 80)
    print("🔓 Decrypted Content (Decrypted locally by client via Gate 4):")
    print(f"   \"{decrypted}\"")
    print("=" * 80)
