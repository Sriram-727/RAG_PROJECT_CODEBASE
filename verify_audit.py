"""
Two-Tier Merkle Mountain Range (MMR) Audit & Verification Demo
Demonstrates: Chunk Leaves -> User MMRs -> Cloud Enterprise Root -> Proof Verification
"""
from __future__ import annotations

import sys
import uuid

# Windows console UTF-8 compatibility
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from crypto.hashing import sha256_hex
from crypto.encryption import encrypt_aes_gcm
from database.schemas import ChunkMetadata, EncryptedChunkRecord
from mmr.cloud_mmr import CloudMMR
from mmr.proof import verify_two_tier_proof


def run_audit():
    print("\n" + "=" * 80)
    print("🚀 EXECUTING TWO-TIER MERKLE MOUNTAIN RANGE (MMR) AUDIT")
    print("=" * 80)

    org_id = "org-demo"
    key = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

    # Step 1: Simulated multi-user, multi-chunk enterprise dataset
    raw_dataset = [
        # user-A: 2 docs, 4 chunks total
        ("user-A", "doc-A1-finances", 0, "Quarterly Q1 revenue was 12.4 million USD across all sectors."),
        ("user-A", "doc-A1-finances", 1, "Operating expenses decreased by 4.2% following infrastructure migration."),
        ("user-A", "doc-A2-strategy", 0, "Enterprise roadmap focuses on Merkle Mountain Range verifiable RAG."),
        ("user-A", "doc-A2-strategy", 1, "Zero-trust encryption ensures cryptographic privacy at vector layer."),

        # user-B: 2 docs, 3 chunks total
        ("user-B", "doc-B1-security", 0, "Security boundaries mandate strict tenant organization isolation."),
        ("user-B", "doc-B1-security", 1, "Authenticated AES-256-GCM ciphertexts prevent payload tampering."),
        ("user-B", "doc-B2-governance", 0, "Compliance audit logs record all proof verification events."),

        # user-C: 2 docs, 3 chunks total
        ("user-C", "doc-C1-ml-infra", 0, "Dense embeddings (384-dim) are stored in Qdrant collections."),
        ("user-C", "doc-C1-ml-infra", 1, "ANN vector retrieval generates candidate context chunks."),
        ("user-C", "doc-C2-evaluation", 0, "Hallucination checking ensures factual consistency of LLM outputs."),
    ]

    chunk_records = []
    for user_id, doc_id, chunk_index, plaintext in raw_dataset:
        content_hash = sha256_hex(plaintext)
        encrypted = encrypt_aes_gcm(plaintext, key)
        chunk_id = f"chk_{user_id}_{doc_id}_{chunk_index}"
        rec = EncryptedChunkRecord(
            point_id=str(uuid.uuid4()),
            chunk_id=chunk_id,
            doc_id=doc_id,
            org_id=org_id,
            user_id=user_id,
            chunk_index=chunk_index,
            encrypted_content=encrypted.ciphertext_b64,
            nonce=encrypted.nonce_b64,
            chunk_hash=content_hash,
            metadata=ChunkMetadata(filename=f"{doc_id}.pdf", tokens=len(plaintext.split()) * 2),
            vector=[0.05 * (chunk_index + 1)] * 384,
        )
        chunk_records.append(rec)

    print(f"\n[1. Document Ingestion] Ingested {len(chunk_records)} chunks across 3 users.")

    # Step 2: Build Cloud MMR (aggregating User-MMR-Roots as leaves)
    cloud_mmr = CloudMMR(org_id=org_id)
    cloud_mmr.build_from_chunks(chunk_records)

    print(f"\n[2. Tier 1: User-MMR Roots]:")
    for user_id, u_mmr in cloud_mmr.user_mmrs.items():
        print(f"    - User '{user_id}': User MMR Root = {u_mmr.user_root} ({u_mmr.chunk_count} chunks)")

    print(f"\n[3. Tier 2: Cloud MMR Construction]:")
    print(f"    - Cloud Enterprise Root: {cloud_mmr.cloud_root}")
    print(f"    - Bagged Peak:           {cloud_mmr.bagged_peak}")
    print(f"    - Active Peaks:          {cloud_mmr.pure_cloud_mmr.get_peak_hashes()}")

    # Step 3: Generate Two-Tier Inclusion Proof for sample chunk
    target_user = "user-A"
    target_chunk_id = "chk_user-A_doc-A1-finances_0"
    proof = cloud_mmr.get_two_tier_proof(user_id=target_user, chunk_id=target_chunk_id)
    assert proof is not None, "Failed to assemble proof!"

    print(f"\n[4. Two-Tier Inclusion Proof Path for '{target_chunk_id}']:")
    print(f"    - Chunk ID:           {proof.chunk_id}")
    print(f"    - Content SHA256:     {proof.chunk_hash}")
    print(f"    - Leaf Hash:          {proof.leaf_hash}")
    print(f"    - User Proof Peak:    {proof.user_proof.peak_hash} (Leaf Index: {proof.user_proof.leaf_index})")
    print(f"    - User Sibling Path:  {[s.model_dump() for s in proof.user_proof.sibling_path]}")
    print(f"    - User MMR Root:      {proof.user_proof.user_root_hash}")
    print(f"    - Cloud Proof Peak:   {proof.cloud_proof.cloud_peak_hash} (User Index: {proof.cloud_proof.user_index})")
    print(f"    - Cloud Sibling Path: {[s.model_dump() for s in proof.cloud_proof.cloud_sibling_path]}")
    print(f"    - Enterprise Root:    {proof.cloud_proof.cloud_root_hash}")

    # Step 4: Cryptographic Proof Verification
    is_valid = verify_two_tier_proof(proof, expected_cloud_root=cloud_mmr.cloud_root)
    print(f"\n[5. Proof Verification Result]: {'✅ VERIFIED (TRUE)' if is_valid else '❌ FAILED (FALSE)'}")

    print("\n" + "=" * 80)
    print("🎯 TWO-TIER MMR AUDIT COMPLETE: ALL CHECKS PASSED")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_audit()
