import os
import sys
import uuid
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from crypto.hashing import sha256_hex, hash_mmr_node, bag_peaks_hash, hash_user_root, hash_cloud_root, hash_chunk_leaf
from crypto.encryption import encrypt_aes_gcm
from database.schemas import ChunkMetadata, EncryptedChunkRecord
from mmr.core import PureMMR
from mmr.user_mmr import UserMMR
from mmr.cloud_mmr import CloudMMR
from mmr.proof import (
    TwoTierInclusionProof,
    UserProof,
    CloudProof,
    verify_user_proof,
    verify_cloud_proof,
    verify_two_tier_proof,
)

def create_mock_chunk(user_id: str, doc_id: str, idx: int, org_id: str = "org_test") -> EncryptedChunkRecord:
    text = f"Sample confidential enterprise text chunk {idx} for user {user_id} in {doc_id}."
    c_hash = sha256_hex(text)
    enc = encrypt_aes_gcm(text, "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef")
    chunk_id = f"chk_{user_id}_{doc_id}_{idx}"
    return EncryptedChunkRecord(
        point_id=str(uuid.uuid4()),
        chunk_id=chunk_id,
        doc_id=doc_id,
        org_id=org_id,
        user_id=user_id,
        chunk_index=idx,
        encrypted_content=enc.ciphertext_b64,
        nonce=enc.nonce_b64,
        chunk_hash=c_hash,
        metadata=ChunkMetadata(filename=f"{doc_id}.txt", tokens=10),
        vector=[0.01 * (idx + 1)] * 384,
        cluster_id="0",
    )

def run_mmr_tests():
    print("=" * 70)
    print("TESTING MERKLE MOUNTAIN RANGE (MMR) MODULE")
    print("=" * 70)

    # ---------------------------------------------------------
    # TEST 1: PureMMR core mathematical properties
    # ---------------------------------------------------------
    print("\n[Test 1] PureMMR Core Mathematical Properties...")
    pure = PureMMR()
    # Insert 7 leaves: in binary, 7 = 4 + 2 + 1, so MMR should have 3 peaks of heights 2, 1, 0.
    leaf_hashes = [sha256_hex(f"leaf_data_{i}") for i in range(7)]
    for lh in leaf_hashes:
        pure.append(lh)

    assert pure.leaf_count == 7, f"Expected 7 leaves, got {pure.leaf_count}"
    assert pure.peak_count == 3, f"Expected 3 peaks for 7 leaves (4+2+1), got {pure.peak_count}"
    peak_heights = [p.height for p in pure._peaks]
    assert peak_heights == [2, 1, 0], f"Expected peak heights [2, 1, 0], got {peak_heights}"
    print(f"   [PASS] 7 leaves correctly merged into {pure.peak_count} peaks with heights {peak_heights}.")

    # Test sibling path ascension to peak for leaf 0 and leaf 5
    for target_idx in [0, 5]:
        sibling_path = pure.get_sibling_path_to_peak(target_idx)
        peak_info = pure.get_peak_containing_leaf(target_idx)
        assert peak_info is not None, f"No peak found for leaf {target_idx}"
        p_idx, p_obj = peak_info

        # Reconstruct peak by hashing along sibling path
        curr = leaf_hashes[target_idx]
        for shash, sdir in sibling_path:
            if sdir == "left":
                curr = hash_mmr_node(shash, curr)
            else:
                curr = hash_mmr_node(curr, shash)
        assert curr == p_obj.hash, f"Reconstructed peak does not match peak hash for leaf {target_idx}!"
    print("   [PASS] Sibling path correctly reconstructs peak hashes.")

    # ---------------------------------------------------------
    # TEST 2: UserMMR Construction, Proof Generation & Verification
    # ---------------------------------------------------------
    print("\n[Test 2] UserMMR Construction & User Proof Verification...")
    user_id = "user_alice"
    org_id = "org_cyber"
    chunks_u1 = [create_mock_chunk(user_id, "doc_policies", i, org_id) for i in range(5)]

    user_mmr = UserMMR(user_id=user_id, org_id=org_id)
    user_mmr.build_from_chunks(chunks_u1)

    assert user_mmr.chunk_count == 5
    assert len(user_mmr.user_root) == 64, "User root hash must be 64-char hex string"
    print(f"   User MMR Root: {user_mmr.user_root}")

    # Generate and verify proof for chunk 2
    target_chk = chunks_u1[2]
    user_proof = user_mmr.get_proof_for_chunk(target_chk.chunk_id)
    assert user_proof is not None, "Failed to generate UserProof"

    leaf_hash = hash_chunk_leaf(
        chunk_id=target_chk.chunk_id,
        doc_id=target_chk.doc_id,
        user_id=target_chk.user_id,
        org_id=target_chk.org_id,
        chunk_index=target_chk.chunk_index,
        content_hash=target_chk.chunk_hash,
    )
    is_user_valid = verify_user_proof(leaf_hash, user_proof)
    assert is_user_valid is True, "User proof validation failed for untampered chunk!"
    print("   [PASS] UserProof verified successfully.")

    # User proof tampering tests
    tampered_u_proof = user_proof.model_copy(deep=True)
    tampered_u_proof.user_root_hash = "0" * 64
    assert verify_user_proof(leaf_hash, tampered_u_proof) is False, "Tampered root passed!"

    tampered_leaf_proof = user_proof.model_copy(deep=True)
    assert verify_user_proof("f" * 64, tampered_leaf_proof) is False, "Forged leaf hash passed!"
    print("   [PASS] UserProof correctly rejects tampered root and forged leaf.")

    # ---------------------------------------------------------
    # TEST 3: Multi-User CloudMMR & Two-Tier Inclusion Proof
    # ---------------------------------------------------------
    print("\n[Test 3] CloudMMR Construction & Two-Tier Proof Verification...")
    all_chunks = []
    # Alice: 3 chunks, Bob: 4 chunks, Charlie: 2 chunks
    for u, count in [("user_alice", 3), ("user_bob", 4), ("user_charlie", 2)]:
        for i in range(count):
            all_chunks.append(create_mock_chunk(u, f"doc_{u}", i, org_id))

    cloud_mmr = CloudMMR(org_id=org_id)
    cloud_mmr.build_from_chunks(all_chunks)

    assert cloud_mmr.user_count == 3
    assert cloud_mmr.total_chunks == 9
    assert len(cloud_mmr.cloud_root) == 64
    print(f"   Cloud MMR Root: {cloud_mmr.cloud_root}")

    # Verify each user's root proof
    for u in ["user_alice", "user_bob", "user_charlie"]:
        c_proof = cloud_mmr.get_user_proof(u)
        assert c_proof is not None, f"Failed to get CloudProof for {u}"
        u_root = cloud_mmr.user_mmrs[u].user_root
        assert verify_cloud_proof(u_root, c_proof) is True, f"Cloud proof validation failed for {u}!"
    print("   [PASS] All User MMR roots verified against Cloud Enterprise Root.")

    # ---------------------------------------------------------
    # TEST 4: End-to-End Two-Tier Proofs & Attack Tampering Scenarios
    # ---------------------------------------------------------
    print("\n[Test 4] Two-Tier Inclusion Proof & Attack Tampering Scenarios...")
    test_user = "user_bob"
    test_chunk_id = f"chk_{test_user}_doc_{test_user}_1"
    proof = cloud_mmr.get_two_tier_proof(user_id=test_user, chunk_id=test_chunk_id)
    assert proof is not None, "Failed to get TwoTierInclusionProof"

    # 4a. Untampered proof
    assert verify_two_tier_proof(proof) is True, "Untampered Two-Tier proof failed verification!"
    assert verify_two_tier_proof(proof, expected_cloud_root=cloud_mmr.cloud_root) is True
    print("   [PASS] 4a. Untampered proof passes full 2-tier cryptographic verification.")

    # 4b. Content tampering (adversary alters decrypted payload or plaintext hash)
    proof_content_tampered = proof.model_copy(deep=True)
    proof_content_tampered.chunk_hash = sha256_hex("malicious spliced text payload")
    assert verify_two_tier_proof(proof_content_tampered) is False, "Content tampering went undetected!"
    print("   [PASS] 4b. Content tampering attack correctly DETECTED & REJECTED.")

    # 4c. Splicing / Metadata substitution (adversary swaps chunk_index or user_id)
    proof_meta_tampered = proof.model_copy(deep=True)
    proof_meta_tampered.chunk_index = 999
    assert verify_two_tier_proof(proof_meta_tampered) is False, "Metadata substitution went undetected!"
    print("   [PASS] 4c. Metadata substitution attack correctly DETECTED & REJECTED.")

    # 4d. Forged Cloud Enterprise Root
    fake_cloud_root = "abcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcdefabcd"
    assert verify_two_tier_proof(proof, expected_cloud_root=fake_cloud_root) is False
    print("   [PASS] 4d. Forged enterprise root correctly DETECTED & REJECTED.")

    # 4e. Sibling path tampering in Tier 1 (User MMR)
    proof_tier1_tampered = proof.model_copy(deep=True)
    if proof_tier1_tampered.user_proof.sibling_path:
        proof_tier1_tampered.user_proof.sibling_path[0].sibling_hash = "0" * 64
        assert verify_two_tier_proof(proof_tier1_tampered) is False
        print("   [PASS] 4e. Tier-1 sibling path corruption correctly DETECTED & REJECTED.")

    # 4f. Sibling path tampering in Tier 2 (Cloud MMR)
    proof_tier2_tampered = proof.model_copy(deep=True)
    if proof_tier2_tampered.cloud_proof.cloud_sibling_path:
        proof_tier2_tampered.cloud_proof.cloud_sibling_path[0].sibling_hash = "0" * 64
        assert verify_two_tier_proof(proof_tier2_tampered) is False
        print("   [PASS] 4f. Tier-2 sibling path corruption correctly DETECTED & REJECTED.")

    # ---------------------------------------------------------
    # TEST 5: Serialization & Deserialization Persistence
    # ---------------------------------------------------------
    print("\n[Test 5] Serialization & Re-hydration Consistency...")
    hierarchy_dict = cloud_mmr.export_full_hierarchy()
    rehydrated_cloud = CloudMMR.from_dict(hierarchy_dict)

    assert rehydrated_cloud.cloud_root == cloud_mmr.cloud_root, "Rehydrated cloud root mismatch!"
    assert rehydrated_cloud.user_count == cloud_mmr.user_count, "Rehydrated user count mismatch!"
    assert rehydrated_cloud.total_chunks == cloud_mmr.total_chunks, "Rehydrated total chunks mismatch!"

    rehydrated_proof = rehydrated_cloud.get_two_tier_proof(user_id=test_user, chunk_id=test_chunk_id)
    assert verify_two_tier_proof(rehydrated_proof, expected_cloud_root=rehydrated_cloud.cloud_root) is True
    print("   [PASS] Serialization/from_dict preserves exact MMR tree topology, roots, and proofs.")

    print("\n" + "=" * 70)
    print("ALL 5 TEST SUITES PASSED! MMR MODULE IS FULLY FUNCTIONAL & VERIFIED.")
    print("=" * 70)

if __name__ == "__main__":
    run_mmr_tests()
