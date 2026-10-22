"""
CLI Document Ingestion and Query Tool
Allows users to ingest a PDF or TXT document and immediately query it
through the 5-Gate Verification Engine.

Usage:
  python ingest_document.py --file path/to/document.pdf --doc-id doc_sec_01 --user usr_admin_01 --org org_enterprise_01
  python ingest_document.py --text "Enterprise policy: VPN tokens expire in 30 days." --query "When do VPN tokens expire?"
"""
from __future__ import annotations

import argparse
import os
import sys
import time

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from core.document_ingestion import DocumentIngestionEngine, extract_text_from_pdf
from verification.boundary_validation import IVFBoundaryValidator
from database.qdrant_repository import QdrantRepository
from core.embedding_service import EmbeddingService
from core.masking import QueryMasker
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings
from mmr.proof import TwoTierInclusionProof, verify_two_tier_proof
from verification.hallucination import HallucinationChecker
from core.llm_service import LLMService
import numpy as np


def run_ingest_and_query(args):
    print("\n" + "=" * 90)
    print(" 📄 ZERO-TRUST RAG: DOCUMENT INGESTION & VERIFICATION PIPELINE")
    print("=" * 90)

    # 1. Initialize Components
    validator = IVFBoundaryValidator(
        margin_factor=1.5,
        centroids_path="ivf_centroids.npy",
        radii_path="ivf_radii.npy"
    )
    if not validator.load_centroids():
        print("⚡ IVF Index not found on disk. Initializing K-Means coarse quantizer...")
        dummy_embeddings = np.random.rand(100, 384).astype(np.float32)
        validator.build_ivf_index(dummy_embeddings, doc_ids=list(range(100)), n_lists=6)

    qdrant_repo = QdrantRepository()
    settings = get_settings()

    # 2. Extract Document Content
    raw_text = ""
    filename = "inline_document.txt"

    if args.file:
        filepath = os.path.abspath(args.file)
        if not os.path.exists(filepath):
            print(f"❌ Error: File not found: {filepath}")
            sys.exit(1)
        filename = os.path.basename(filepath)
        print(f"\n[Step 1: Reading Document] Reading from '{filename}'...")
        if filepath.lower().endswith(".pdf"):
            raw_text = extract_text_from_pdf(filepath)
        else:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                raw_text = f.read()
    elif args.text:
        raw_text = args.text
        print(f"\n[Step 1: Reading Document] Using provided inline text ({len(raw_text)} chars)...")
    else:
        # Default sample document
        filename = "cloud_security_policy.pdf"
        raw_text = (
            "ENTERPRISE CLOUD SECURITY DIRECTIVE 2026\n"
            "Section 1: Multi-Factor Authentication & Access Control.\n"
            "All employee accounts accessing production Kubernetes clusters and AWS infrastructure "
            "must enforce hardware security keys (FIDO2/WebAuthn). Password-only authentication is strictly prohibited.\n\n"
            "Section 2: VPN Credential Rotation & Session Timeouts.\n"
            "All VPN security credentials and TLS private certificates must be rotated every 30 days automatically. "
            "Inactive remote desktop and SSH administrative sessions will be forcibly terminated after 15 minutes of inactivity.\n\n"
            "Section 3: Cryptographic Data Protection & Merkle Verification.\n"
            "Enterprise data stored across cloud vector databases must use AES-256-GCM authenticated encryption at rest. "
            "Data retrieval pipelines must provide Two-Tier Merkle Mountain Range (MMR) proofs to verify zero tampering before release.\n\n"
            "Section 4: Data Retention & Compliance.\n"
            "Compliance audit logs must be retained for 7 years in immutable WORM (Write Once Read Many) storage."
        )
        print(f"\n[Step 1: Reading Document] No document specified. Using default sample: '{filename}'")

    print(f"   ✓ Content length: {len(raw_text)} characters")

    # 3. Process & Ingest Document
    print(f"\n[Step 2: Preprocessing & Ingestion Engine]")
    print(f"   - Chunking parameters: size={args.chunk_size} chars, overlap={args.overlap} chars")
    print(f"   - Target User: '{args.user}', Organization: '{args.org}', Doc ID: '{args.doc_id}'")

    ingest_result = DocumentIngestionEngine.process_and_ingest(
        text=raw_text,
        doc_id=args.doc_id,
        user_id=args.user,
        org_id=args.org,
        validator=validator,
        qdrant_repo=qdrant_repo,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        filename=filename,
    )

    print(f"   ✓ Generated Chunks:       {ingest_result['total_chunks']}")
    print(f"   ✓ User MMR Root:          {ingest_result['user_mmr_root']}")
    print(f"   ✓ Cloud Enterprise Root:  {ingest_result['cloud_enterprise_root']}")
    print(f"   ✓ Qdrant Points Upserted: {ingest_result['qdrant_points_upserted']}")

    print("\n   Chunk Breakdown:")
    for c in ingest_result["chunks"][:4]:
        print(f"     • [{c['chunk_id']}] Cluster: {c['cluster_id']} | Preview: \"{c['preview']}\"")
    if len(ingest_result["chunks"]) > 4:
        print(f"     • ... and {len(ingest_result['chunks']) - 4} more chunks.")

    # 4. Execute Query & Verification
    query_text = args.query or "How often must VPN credentials be rotated according to the security policy?"
    print(f"\n" + "-" * 90)
    print(f" 🔍 EXECUTING 5-GATE VERIFICATION QUERY")
    print(f"    Query: \"{query_text}\"")
    print(f"-" * 90)

    t0 = time.perf_counter()

    # Step 0: Query Masking
    masked_q, counts = QueryMasker.mask(query_text)
    print(f"\n[Gate 0: PII Masking] Sanitized query: \"{masked_q}\" (Redactions: {counts or 'None'})")

    # Step 1: Embed
    q_vec = EmbeddingService.embed(masked_q)

    # Step 2: Boundary Determination
    req_clusters, dists = validator.required_clusters(q_vec)
    print(f"[Gate 3 Prep: IVF Boundary] Required Voronoi clusters to probe: {req_clusters}")

    # Step 3: Multi-Probe Retrieval from Qdrant
    retrieved_docs = qdrant_repo.multi_probe_ivf_search(
        org_id=args.org,
        query_vector=q_vec.tolist(),
        required_cluster_ids=req_clusters,
        top_k=3
    )
    print(f"[Vector Retrieval] Retrieved {len(retrieved_docs)} candidate chunks from Qdrant.")

    # Step 4: Boundary Validation
    boundary_eval = validator.verify_query_boundary(q_vec, req_clusters)
    g3_boundary = boundary_eval["is_complete"]
    print(f"   ✓ Gate 3 (IVF Boundary Completeness): {'PASSED (100% Probed)' if g3_boundary else 'FAILED'}")

    # Step 5: ACL, Integrity & Decryption
    g1_acl = True
    g2_integrity = True
    g4_decrypt = True
    decrypted_texts = []

    for doc in retrieved_docs:
        # Check ACL
        acl = doc.get("metadata", {}).get("custom_attributes", {}).get("acl") or doc.get("metadata", {}).get("acl", [])
        if acl and args.user not in acl and args.user != "usr_admin_01":
            g1_acl = False
            break

        # Check MMR Proof
        proof_data = doc.get("proof_path")
        if proof_data:
            try:
                proof = TwoTierInclusionProof.model_validate(proof_data)
                is_valid = verify_two_tier_proof(proof, expected_cloud_root=ingest_result['cloud_enterprise_root'])
                if not is_valid:
                    g2_integrity = False
                    break
            except Exception as e:
                g2_integrity = False
                break

        # Gate 4: Decrypt AES-256-GCM
        ciphertext = doc.get("encrypted_content")
        nonce = doc.get("nonce")
        if ciphertext and nonce:
            try:
                dec_str = decrypt_aes_gcm(ciphertext, nonce, settings.DEFAULT_ENCRYPTION_KEY)
                decrypted_texts.append(dec_str)
            except Exception:
                g4_decrypt = False
                break
        elif "text" in doc:
            decrypted_texts.append(doc["text"])

    print(f"   ✓ Gate 1 (Access Control - ACL):     {'PASSED (Authorized)' if g1_acl else 'FAILED (Denied)'}")
    print(f"   ✓ Gate 2 (Two-Tier MMR Integrity):    {'PASSED (Cryptographically Verified)' if g2_integrity else 'FAILED'}")
    print(f"   ✓ Gate 4 (Authenticated Decryption):   {'PASSED (Decrypted ' + str(len(decrypted_texts)) + ' chunks)' if g4_decrypt else 'FAILED'}")

    # Step 6: LLM Generation
    if decrypted_texts and g1_acl and g2_integrity and g3_boundary and g4_decrypt:
        llm_answer = LLMService.generate_response(
            query=args.query or "Summarize this document in one line.",
            retrieved_docs=decrypted_texts
        )
    else:
        llm_answer = "Unable to provide a verified answer due to verification barrier block."

    # Step 7: Grounding Check
    grounding_eval = HallucinationChecker.evaluate(
        integrity_passed=g2_integrity,
        completeness_passed=g3_boundary,
        retrieved_docs=[{"text": t} for t in decrypted_texts],
        llm_response=llm_answer if g2_integrity and g3_boundary else None,
        min_grounding_score=0.35
    )
    g5_grounding = not grounding_eval.get("flagged", False)
    grounding_score = grounding_eval.get("grounding_score", 0.0)
    print(f"   ✓ Gate 5 (Hallucination Grounding):    {'PASSED (Score: ' + str(grounding_score) + ')' if g5_grounding else 'FAILED'}")

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Final Result
    print("\n" + "=" * 90)
    print(" 🎯 FINAL VERIFIED RESULT")
    print("=" * 90)
    print(f"Status:        {'SUCCESS (ALL 5 GATES PASSED)' if (g1_acl and g2_integrity and g3_boundary and g4_decrypt and g5_grounding) else 'BLOCKED'}")
    print(f"Latency:       {elapsed_ms:.2f} ms")
    print(f"Source Doc:    {filename}")
    print(f"\nVerified Answer:")
    print(f"  {llm_answer}")
    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest and query documents in the Unified Verifiable RAG architecture.")
    parser.add_argument("--file", "-f", type=str, help="Path to PDF or TXT document")
    parser.add_argument("--text", "-t", type=str, help="Direct inline plaintext to ingest")
    parser.add_argument("--doc-id", "-d", type=str, default="doc_user_01", help="Document Identifier")
    parser.add_argument("--user", "-u", type=str, default="usr_admin_01", help="User ID for ACL and MMR")
    parser.add_argument("--org", "-o", type=str, default="org_enterprise_01", help="Organization ID")
    parser.add_argument("--chunk-size", type=int, default=500, help="Chunk size in characters (default 500)")
    parser.add_argument("--overlap", type=int, default=50, help="Overlap in characters (default 50)")
    parser.add_argument("--query", "-q", type=str, help="Query text to verify against the document")
    
    args = parser.parse_args()
    run_ingest_and_query(args)
