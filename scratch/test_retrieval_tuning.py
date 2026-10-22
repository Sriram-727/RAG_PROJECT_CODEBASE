"""
Comprehensive test suite verifying RAG retrieval fine-tuning enhancements:
1. Semantic structure-aware chunking vs sliding window.
2. Section classification heuristics.
3. Hybrid BM25 + Dense + RRF scoring.
4. Cross-Encoder deep relevance reranking.
5. End-to-end retrieval and reranking pipeline.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.document_ingestion import chunk_text, classify_chunk_section
from core.hybrid_reranker import HybridReranker
from core.cross_encoder_reranker import CrossEncoderReranker, HeuristicCrossScorer


def test_semantic_chunking():
    sample_doc = """
JOHN DOE - SOFTWARE ARCHITECT
Email: john@example.com

PROFESSIONAL SUMMARY:
Experienced distributed systems engineer specializing in high-throughput data processing and security compliance.

PROJECTS:
1. Real-Time Fraud Detection Engine: Stream processing system built with Apache Kafka and Apache Flink, processing 50,000 transactions/sec.
2. Asymmetric Cryptographic Audit System: Zero-knowledge verification layer using Merkle Mountain Ranges and AES-256-GCM.

EDUCATION:
Bachelor of Science in Computer Science, University of Technology, 2020.
GPA: 3.9/4.0.

POLICIES AND COMPLIANCE:
All cryptographic keys must be rotated every 90 days according to enterprise NIST guidelines.
"""
    chunks = chunk_text(sample_doc, chunk_size=350, overlap=40, use_semantic=True)
    print(f"[TEST 1: Semantic Chunking] Generated {len(chunks)} chunks.")
    assert len(chunks) >= 2, "Expected at least 2 semantic chunks"
    
    # Check that sections are categorized appropriately
    classifications = [classify_chunk_section(c) for c in chunks]
    print(f"[TEST 1: Classifications] {classifications}")
    assert any(c in ["projects", "security", "policy", "education"] for c in classifications)
    print("  -> Semantic chunking & classification PASSED.")


def test_hybrid_and_rrf():
    docs = [
        {"chunk_id": "c1", "text": "The rotating frequency for VPN credentials is every 30 days mandatory.", "score": 0.65},
        {"chunk_id": "c2", "text": "Enterprise employees receive 20 days of paid vacation per calendar year.", "score": 0.70},
        {"chunk_id": "c3", "text": "VPN access policy states that multi-factor authentication is required.", "score": 0.60},
    ]
    query = "How often must VPN credentials be rotated?"
    
    # Test BM25
    bm25_scores = HybridReranker.compute_bm25_scores(query, docs)
    print(f"[TEST 2: BM25 Scores] {bm25_scores}")
    assert bm25_scores[0] > bm25_scores[1], "Chunk 1 should have much higher BM25 score than Chunk 2"
    
    # Test RRF
    dense_scores = [d["score"] for d in docs]
    fused = HybridReranker.rrf_fuse(docs, dense_scores, bm25_scores, k=20)
    print(f"[TEST 2: RRF Top] Chunk ID: {fused[0]['chunk_id']}, Score: {fused[0]['score']}")
    assert fused[0]["chunk_id"] == "c1", "Chunk 1 must rank #1 under RRF fusion"
    print("  -> Hybrid BM25 & RRF fusion PASSED.")


def test_cross_encoder():
    query = "What is the project name for fraud detection?"
    docs = [
        {"chunk_id": "c1", "text": "Personal Hobbies: Avid marathon runner and amateur landscape photographer.", "score": 0.85},
        {"chunk_id": "c2", "text": "PROJECTS: 1. Real-Time Fraud Detection Engine built using Kafka and Flink.", "score": 0.60},
    ]
    
    reranked = CrossEncoderReranker.rerank(query, docs, top_k=2)
    print(f"[TEST 3: Cross-Encoder] Winner: {reranked[0]['chunk_id']}, Scores: {[d['score'] for d in reranked]}")
    assert reranked[0]["chunk_id"] == "c2", "Cross-Encoder must elevate Chunk 2 (Fraud Detection) to rank #1"
    print("  -> Cross-Encoder reranking PASSED.")


def test_end_to_end_pipeline():
    query = "NIST cryptographic key rotation guidelines"
    docs = [
        {"chunk_id": "c1", "text": "Annual leave policy: 15 paid time off days for full-time employees.", "score": 0.5},
        {"chunk_id": "c2", "text": "NIST Guidelines: Cryptographic keys must be rotated every 90 days using AES-256-GCM.", "score": 0.6},
        {"chunk_id": "c3", "text": "Office kitchen clean-up duties are rotated weekly among team leads.", "score": 0.75}, # Distractor with 'rotated'
    ]
    
    final_results = HybridReranker.execute_advanced_retrieval_pipeline(
        query=query,
        documents=docs,
        alpha=0.60,
        enable_rrf=True,
        enable_cross_encoder=True,
        top_k=3
    )
    print(f"[TEST 4: End-to-End Pipeline] Top match: {final_results[0]['chunk_id']} -> '{final_results[0]['text'][:50]}...'")
    assert final_results[0]["chunk_id"] == "c2", "Pipeline must select the NIST cryptographic key rotation chunk"
    print("  -> End-to-End Advanced Retrieval Pipeline PASSED.")


if __name__ == "__main__":
    print("\n=== RUNNING RAG RETRIEVAL TUNING VERIFICATION ===")
    test_semantic_chunking()
    test_hybrid_and_rrf()
    test_cross_encoder()
    test_end_to_end_pipeline()
    print("=== ALL 4 RETRIEVAL TUNING TESTS PASSED SUCCESSFULLY! ===\n")
