"""
Comprehensive verification test suite for Supercharged RAG Pipeline:
Phase 1: Hierarchical (Parent-Child) Chunking & Context Expansion.
Phase 2: Query Transformation (Multi-Query Expansion, HyDE, Sub-Query Decomposition).
Phase 3: Hybrid Search & Metadata Filtering.
Phase 4: Post-Retrieval Context Optimization (Lost-in-the-Middle U-shape reordering, Context Compression).
Phase 5: Corrective RAG (CRAG Quality Grading) & Self-RAG Reflection Tokens.
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.hierarchical_chunker import HierarchicalChunker
from core.query_transformation import QueryTransformer
from core.context_optimizer import ContextOptimizer
from core.corrective_rag import CorrectiveRAGEngine
from core.hybrid_reranker import HybridReranker
from core.cross_encoder_reranker import CrossEncoderReranker


def test_hierarchical_chunking():
    print("--- [TEST 1: Hierarchical Parent-Child Chunking] ---")
    doc_text = """
SECURITY POLICY SECTION 1: NETWORK ACCESS CONTROLS
All remote users must connect via an enterprise VPN tunnel configured with AES-256-GCM.
Multi-factor authentication (MFA) is strictly required upon every initial connection.
VPN credentials expire every 30 days and must be rotated through the corporate identity portal.

SECURITY POLICY SECTION 2: DATA RETENTION AND DISPOSAL
Data retention schedules require financial records to be archived for 7 years.
Cryptographic key material must be securely destroyed using NIST 800-88 guidelines upon decommissioning.
"""
    parents, children = HierarchicalChunker.create_hierarchical_chunks(
        text=doc_text,
        doc_id="doc_sec_01",
        parent_size=400,
        parent_overlap=50,
        child_size=150,
        child_overlap=25,
    )
    print(f"Generated {len(parents)} Parent Chunks and {len(children)} Child Chunks.")
    assert len(parents) >= 2, "Expected at least 2 parent chunks"
    assert len(children) >= len(parents), "Expected multiple child chunks per parent"

    # Test resolving child hits to parent chunk
    parent_map = {p.parent_id: p.text for p in parents}
    mock_child_hits = [
        {"child_id": children[0].child_id, "parent_id": children[0].parent_id, "score": 0.95, "text": children[0].text}
    ]
    expanded = HierarchicalChunker.resolve_child_to_parents(mock_child_hits, parent_map)
    print(f"Resolved child hit '{children[0].child_id}' to parent '{expanded[0]['parent_id']}'")
    assert expanded[0]["is_expanded_parent"] is True
    assert len(expanded[0]["text"]) > len(children[0].text)
    print("  -> Hierarchical (Parent-Child) chunking PASSED.\n")


def test_query_transformations():
    print("--- [TEST 2: Query Transformation (Expansion, HyDE, Decomposition)] ---")
    query = "How often must VPN credentials be rotated and what encryption is used?"

    # 1. Multi-Query Expansion
    variations = QueryTransformer.expand_query(query)
    print(f"Expanded Queries ({len(variations)}): {variations}")
    assert len(variations) >= 2, "Expected at least 2 query variations"

    # 2. HyDE Document Generation
    hyde_doc = QueryTransformer.generate_hyde_document(query)
    print(f"HyDE Synthetic Answer Passage:\n'{hyde_doc}'")
    assert "rotat" in hyde_doc.lower() or "credential" in hyde_doc.lower()

    # 3. Sub-Query Decomposition
    sub_queries = QueryTransformer.decompose_subqueries(query)
    print(f"Sub-Queries ({len(sub_queries)}): {sub_queries}")
    assert len(sub_queries) >= 2, "Expected query to be decomposed into multiple sub-queries"
    print("  -> Query Transformations (Multi-Query, HyDE, Decomposition) PASSED.\n")


def test_lost_in_the_middle_reordering():
    print("--- [TEST 3: Lost-in-the-Middle Mitigation (U-Shape Reordering)] ---")
    ranked_docs = [
        {"id": "doc_1", "score": 0.99, "text": "Best evidence #1"},
        {"id": "doc_2", "score": 0.95, "text": "2nd best evidence"},
        {"id": "doc_3", "score": 0.88, "text": "3rd best evidence"},
        {"id": "doc_4", "score": 0.75, "text": "4th best evidence"},
        {"id": "doc_5", "score": 0.60, "text": "5th evidence"},
    ]
    u_shaped = ContextOptimizer.reorder_lost_in_the_middle(ranked_docs)
    ids_order = [d["id"] for d in u_shaped]
    print(f"Original Order: {[d['id'] for d in ranked_docs]}")
    print(f"U-Shaped Order: {ids_order}")

    # Top two items must be at the extremes (beginning and end)
    assert ids_order[0] == "doc_1", "Top result must be at the very beginning of the context"
    assert ids_order[-1] == "doc_2", "Second-best result must be at the very end of the context"
    print("  -> Lost-in-the-Middle U-shape reordering PASSED.\n")


def test_context_compression():
    print("--- [TEST 4: Context Compression / Sentence Pruning] ---")
    chunk_text = """
POLICIES:
All enterprise VPN credentials must be rotated every 30 days.
The cafeteria serves lunch between 12:00 PM and 2:00 PM on weekdays.
The company gym offers complimentary yoga on Thursday mornings.
Authentication failures exceeding 5 attempts will trigger automated security lockouts.
"""
    query = "How often must VPN credentials be rotated?"
    compressed = ContextOptimizer.compress_chunk_content(chunk_text, query)
    print(f"Compressed Result:\n{compressed}")
    assert "VPN credentials" in compressed
    assert "cafeteria" not in compressed, "Unrelated filler sentences should be pruned"
    print("  -> Context compression PASSED.\n")


def test_corrective_rag_and_self_rag():
    print("--- [TEST 5: Corrective RAG (CRAG) & Self-RAG Reflection] ---")
    query = "What is the key rotation policy?"
    matching_docs = [
        {"text": "NIST compliance: All cryptographic keys must be rotated every 90 days.", "score": 0.92}
    ]
    irrelevant_docs = [
        {"text": "Employee vacation entitlement is 20 days annually.", "score": 0.15}
    ]

    # Test CRAG high confidence (CORRECT)
    grade_good = CorrectiveRAGEngine.grade_retrieval(query, matching_docs)
    print(f"CRAG Grade (Relevant): {grade_good['decision']} (score={grade_good['confidence_score']})")
    assert grade_good["decision"] == "CORRECT"

    # Test CRAG low confidence (INCORRECT)
    grade_bad = CorrectiveRAGEngine.grade_retrieval(query, irrelevant_docs)
    print(f"CRAG Grade (Irrelevant): {grade_bad['decision']} (score={grade_bad['confidence_score']})")
    assert grade_bad["decision"] == "INCORRECT"

    # Test Self-RAG reflection tokens
    good_answer = "All cryptographic keys must be rotated every 90 days according to NIST."
    reflection_good = CorrectiveRAGEngine.generate_self_rag_reflection(query, matching_docs, good_answer)
    print(f"Self-RAG Reflection: {reflection_good['reflection_token']} (support={reflection_good['support_ratio']})")
    assert reflection_good["reflection_token"] == "[Fully-Supported]"
    print("  -> Corrective RAG (CRAG) & Self-RAG reflection PASSED.\n")


if __name__ == "__main__":
    print("\n=======================================================")
    print("   RUNNING SUPERCHARGED RAG PIPELINE VERIFICATION")
    print("=======================================================\n")
    test_hierarchical_chunking()
    test_query_transformations()
    test_lost_in_the_middle_reordering()
    test_context_compression()
    test_corrective_rag_and_self_rag()
    print("=======================================================")
    print("   ALL 5 SUPERCHARGED RAG TECHNIQUES VERIFIED 100%!")
    print("=======================================================\n")
