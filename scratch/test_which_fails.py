import sys
import os
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

items = [
    ("boundary_validation", "from verification.boundary_validation import IVFBoundaryValidator"),
    ("hallucination", "from verification.hallucination import HallucinationChecker"),
    ("qdrant_repo", "from database.qdrant_repository import QdrantRepository"),
    ("embedding", "from core.embedding_service import EmbeddingService"),
    ("masking", "from core.masking import QueryMasker"),
    ("ingestion", "from core.document_ingestion import DocumentIngestionEngine"),
    ("llm", "from core.llm_service import LLMService"),
    ("reranker", "from core.hybrid_reranker import HybridReranker"),
    ("auth", "from core.auth_service import AuthService"),
    ("crypto", "from crypto.encryption import decrypt_aes_gcm"),
    ("mmr", "from mmr.proof import TwoTierInclusionProof, verify_two_tier_proof"),
    ("hashing", "from crypto.hashing import sha256_hex"),
    ("langchain", "from core.langchain_retriever import retrieve_optimized_chunks"),
]

for name, stmt in items:
    try:
        exec(stmt)
        print(f"SUCCESS: {name}")
    except Exception as e:
        print(f"FAILED: {name} -> {e}")
        traceback.print_exc()

print("ALL TEST IMPORTS FINISHED.")
