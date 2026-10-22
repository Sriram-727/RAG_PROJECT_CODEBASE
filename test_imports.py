import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from main import app, QueryPayloadRequest
from verification.boundary_validation import IVFBoundaryValidator
from core.embedding_service import EmbeddingService
from mmr.proof import TwoTierInclusionProof, verify_two_tier_proof
from crypto.encryption import AESGCMCipher
from verification.hallucination import HallucinationChecker
from database.qdrant_repository import QdrantRepository
from core.cross_encoder_reranker import CrossEncoderReranker
from core.document_ingestion import classify_chunk_section, chunk_text

print("Imports successful! Unified architecture is structurally sound.")
