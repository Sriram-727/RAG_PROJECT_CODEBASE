import sys
from config.settings import get_settings
from database.qdrant_repository import QdrantRepository
from core.embedding_service import EmbeddingService
from core.masking import QueryMasker
from core.hybrid_reranker import HybridReranker
from core.llm_service import LLMService
from crypto.encryption import decrypt_aes_gcm
from verification.boundary_validation import IVFBoundaryValidator
from verification.hallucination import HallucinationChecker

qdrant_repo = QdrantRepository()
validator = IVFBoundaryValidator()
cfg_settings = get_settings()

query_text = "list only the project names in the resume"
masked_query_text, _ = QueryMasker.mask(query_text)
query_vector = EmbeddingService.embed(masked_query_text)
required_probe_lists, _ = validator.required_clusters(query_vector)

retrieved_docs = qdrant_repo.multi_probe_ivf_search(
    org_id="org_enterprise_01",
    query_vector=query_vector.tolist(),
    required_cluster_ids=required_probe_lists,
    top_k=8
)

print(f"Retrieved docs count: {len(retrieved_docs)}")
decrypted_texts = []
for doc in retrieved_docs:
    c = doc.get("encrypted_content")
    n = doc.get("nonce")
    if c and n:
        txt = decrypt_aes_gcm(c, n, cfg_settings.DEFAULT_ENCRYPTION_KEY)
        decrypted_texts.append({
            "text": txt,
            "score": float(doc.get("score") or 0.0),
            "chunk_index": int(doc.get("chunk_index", 0)),
            "doc_id": doc.get("doc_id", ""),
            "cluster_id": doc.get("cluster_id", "0"),
        })

print(f"Decrypted texts count: {len(decrypted_texts)}")
for d in decrypted_texts:
    safe = d['text'][:70].encode('ascii', errors='replace').decode('ascii')
    print(f"  idx={d['chunk_index']} score={d['score']:.4f} preview={safe!r}")

# Check LLM response
llm_resp = LLMService.generate_response(query=query_text, retrieved_docs=decrypted_texts)
print("\nLLM Response:")
print(llm_resp)

eval_res = HallucinationChecker.evaluate(
    integrity_passed=True,
    completeness_passed=True,
    retrieved_docs=decrypted_texts,
    llm_response=llm_resp,
    min_grounding_score=0.35
)
print("\nHallucination Report:")
print(eval_res)
