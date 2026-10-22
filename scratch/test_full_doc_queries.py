from database.qdrant_client import QdrantClientManager
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings
from core.llm_service import LLMService

client = QdrantClientManager.get_client()
key = get_settings().DEFAULT_ENCRYPTION_KEY
pts, _ = client.scroll("org_chunks_org_enterprise_01", limit=100, with_payload=True)
pts.sort(key=lambda p: (p.payload.get("doc_id", ""), p.payload.get("chunk_index", 0)))

all_chunks = []
for p in pts:
    pl = p.payload
    txt = decrypt_aes_gcm(pl['encrypted_content'], pl['nonce'], key) if 'encrypted_content' in pl else pl.get('text', '')
    all_chunks.append({
        "text": txt,
        "score": 0.95,
        "chunk_index": pl.get("chunk_index", 0),
        "doc_id": pl.get("doc_id", ""),
        "cluster_id": pl.get("cluster_id", "0"),
        "filename": "23i365_sriram_r_resume__3_.pdf"
    })

test_queries = [
    "list only the project names in the resume",
    "objectives",
    "what are the technical skills?",
    "what is the education?",
    "tell me about areas of interest",
    "contact details",
]

for q in test_queries:
    ans = LLMService.generate_response(query=q, retrieved_docs=all_chunks)
    safe_ans = ans.encode('ascii', errors='replace').decode('ascii')
    print(f"\n================ QUERY: {q} ================")
    print(safe_ans)
