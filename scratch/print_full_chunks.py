from database.qdrant_client import QdrantClientManager
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings

client = QdrantClientManager.get_client()
key = get_settings().DEFAULT_ENCRYPTION_KEY
pts, _ = client.scroll("org_chunks_org_enterprise_01", limit=100, with_payload=True)

pts.sort(key=lambda p: (p.payload.get("doc_id", ""), p.payload.get("chunk_index", 0)))

for p in pts:
    pl = p.payload
    txt = decrypt_aes_gcm(pl['encrypted_content'], pl['nonce'], key) if 'encrypted_content' in pl else pl.get('text', '')
    print(f"\n--- DOC: {pl.get('doc_id')} | CHUNK INDEX: {pl.get('chunk_index')} | CLUSTER: {pl.get('cluster_id')} ---")
    safe_txt = txt.encode('ascii', errors='replace').decode('ascii')
    print(safe_txt)
