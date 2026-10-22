import sys
from database.qdrant_client import QdrantClientManager
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings

client = QdrantClientManager.get_client()
key = get_settings().DEFAULT_ENCRYPTION_KEY
collections = [c.name for c in client.get_collections().collections]
print("Collections:", collections)

for cname in collections:
    pts, _ = client.scroll(cname, limit=50, with_payload=True)
    print(f"\nCollection {cname}: {len(pts)} points")
    for p in pts:
        pl = p.payload or {}
        txt = ""
        if 'encrypted_content' in pl and 'nonce' in pl:
            try:
                txt = decrypt_aes_gcm(pl['encrypted_content'], pl['nonce'], key)
            except Exception as e:
                txt = f"[Decryption Error: {e}]"
        else:
            txt = pl.get('text', '')
        safe_preview = txt[:80].encode('ascii', errors='replace').decode('ascii')
        print(f"  doc_id={pl.get('doc_id')}, idx={pl.get('chunk_index')}, cluster={pl.get('cluster_id')}, preview={safe_preview!r}")
