from database.qdrant_repository import QdrantRepository
import json

repo = QdrantRepository()
collections = ['org_chunks_org_enterprise_01', 'org_chunks_org_finance_corp', 'org_chunks_org_healthcare_plus']

for col in collections:
    print(f"\n========================================================")
    print(f"Collection: {col}")
    print(f"========================================================")
    try:
        points, _ = repo.client.scroll(collection_name=col, limit=150, with_payload=True)
    except Exception as e:
        print(f"Error fetching {col}: {e}")
        continue

    doc_acls = {}
    for p in points:
        pl = p.payload or {}
        meta = pl.get('metadata', {})
        custom = meta.get('custom_attributes', {})
        acl = custom.get('acl') or meta.get('acl')
        did = pl.get('doc_id')
        uid = pl.get('user_id')
        fname = meta.get('filename')
        if did not in doc_acls:
            doc_acls[did] = {
                'filename': fname,
                'user_id': uid,
                'acl': acl,
                'chunk_count': 0
            }
        doc_acls[did]['chunk_count'] += 1

    for did, info in doc_acls.items():
        print(f"  Doc: {did}")
        print(f"    Filename: {info['filename']}")
        print(f"    Owner:    {info['user_id']}")
        print(f"    ACL:      {info['acl']}")
        print(f"    Chunks:   {info['chunk_count']}")
