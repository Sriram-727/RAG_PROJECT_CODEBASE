from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("\n--- 1. Testing admin_tech query ---")
r1 = client.post('/api/v1/query-verify', json={
    'query_text': 'What are the cloud security access policies?',
    'user_id': 'admin_tech',
    'org_id': 'org_enterprise_01'
})
d1 = r1.json()
print("Status code:", r1.status_code)
print("ACL Verified:", d1.get('acl_verified'))
print("Pipeline Status:", d1.get('status'))
print("Documents count:", d1.get('retrieved_documents_count'))
print("Has answer:", bool(d1.get('answer')))

print("\n--- 2. Testing usr_admin_01 query ---")
r2 = client.post('/api/v1/query-verify', json={
    'query_text': 'What are the cloud security access policies?',
    'user_id': 'usr_admin_01',
    'org_id': 'org_enterprise_01'
})
d2 = r2.json()
print("Status code:", r2.status_code)
print("ACL Verified:", d2.get('acl_verified'))
print("Pipeline Status:", d2.get('status'))
print("Documents count:", d2.get('retrieved_documents_count'))
print("Has answer:", bool(d2.get('answer')))

print("\n--- 3. Testing alice_dev query (member with no access to cloud security) ---")
r3 = client.post('/api/v1/query-verify', json={
    'query_text': 'What are the cloud security access policies?',
    'user_id': 'alice_dev',
    'org_id': 'org_enterprise_01'
})
d3 = r3.json()
print("Status code:", r3.status_code)
print("ACL Verified:", d3.get('acl_verified'))
print("Pipeline Status:", d3.get('status'))
print("ACL Reason:", d3.get('verification_breakdown', {}).get('acl', {}).get('details'))
print("Has answer:", bool(d3.get('answer')))

print("\n--- 4. Testing unauthorized guest query ---")
r4 = client.post('/api/v1/query-verify', json={
    'query_text': 'What are the cloud security access policies?',
    'user_id': 'usr_guest_unauthorized',
    'org_id': 'org_enterprise_01'
})
d4 = r4.json()
print("Status code:", r4.status_code)
print("ACL Verified:", d4.get('acl_verified'))
print("Pipeline Status:", d4.get('status'))
print("Has answer:", bool(d4.get('answer')))
