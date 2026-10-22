import requests

def test_upload():
    print("--- 1. Testing /api/v1/ingest-text ---")
    res1 = requests.post('http://127.0.0.1:8000/api/v1/ingest-text', json={
        'text': 'Internal cloud security protocol: All production SSH keys must be rotated every 30 days without exception.',
        'doc_id': 'test_security_policy_01',
        'user_id': 'usr_admin_01',
        'org_id': 'org_enterprise_01',
        'chunk_size': 500,
        'overlap': 50,
        'filename': 'security_policy_01.txt'
    })
    print('Ingest text status:', res1.status_code)
    try:
        print('Ingest text response:', res1.json())
    except Exception as e:
        print('Ingest text raw:', res1.text)

    print("\n--- 2. Testing /api/v1/ingest-document (TXT file) ---")
    files = {'file': ('test_upload_doc.txt', b'Clinical Oncology Protocol: Patient dosage must not exceed 45mg per square meter of body surface area.', 'text/plain')}
    data = {
        'doc_id': 'test_clinical_protocol_01',
        'user_id': 'usr_clinical_director',
        'org_id': 'org_healthcare_plus',
        'chunk_size': '500',
        'overlap': '50'
    }
    res2 = requests.post('http://127.0.0.1:8000/api/v1/ingest-document', files=files, data=data)
    print('Ingest document status:', res2.status_code)
    try:
        print('Ingest document response:', res2.json())
    except Exception as e:
        print('Ingest document raw:', res2.text)

    print("\n--- 3. Querying to verify document retrieval ---")
    res3 = requests.post('http://127.0.0.1:8000/api/v1/query-verify', json={
        'query_text': 'How often must production SSH keys be rotated?',
        'user_id': 'usr_admin_01',
        'org_id': 'org_enterprise_01'
    })
    print('Query verify status:', res3.status_code)
    try:
        ans = res3.json()
        print('Query verdict:', ans.get('verdict'))
        print('Generated answer:', ans.get('answer'))
    except Exception as e:
        print('Query verify raw:', res3.text)

if __name__ == '__main__':
    test_upload()
