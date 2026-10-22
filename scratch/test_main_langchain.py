import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

print("--- 1. Testing /api/v1/retrieve-langchain ---")
try:
    r1 = client.post("/api/v1/retrieve-langchain", json={"query_text": "cloud security access policies", "org_id": "org_1"})
    print("Status:", r1.status_code)
    data1 = r1.json()
    print("Retrieved chunks:", data1.get("retrieved_chunks_count"))
    for i, c in enumerate(data1.get("top_chunks", []), 1):
        print(f"  Chunk {i}:", c[:80].replace("\n", " "))
except Exception as e:
    print("Error on r1:", e)

print("\n--- 2. Testing /api/v1/query-verify with LangChain ---")
try:
    r2 = client.post("/api/v1/query-verify", json={"query_text": "What are the cloud security policies?", "user_id": "sriram10", "org_id": "org_1"})
    print("Status:", r2.status_code)
    data2 = r2.json()
    print("Answer:", data2.get("answer"))
    print("LangChain pipeline in query_enhancements:")
    print(" ", data2.get("query_enhancements", {}).get("langchain_pipeline"))
except Exception as e:
    print("Error on r2:", e)
