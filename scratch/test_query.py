import urllib.request
import json
import sys

def test(q, user="usr_admin_01", org="org_enterprise_01"):
    url = "http://localhost:8000/api/v1/query-verify"
    payload = {
        "query_text": q,
        "org_id": org,
        "user_id": user
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            print("="*60)
            print(f"QUERY: {q}")
            print(f"ANSWER: {data.get('answer')}")
            print(f"STATUS: {data.get('status')}")
            chunks = data.get("decrypted_chunks", [])
            print(f"RETRIEVED CHUNKS COUNT: {len(chunks)}")
            for idx, c in enumerate(chunks[:3]):
                score = c.get('score')
                d_id = c.get('doc_id')
                c_idx = c.get('chunk_index')
                text_preview = c.get('text', '').replace('\n', ' ')[:150]
                print(f"  [{idx+1}] Score: {score} | Doc: {d_id} | ChunkIdx: {c_idx}")
                print(f"      Text: {text_preview}...")
            print("="*60)
    except Exception as e:
        print(f"Request failed: {e}")

if __name__ == "__main__":
    queries = [
        "what is multithreading in java give solution for this question",
        "tell the project listed in the resume",
        "tell about the career objective",
        "give company information",
        "what was the system design question"
    ]
    for query in queries:
        test(query)
