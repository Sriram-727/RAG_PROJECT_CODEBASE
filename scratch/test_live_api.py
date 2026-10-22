import json
import urllib.request

def test_query(q, sim_cluster=False, sim_integrity=False):
    url = "http://127.0.0.1:8000/api/v1/query-verify"
    payload = {
        "query_text": q,
        "user_id": "usr_enterprise_01",
        "org_id": "org_enterprise_01",
        "simulate_missing_cluster": sim_cluster,
        "simulate_integrity_failure": sim_integrity
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        print("="*60)
        print(f"QUERY: {q}")
        print(f"STATUS: {data.get('status')}")
        print(f"ANSWER:\n{data.get('answer')}")

if __name__ == "__main__":
    test_queries = [
        # The user's exact query with typo and negation:
        "frontent languages only not backend",
        # General skills query:
        "what are the technical skills?",
        # Skills with exclusion:
        "all skills except backend",
        # Skills combining multiple:
        "frontend and database",
        # Projects all:
        "list only the project names in the resume",
        # Projects with exclusion:
        "projects without unbound",
        # Education:
        "education",
        # Candidate name:
        "who is the candidate?",
        # Contact:
        "contact details",
        # Interests with exclusion:
        "areas of interest without networks",
        # Specific sub-skill:
        "database tools"
    ]
    for q in test_queries:
        test_query(q)
