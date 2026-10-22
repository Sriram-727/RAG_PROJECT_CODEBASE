import asyncio
from main import app, QueryPayloadRequest, query_and_verify_endpoint

async def run_test():
    for q in ["objectives", "list only the project names in the resume", "what are the projects?"]:
        print(f"\n================ QUERY: {q} ================")
        payload = QueryPayloadRequest(
            org_id="org_enterprise_01",
            user_id="usr_admin_01",
            query_text=q,
            simulate_missing_cluster=False,
            simulate_integrity_failure=False
        )
        try:
            res = await query_and_verify_endpoint(payload)
            print(f"STATUS: {res.status}")
            safe_ans = (res.answer or "").encode("ascii", errors="replace").decode("ascii")
            print(f"ANSWER:\n{safe_ans}")
            print(f"RETRIEVED CHUNKS: {res.retrieved_chunks_count}")
            if res.hallucination_report:
                print(f"HALLUCINATION: {res.hallucination_report}")
        except Exception as e:
            print(f"ERROR: {e}")

asyncio.run(run_test())
