import asyncio
from main import QueryPayloadRequest, query_and_verify_endpoint

async def run_suite():
    queries = [
        ("what are the technical skills?", False, False),
        ("what is the education?", False, False),
        ("tell me about areas of interest", False, False),
        ("candidate name", False, False),
        ("contact information", False, False),
        ("list only the project names in the resume", True, False), # sim missing cluster
        ("objectives", False, True), # sim integrity tampering
    ]

    for q, sim_cluster, sim_integrity in queries:
        print(f"\n>>> QUERY: {q} | sim_cluster={sim_cluster} | sim_integrity={sim_integrity}")
        payload = QueryPayloadRequest(
            org_id="org_enterprise_01",
            user_id="usr_admin_01",
            query_text=q,
            simulate_missing_cluster=sim_cluster,
            simulate_integrity_failure=sim_integrity
        )
        res = await query_and_verify_endpoint(payload)
        safe_ans = (res.answer or "").encode("ascii", errors="replace").decode("ascii")
        print(f"STATUS: {res.status}")
        print(f"ANSWER:\n{safe_ans}")
        print(f"INTEGRITY_VERIFIED: {res.integrity_verified} | BOUNDARY_COMPLETE: {res.ivf_boundary_verification.is_complete}")

asyncio.run(run_suite())
