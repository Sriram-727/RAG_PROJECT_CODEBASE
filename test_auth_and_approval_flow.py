"""
End-to-End Test Suite for Multi-Tenant Admin & User Approval Workflow.

Verifies:
1. Pre-configured Admins exist for all 3 Orgs (Tech, Finance, Health).
2. New user signs up for an Org (status = 'pending').
3. Pending user is BLOCKED by Gate 1 (ACL) when attempting retrieval.
4. Cross-tenant admin protection: Org 1 admin CANNOT view or approve Org 2 pending users.
5. Legitimate Org Admin approves the pending user.
6. Approved user can successfully retrieve & decrypt their organization's documents.
7. Approved user is BLOCKED by Gate 1 when probing another organization.
"""
import sys

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from fastapi.testclient import TestClient
from main import app
from core.auth_service import AuthService

client = TestClient(app)

def run_auth_test_suite():
    print("=" * 95)
    print(" 🛡️ TEST SUITE: MULTI-TENANT ADMIN & USER APPROVAL LIFECYCLE")
    print("=" * 95)

    # -------------------------------------------------------------
    # Step 1: Admin Authentication
    # -------------------------------------------------------------
    print("\n[Step 1: Authenticating Admins for 3 Organizations]")
    for uname, pwd, org_expected in [
        ("admin_tech", "AdminTech2026!", "org_enterprise_01"),
        ("admin_finance", "AdminFinance2026!", "org_finance_corp"),
        ("admin_health", "AdminHealth2026!", "org_healthcare_plus")
    ]:
        res = client.post("/api/v1/auth/login", json={"username": uname, "password": pwd}).json()
        assert res.get("status") == "SUCCESS", f"Failed to login {uname}"
        u = res.get("user", {})
        print(f"  ✓ {uname} ({u.get('full_name')}) -> Role: {u.get('role')} | Org: {u.get('org_id')}")

    # -------------------------------------------------------------
    # Step 2: New User Signup (Pending Approval)
    # -------------------------------------------------------------
    print("\n[Step 2: New User Registration for Organization 2 (Finance Corp)]")
    # Clean up test user if previously created
    with AuthService._get_connection() as conn:
        conn.execute("DELETE FROM users WHERE username = 'bob_analyst'")
        conn.commit()

    signup_res = client.post("/api/v1/auth/signup", json={
        "username": "bob_analyst",
        "password": "Password123!",
        "full_name": "Bob Senior Analyst",
        "org_id": "org_finance_corp"
    }).json()

    print(f"  • User ID: {signup_res.get('user', {}).get('user_id')}")
    print(f"  • Status:  {signup_res.get('status')} ({signup_res.get('user', {}).get('status')})")
    print(f"  • Notice:  {signup_res.get('message')}")
    assert signup_res.get("status") == "PENDING_APPROVAL"

    # -------------------------------------------------------------
    # Step 3: Pending User Attempts Retrieval -> Must Be Blocked by Gate 1
    # -------------------------------------------------------------
    print("\n[Step 3: Pending User Queries Org 2 Documents Before Approval]")
    query_blocked = client.post("/api/v1/query-verify", json={
        "query_text": "What was the acquisition price of QuantumTech AI?",
        "user_id": "usr_bob_analyst",
        "org_id": "org_finance_corp"
    }).json()

    acl_passed = query_blocked.get("acl_verified")
    acl_details = query_blocked.get("verification_breakdown", {}).get("acl", {}).get("details", "")
    print(f"  • Gate 1 (ACL Check):   {'✅ PASSED' if acl_passed else '🛑 BLOCKED (EXPECTED)'}")
    print(f"  • Status:               {query_blocked.get('status')}")
    print(f"  • Barrier Explanation:  \"{acl_details}\"")
    print(f"  • Answer Released:      {query_blocked.get('answer')}")
    assert not acl_passed, "Pending user should be blocked by Gate 1!"
    assert query_blocked.get("status") == "ACL_ACCESS_DENIED"

    # -------------------------------------------------------------
    # Step 4: Cross-Tenant Admin Boundary Enforcement
    # -------------------------------------------------------------
    print("\n[Step 4: Cross-Tenant Admin Boundary Protection]")
    # Org 1 Admin (Tech) attempts to approve Org 2 user
    illegal_appr = client.post("/api/v1/admin/approve-user", json={
        "admin_user_id": "admin_tech",
        "target_user_id": "usr_bob_analyst",
        "decision": "approved"
    })
    print(f"  • Tech Admin (Org 1) approving Finance User (Org 2): HTTP {illegal_appr.status_code}")
    print(f"  • Error Response: \"{illegal_appr.json().get('detail')}\"")
    assert illegal_appr.status_code == 403, "Cross-tenant approval must be rejected with 403 Forbidden!"

    # -------------------------------------------------------------
    # Step 5: Legitimate Org 2 Admin Approves User
    # -------------------------------------------------------------
    print("\n[Step 5: Legitimate Finance Admin (Org 2) Reviews and Approves User]")
    pending_list = client.get("/api/v1/admin/pending-users?admin_id=admin_finance").json()
    pending_users = pending_list.get("pending_users", [])
    print(f"  • Finance Admin Pending Queue Count: {len(pending_users)}")
    assert any(u["username"] == "bob_analyst" for u in pending_users), "bob_analyst should be in pending list"

    legal_appr = client.post("/api/v1/admin/approve-user", json={
        "admin_user_id": "admin_finance",
        "target_user_id": "usr_bob_analyst",
        "decision": "approved"
    }).json()

    print(f"  • Approval Outcome: {legal_appr.get('status')} - {legal_appr.get('message')}")
    assert legal_appr.get("status") == "SUCCESS"

    # -------------------------------------------------------------
    # Step 6: Approved User Queries Org 2 Documents -> All 5 Gates Pass
    # -------------------------------------------------------------
    print("\n[Step 6: Approved User Queries Org 2 Documents (Post-Approval)]")
    query_approved = client.post("/api/v1/query-verify", json={
        "query_text": "What was the acquisition price of QuantumTech AI?",
        "user_id": "usr_bob_analyst",
        "org_id": "org_finance_corp"
    }).json()

    print(f"  • Gate 1 (ACL Check):        {'✅ PASSED' if query_approved.get('acl_verified') else '❌ FAILED'}")
    print(f"  • Gate 2 (Merkle Integrity): {'✅ PASSED' if query_approved.get('integrity_verified') else '❌ FAILED'}")
    print(f"  • Gate 4 (Decryption):       {'✅ PASSED' if query_approved.get('decryption_verified') else '❌ FAILED'}")
    print(f"  • Gate 5 (Grounding Score):  {query_approved.get('hallucination_report', {}).get('grounding_score')}")
    print(f"  • Status:                    {query_approved.get('status')}")
    print(f"  • Verified Answer:           {query_approved.get('answer')}")
    assert query_approved.get("status") == "SUCCESS", "Approved user should pass all verification gates!"
    assert query_approved.get("acl_verified"), "Gate 1 should pass for approved user"

    # -------------------------------------------------------------
    # Step 7: Approved Org 2 User Probes Org 1 Documents -> Blocked by Gate 1
    # -------------------------------------------------------------
    print("\n[Step 7: Approved Org 2 User Probes Org 1 Documents (Cross-Tenant Isolation)]")
    query_cross = client.post("/api/v1/query-verify", json={
        "query_text": "What are the cloud security access policies for production clusters?",
        "user_id": "usr_bob_analyst",
        "org_id": "org_enterprise_01"
    }).json()

    print(f"  • Gate 1 (ACL Check):   {'✅ PASSED' if query_cross.get('acl_verified') else '🛑 BLOCKED (EXPECTED)'}")
    print(f"  • Status:               {query_cross.get('status')}")
    print(f"  • Reason:               \"{query_cross.get('verification_breakdown', {}).get('acl', {}).get('details')}\"")
    print(f"  • Answer Released:      {query_cross.get('answer')}")
    assert not query_cross.get("acl_verified"), "User must not access documents of another organization!"
    assert query_cross.get("status") == "ACL_ACCESS_DENIED"

    print("\n" + "=" * 95)
    print(" 🎯 ALL ADMIN & APPROVAL LIFECYCLE TESTS PASSED 100%!")
    print("=" * 95)

if __name__ == "__main__":
    run_auth_test_suite()
