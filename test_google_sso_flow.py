"""
Automated Validation Suite for Google Single Sign-On (SSO) Flow.
Tests JIT provisioning, Zero-Trust Admin Approval, Gate 1 ACL blocking, and post-approval retrieval.
"""
import sys
import unittest
from fastapi.testclient import TestClient

from main import app
from core.auth_service import AuthService

class TestGoogleSSOFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        AuthService.init_db()
        with AuthService._get_connection() as conn:
            conn.execute("DELETE FROM users WHERE email = 'sarah.connor.tech@gmail.com' OR username = 'google_sarah_connor_tech'")
        cls.client = TestClient(app)

    def test_01_google_sso_jit_provisioning_pending(self):
        """New Google user signs in -> account is JIT-provisioned in 'pending' status."""
        payload = {
            "email": "sarah.connor.tech@gmail.com",
            "full_name": "Sarah Connor",
            "google_id": "google_sub_987654321",
            "target_org_id": "org_enterprise_01"
        }
        res = self.client.post("/api/v1/auth/google", json=payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "PENDING_APPROVAL")
        self.assertEqual(data["user"]["auth_provider"], "google")
        self.assertEqual(data["user"]["org_id"], "org_enterprise_01")
        self.assertEqual(data["user"]["status"], "pending")
        print("\n  [+] Step 1: Google SSO JIT Provisioning -> User created in 'pending' status.")

    def test_02_pending_google_user_blocked_at_gate1(self):
        """Unapproved Google user attempts to retrieve documents -> Blocked by Gate 1 ACL."""
        query_payload = {
            "query_text": "How often must VPN credentials be rotated?",
            "user_id": "google_sarah_connor_tech",
            "org_id": "org_enterprise_01"
        }
        res = self.client.post("/api/v1/query-verify", json=query_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ACL_ACCESS_DENIED")
        self.assertFalse(data["acl_verified"])
        self.assertIn("pending approval", data["verification_breakdown"]["acl"]["details"])
        self.assertIsNone(data["answer"])
        print("  [+] Step 2: Unapproved Google user query correctly blocked at Gate 1 ACL.")

    def test_03_admin_approves_google_user(self):
        """Organization Admin reviews pending queue and approves Google SSO account."""
        # 1. Fetch pending list
        pending_res = self.client.get("/api/v1/admin/pending-users?admin_id=admin_tech")
        self.assertEqual(pending_res.status_code, 200)
        pending_users = pending_res.json()["pending_users"]
        target = next((u for u in pending_users if u["username"] == "google_sarah_connor_tech"), None)
        self.assertIsNotNone(target)
        self.assertEqual(target.get("auth_provider"), "google")

        # 2. Approve user
        approval_payload = {
            "admin_id": "admin_tech",
            "target_username": "google_sarah_connor_tech",
            "status": "approved"
        }
        app_res = self.client.post("/api/v1/admin/approve-user", json=approval_payload)
        self.assertEqual(app_res.status_code, 200)
        self.assertEqual(app_res.json()["status"], "SUCCESS")
        print("  [+] Step 3: Tech Admin successfully approved Google SSO user in Approvals Portal.")

    def test_04_approved_google_user_retrieval_success(self):
        """Post-approval: Google user logs in via SSO and verifies query successfully across all 5 gates."""
        # 1. Re-login via Google SSO
        login_res = self.client.post("/api/v1/auth/google", json={
            "email": "sarah.connor.tech@gmail.com",
            "full_name": "Sarah Connor",
            "google_id": "google_sub_987654321",
            "target_org_id": "org_enterprise_01"
        })
        self.assertEqual(login_res.status_code, 200)
        self.assertEqual(login_res.json()["status"], "SUCCESS")
        self.assertEqual(login_res.json()["user"]["status"], "approved")

        # 2. Query documents
        query_payload = {
            "query_text": "How often must VPN credentials be rotated?",
            "user_id": "google_sarah_connor_tech",
            "org_id": "org_enterprise_01"
        }
        res = self.client.post("/api/v1/query-verify", json=query_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "SUCCESS")
        self.assertTrue(data["acl_verified"])
        self.assertTrue(data["integrity_verified"])
        self.assertTrue(data["decryption_verified"])
        self.assertIsNotNone(data["answer"])
        print(f"  [+] Step 4: Approved Google user passed all 5 gates! Answer: '{data['answer'][:60]}...'")

    def test_05_google_user_cross_tenant_isolation(self):
        """Approved Org 1 Google user cannot query Org 2 collection."""
        probe_payload = {
            "query_text": "What was the acquisition price of QuantumTech AI?",
            "user_id": "google_sarah_connor_tech",
            "org_id": "org_finance_corp"
        }
        res = self.client.post("/api/v1/query-verify", json=probe_payload)
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ACL_ACCESS_DENIED")
        self.assertFalse(data["acl_verified"])
        self.assertIn("Cross-tenant isolation block", data["verification_breakdown"]["acl"]["details"])
        self.assertIsNone(data["answer"])
        print("  [+] Step 5: Cross-tenant query by Google user correctly blocked at Gate 1.")

if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print("=" * 80)
    print(" [*] RUNNING GOOGLE SSO & JIT PROVISIONING TEST SUITE")
    print("=" * 80)
    suite = unittest.TestLoader().loadTestsFromTestCase(TestGoogleSSOFlow)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n" + "=" * 80)
        print(" [SUCCESS] ALL GOOGLE SSO VALIDATION TESTS PASSED 100%!")
        print("=" * 80)
        sys.exit(0)
    else:
        sys.exit(1)
