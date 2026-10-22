import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import json
import time
import subprocess
import urllib.request
import urllib.error

API_HOST = "127.0.0.1"
API_PORT = 8000
BASE_URL = f"http://{API_HOST}:{API_PORT}"
HEALTH_URL = f"{BASE_URL}/docs"
QUERY_URL = f"{BASE_URL}/api/v1/query-verify"

def is_server_ready(url: str, timeout: float = 1.0) -> bool:
    for check_path in ["/health", "/docs", "/api/v1/openapi.json"]:
        try:
            req = urllib.request.Request(f"{BASE_URL}{check_path}")
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    return True
        except Exception:
            continue
    return False

def wait_for_server(process: subprocess.Popen, max_wait: float = 60.0, poll_interval: float = 0.5) -> bool:
    start_time = time.time()
    print("Waiting for server to initialize...", end="", flush=True)
    while time.time() - start_time < max_wait:
        # Check if the process died prematurely
        if process.poll() is not None:
            print("\n[ERROR] Server process terminated unexpectedly!")
            if os.path.exists("uvicorn_test.log"):
                with open("uvicorn_test.log", "r", encoding="utf-8", errors="replace") as f:
                    print("Server output:\n", f.read())
            return False

        if is_server_ready(HEALTH_URL):
            print(f" Ready! ({time.time() - start_time:.1f}s)")
            return True

        print(".", end="", flush=True)
        time.sleep(poll_interval)

    print(f"\n[ERROR] Timed out after {max_wait}s waiting for server to start.")
    if os.path.exists("uvicorn_test.log"):
        with open("uvicorn_test.log", "r", encoding="utf-8", errors="replace") as f:
            print("Server output:\n", f.read()[-2000:])
    return False

def main():
    server_process = None
    log_file = None
    already_running = is_server_ready(HEALTH_URL)

    if already_running:
        print(f"Detected existing API Server running at {BASE_URL}. Using it directly.")
    else:
        print(f"Starting API Server using '{sys.executable}' on {API_HOST}:{API_PORT}...")
        log_file = open("uvicorn_test.log", "w", encoding="utf-8")
        env = os.environ.copy()
        env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        server_process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "main:app", "--host", API_HOST, "--port", str(API_PORT)],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env
        )

        if not wait_for_server(server_process):
            if server_process and server_process.poll() is None:
                server_process.terminate()
                server_process.wait()
            if log_file:
                log_file.close()
            sys.exit(1)

    try:
        print("\n--- 1. Testing LangChain Direct Endpoint (/api/v1/retrieve-langchain) ---")
        lc_payload = {
            "query_text": "How often must VPN credentials be rotated?",
            "org_id": "org_enterprise_01"
        }
        lc_req = urllib.request.Request(
            f"{BASE_URL}/api/v1/retrieve-langchain",
            data=json.dumps(lc_payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(lc_req, timeout=60) as lc_resp:
                print(f"Status Code: {lc_resp.status}")
                lc_json = json.loads(lc_resp.read().decode("utf-8"))
                print(f"Retrieved Chunks: {lc_json.get('retrieved_chunks_count')}")
                for i, chk in enumerate(lc_json.get('top_chunks', []), 1):
                    clean_chk = chk[:80].replace("\n", " ").encode("ascii", "replace").decode("ascii")
                    print(f"  [Chunk {i}]: {clean_chk}")
        except Exception as lce:
            print(f"LangChain endpoint note: {lce}")

        print("\n--- 2. Sending Full Verification Query (/api/v1/query-verify) ---")
        payload = {
            "query_text": "How often must VPN credentials be rotated?",
            "user_id": "usr_admin_01",
            "org_id": "org_enterprise_01",
            "simulate_missing_cluster": False,
            "simulate_integrity_failure": False
        }

        req = urllib.request.Request(
            QUERY_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                print(f"Status Code: {response.status}")
                resp_json = json.loads(response.read().decode("utf-8"))
                print(f"Status: {resp_json.get('status')}")
                ans = str(resp_json.get('answer', '')).encode('ascii', 'replace').decode('ascii')
                print(f"Answer: {ans}")
                print(f"LangChain Pipeline Active: {resp_json.get('query_enhancements', {}).get('langchain_pipeline', {}).get('active')}")
                print(f"Retrieved Documents: {resp_json.get('retrieved_documents_count')}")
        except urllib.error.HTTPError as e:
            print(f"HTTP Error: {e.code}")
            print(e.read().decode("utf-8", errors="replace"))
        except urllib.error.URLError as e:
            print(f"URL Error: {e.reason}")
            if server_process and server_process.poll() is not None:
                _, stderr = server_process.communicate()
                print("Server Stderr:\n", stderr.decode("utf-8", errors="replace"))

    finally:
        if server_process:
            print("\nShutting down server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
                server_process.wait()
            print("Done.")
        if log_file:
            log_file.close()

if __name__ == "__main__":
    main()
