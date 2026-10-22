"""
Script to populate Qdrant collection with rich enterprise sample datasets.
Creates multiple distinct semantic clusters:
1. Enterprise Cloud Security Policy & Compliance
2. Senior Software Engineer Resume & Project Portfolio
3. Kubernetes Microservices Architecture Specification
4. AI / RAG Evaluation and Hallucination Framework
"""
import sys
import time
import requests

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SAMPLE_DOCUMENTS = [
    {
        "doc_id": "doc_cloud_security_policy",
        "filename": "cloud_security_policy_2026.pdf",
        "user_id": "usr_admin_01",
        "org_id": "org_enterprise_01",
        "text": (
            "ENTERPRISE CLOUD SECURITY DIRECTIVE & COMPLIANCE STANDARD (2026)\n\n"
            "Section 1: Zero-Trust Identity & Access Management.\n"
            "All engineering staff accessing production Kubernetes clusters, AWS infrastructure, and internal databases "
            "must authenticate using hardware FIDO2/WebAuthn security keys. Password-only authentication is strictly prohibited. "
            "Multi-factor authentication (MFA) must be enforced with push notifications and biometric verification.\n\n"
            "Section 2: VPN Credential Rotation & Session Lifetime.\n"
            "All VPN security credentials, SSH keys, and TLS certificates must be automatically rotated every 30 days. "
            "Inactive administrative sessions and remote desktop connections are forcibly terminated after 15 minutes of idle time.\n\n"
            "Section 3: Cryptographic Storage & Merkle Verification.\n"
            "All enterprise data stored in cloud vector databases must enforce AES-256-GCM authenticated encryption at rest. "
            "Data retrieval pipelines must provide Two-Tier Merkle Mountain Range (MMR) proofs to verify zero tampering before release.\n\n"
            "Section 4: Audit Logs & Compliance Retention.\n"
            "Compliance audit logs and system telemetry must be retained for 7 years in immutable Write-Once-Read-Many (WORM) storage "
            "to satisfy SOC2 Type II, ISO 27001, and HIPAA compliance mandates."
        )
    },
    {
        "doc_id": "doc_john_doe_resume",
        "filename": "john_doe_senior_engineer_resume.pdf",
        "user_id": "usr_admin_01",
        "org_id": "org_enterprise_01",
        "text": (
            "JOHN DOE - SENIOR FULL-STACK & SYSTEMS ENGINEER\n\n"
            "Career Objective:\n"
            "Innovative software engineer with 6+ years of experience designing high-throughput distributed microservices, "
            "scalable vector databases, and zero-trust verification pipelines.\n\n"
            "Core Technical Skills:\n"
            "Languages: Java (Multithreading, Concurrency, CompletableFuture, Virtual Threads), Python, TypeScript, Go, SQL.\n"
            "Databases & Search: Qdrant Vector DB, PostgreSQL, Redis, Elasticsearch, Apache Kafka.\n"
            "Frameworks: Spring Boot, FastAPI, React, Node.js, Docker, Kubernetes, AWS.\n\n"
            "Featured Engineering Projects:\n"
            "1. Real Estate Management System: Built an enterprise property management portal using Spring Boot microservices, "
            "Kafka event streaming, and React. Implemented property listings, automated booking pipelines, and tenant payment gateways.\n"
            "2. Zero-Trust Verifiable RAG Engine: Architected an asymmetric RAG system incorporating Qdrant vector retrieval, "
            "Two-Tier Merkle Mountain Range (MMR) cryptographic proofs, and IVF Voronoi boundary validation.\n"
            "3. Distributed Lock Manager: Engineered a fault-tolerant distributed locking coordinator implementing Raft consensus "
            "and Redis Redlock algorithm, handling 100k lock acquisitions per second.\n\n"
            "Education:\n"
            "Bachelor of Technology in Computer Science & Engineering. First Class with Distinction."
        )
    },
    {
        "doc_id": "doc_k8s_architecture_spec",
        "filename": "k8s_microservices_architecture_spec.txt",
        "user_id": "usr_admin_01",
        "org_id": "org_enterprise_01",
        "text": (
            "PRODUCTION KUBERNETES & CLOUD ARCHITECTURE SPECIFICATION\n\n"
            "1. Cluster Topology:\n"
            "The production environment is deployed on Amazon EKS spanning 3 AWS Availability Zones (us-east-1a, 1b, 1c). "
            "Worker nodes run across managed node groups with m6i.xlarge instances utilizing Bottlerocket OS for security hardening.\n\n"
            "2. Traffic Ingress & Service Mesh:\n"
            "External traffic enters via AWS Application Load Balancers (ALB) into Traefik Ingress Controller. "
            "Internal service-to-service communication is governed by Istio Service Mesh with strict mutual TLS (mTLS) and "
            "automated Envoy sidecar proxy injection.\n\n"
            "3. CI/CD & GitOps Deployment:\n"
            "Automated continuous delivery is managed via ArgoCD following GitOps principles. Application container images "
            "are scanned with Trivy before deployment to eliminate CVE vulnerabilities.\n\n"
            "4. Observability & Telemetry:\n"
            "Distributed tracing is captured via OpenTelemetry collectors and exported to Jaeger. Metrics are scraped by Prometheus "
            "and visualized in unified Grafana operations dashboards with automated Slack paging alerts."
        )
    },
    {
        "doc_id": "doc_rag_eval_framework",
        "filename": "rag_evaluation_framework_guide.pdf",
        "user_id": "usr_admin_01",
        "org_id": "org_enterprise_01",
        "text": (
            "RETRIEVAL-AUGMENTED GENERATION (RAG) VERIFICATION & EVALUATION GUIDE\n\n"
            "Overview:\n"
            "Modern RAG systems suffer from vector drift, data tampering, and model hallucinations. "
            "This guide defines the 5-Gate Verifiable Architecture for high-assurance enterprise RAG.\n\n"
            "The 5 Verification Gates:\n"
            "Gate 1 - Access Control List (ACL): Verifies user role and organizational ownership before retrieval.\n"
            "Gate 2 - Cryptographic Integrity: Validates Two-Tier Merkle Mountain Range (MMR) inclusion proofs against Cloud Root.\n"
            "Gate 3 - Geometric Completeness: Probes Voronoi clusters to ensure the entire semantic candidate space is covered.\n"
            "Gate 4 - Authenticated Decryption: Decrypts AES-256-GCM ciphertext on the client boundary, rejecting forged tags.\n"
            "Gate 5 - Semantic Grounding: Measures non-stopword token overlap and sentence-level entailment between corpus and LLM response.\n\n"
            "Hybrid Search Strategy:\n"
            "Combining dense embeddings (all-MiniLM-L6-v2) with BM25 keyword matching via Reciprocal Rank Fusion (RRF) yields "
            "optimal recall for technical acronyms and exact keyword lookups."
        )
    }
]

def seed_qdrant():
    print("=" * 80)
    print(" 🚀 POPULATING QDRANT WITH REALISTIC ENTERPRISE SAMPLE DOCUMENTS")
    print("=" * 80)

    api_url = "http://127.0.0.1:8000/api/v1/ingest-text"
    total_points = 0

    for i, doc in enumerate(SAMPLE_DOCUMENTS, 1):
        print(f"\n[{i}/{len(SAMPLE_DOCUMENTS)}] Ingesting: {doc['doc_id']} ({doc['filename']})")
        payload = {
            "text": doc["text"],
            "doc_id": doc["doc_id"],
            "user_id": doc["user_id"],
            "org_id": doc["org_id"],
            "chunk_size": 400,
            "overlap": 40,
            "filename": doc["filename"]
        }

        try:
            t0 = time.perf_counter()
            res = requests.post(api_url, json=payload, timeout=60)
            elapsed = (time.perf_counter() - t0) * 1000

            if res.status_code == 200:
                data = res.json()
                total_points += data["qdrant_points_upserted"]
                print(f"   ✓ Success ({elapsed:.1f}ms) | {data['total_chunks']} chunks | {data['qdrant_points_upserted']} points in Qdrant")
                print(f"   • Cloud MMR Root: {data['cloud_enterprise_root'][:24]}...")
            else:
                print(f"   ❌ Failed (Status {res.status_code}): {res.text}")
        except Exception as e:
            print(f"   ❌ Ingestion error: {e}")

    print("\n" + "=" * 80)
    print(f" 🎉 COLLECTION POPULATION COMPLETE! Total Points Upserted: {total_points}")
    print("=" * 80)

if __name__ == "__main__":
    seed_qdrant()
