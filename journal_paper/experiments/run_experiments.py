"""
Experiments for the IJCC paper. Runs the project's own modules (crypto/, mmr/,
verification/boundary_validation.py, core/auth_service.py) -- nothing is simulated
except where stated. Writes results.json next to this script.

Run from the repository root:
    python journal_paper/experiments/run_experiments.py [path/to/corpus.txt]

Environment notes (recorded in results.json):
  * scikit-learn is replaced by a small numpy Lloyd k-means with the same
    interface so the IVF validator can be built without the dependency.
  * If sentence-transformers is installed, the project's all-MiniLM-L6-v2 model is
    used; otherwise the project's own hashed bag-of-words fallback embedder is used.
"""
import ast
import base64
import json
import os
import platform
import re
import statistics
import sys
import tempfile
import time
import types

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---- stand-in for sklearn.cluster.KMeans (only if sklearn is missing) ----------
try:
    import sklearn  # noqa: F401
    KMEANS_IMPL = "scikit-learn"
except Exception:
    KMEANS_IMPL = "numpy Lloyd k-means (kmeans++ init, 10 restarts, 300 iterations)"

    class KMeans:
        def __init__(self, n_clusters, random_state=42, n_init=10, max_iter=300):
            self.k, self.seed, self.n_init, self.max_iter = n_clusters, random_state, n_init, max_iter
            self.cluster_centers_ = None

        def fit_predict(self, X):
            best = None
            for r in range(self.n_init):
                rng = np.random.default_rng(self.seed + r)
                cent = [X[rng.integers(len(X))]]
                for _ in range(1, self.k):
                    d2 = np.min(((X[:, None, :] - np.array(cent)[None]) ** 2).sum(-1), axis=1)
                    cent.append(X[rng.choice(len(X), p=d2 / d2.sum())])
                cent = np.array(cent)
                for _ in range(self.max_iter):
                    lab = ((X[:, None, :] - cent[None]) ** 2).sum(-1).argmin(1)
                    new = np.array([X[lab == c].mean(0) if (lab == c).any() else cent[c] for c in range(self.k)])
                    if np.allclose(new, cent):
                        break
                    cent = new
                inertia = ((X - cent[lab]) ** 2).sum()
                if best is None or inertia < best[0]:
                    best = (inertia, cent, lab)
            self.cluster_centers_ = best[1]
            return best[2]

    sk = types.ModuleType("sklearn")
    skc = types.ModuleType("sklearn.cluster")
    skc.KMeans = KMeans
    sk.cluster = skc
    sys.modules["sklearn"], sys.modules["sklearn.cluster"] = sk, skc

# qdrant-client is only needed by database/qdrant_client.py (network layer); stub it for offline runs
try:
    import qdrant_client  # noqa: F401
except Exception:
    from unittest.mock import MagicMock
    for _m in ("qdrant_client", "qdrant_client.models", "qdrant_client.http", "qdrant_client.http.models",
               "qdrant_client.http.exceptions"):
        sys.modules[_m] = MagicMock()

try:
    import pydantic_settings  # noqa: F401
except Exception:
    import pydantic
    _ps = types.ModuleType("pydantic_settings")
    _ps.BaseSettings = pydantic.BaseModel
    _ps.SettingsConfigDict = dict
    sys.modules["pydantic_settings"] = _ps

from crypto.encryption import encrypt_aes_gcm, decrypt_aes_gcm  # noqa: E402
from crypto.hashing import sha256_hex  # noqa: E402
from database.schemas import ChunkMetadata, EncryptedChunkRecord  # noqa: E402
from mmr.cloud_mmr import CloudMMR  # noqa: E402
from mmr.proof import TwoTierInclusionProof, verify_two_tier_proof  # noqa: E402
from verification.boundary_validation import IVFBoundaryValidator  # noqa: E402
from core.embedding_service import EmbeddingService  # noqa: E402

# ---- chunker taken verbatim from core/document_ingestion.py (AST-extracted so the
# ---- qdrant/pydantic-settings imports of that module are not needed) -------------
src = open(os.path.join(ROOT, "core", "document_ingestion.py"), encoding="utf-8").read()
tree = ast.parse(src)
wanted = {"SECTION_PATTERNS", "chunk_text", "_sliding_window_chunks"}
import logging
ns = {"re": re, "List": list, "logger": logging.getLogger("chunker")}
for node in tree.body:
    name = getattr(node, "name", None)
    if name is None and isinstance(node, ast.Assign):
        name = node.targets[0].id
    if name in wanted:
        exec(compile(ast.Module([node], []), "document_ingestion", "exec"), ns)
chunk_text = ns["chunk_text"]

KEY = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
rng = np.random.default_rng(7)
results = {
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "kmeans": KMEANS_IMPL,
        "embedder": "all-MiniLM-L6-v2" if EmbeddingService.get_model() != "MOCK_MODEL"
                    else "hashed bag-of-words fallback (EmbeddingService._mock_embed)",
    }
}


def make_records(n_chunks, n_users, org="org_bench"):
    recs = []
    for i in range(n_chunks):
        user = f"user_{i % n_users:02d}"
        text = f"synthetic enterprise chunk number {i} " + "x" * 450
        enc = encrypt_aes_gcm(text, KEY)
        recs.append(EncryptedChunkRecord(
            point_id=str(i), chunk_id=f"chk_{user}_doc_{i // n_users}_{i}", doc_id=f"doc_{i // n_users}",
            org_id=org, user_id=user, chunk_index=i, encrypted_content=enc.ciphertext_b64,
            nonce=enc.nonce_b64, chunk_hash=sha256_hex(text), metadata=ChunkMetadata()))
    return recs


def med_ms(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        ts.append((time.perf_counter() - t) * 1000)
    return statistics.median(ts), float(np.percentile(ts, 95))


# =========================== 1. micro-benchmarks =================================
bench = []
for n in (100, 1000, 10000):
    recs = make_records(n, 10)
    t0 = time.perf_counter()
    cmmr = CloudMMR(org_id="org_bench").build_from_chunks(recs)
    build_s = time.perf_counter() - t0
    sample = [recs[int(i)] for i in rng.choice(n, size=min(25, n), replace=False)]
    proofs = [cmmr.get_two_tier_proof(r.user_id, r.chunk_id) for r in sample]
    k = [0]

    def vp():
        verify_two_tier_proof(proofs[k[0] % len(proofs)])
        k[0] += 1
    m, p = med_ms(vp, 400)
    path = [len(p.user_proof.sibling_path) + len(p.cloud_proof.cloud_sibling_path) for p in proofs]
    bench.append({"chunks": n, "mmr_build_s": round(build_s, 3), "verify_median_ms": round(m, 4),
                  "verify_p95_ms": round(p, 4), "mean_sibling_hashes": round(float(np.mean(path)), 2)})
results["mmr_proof_verification"] = bench

enc = encrypt_aes_gcm("y" * 500, KEY)
m, p = med_ms(lambda: decrypt_aes_gcm(enc.ciphertext_b64, enc.nonce_b64, KEY), 2000)
results["aes_gcm_decrypt_500B"] = {"median_ms": round(m, 4), "p95_ms": round(p, 4)}

# =========================== 2. corpus, index, ablation ===========================
corpus_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "corpus.txt")
text = open(corpus_path, encoding="utf-8").read()
chunks = chunk_text(text, chunk_size=500, overlap=50)
X = EmbeddingService.embed_batch(chunks)
N = len(chunks)
results["corpus"] = {"source": os.path.basename(corpus_path), "chars": len(text), "chunks": N,
                     "chunk_size": 500, "overlap": 50, "dim": int(X.shape[1])}

# queries: a sentence from a chunk; ground truth = exact top-5 among the *other* chunks
sents = []
for ci, c in enumerate(chunks):
    for s in re.split(r"(?<=[.!?])\s+", c):
        if len(s.split()) >= 8:
            sents.append((ci, s))
sel = rng.choice(len(sents), size=min(200, len(sents)), replace=False)
queries = [sents[i] for i in sel]
Q = EmbeddingService.embed_batch([q for _, q in queries])
sims = Q @ X.T
truth = []
for qi, (src_c, _) in enumerate(queries):
    s = sims[qi].copy()
    s[src_c] = -9
    truth.append(set(np.argsort(-s)[:5].tolist()))

tmp = tempfile.mkdtemp()
ablation = []
for n_lists in (6, 16):
    val = IVFBoundaryValidator(1.5, os.path.join(tmp, f"c{n_lists}.npy"), os.path.join(tmp, f"r{n_lists}.npy"))
    labels = val.build_ivf_index(X.astype(np.float32), list(range(N)), n_lists=n_lists)
    labels = np.asarray(labels)
    for mf in (0.1, 0.25, 0.5, 1.0, 1.2, 1.5, 2.0):
        val.margin_factor = mf
        rec, cells, scanned, multi, omit_det, omit_tot = [], [], [], 0, 0, 0
        for qi in range(len(queries)):
            req, _ = val.required_clusters(Q[qi])
            probed = np.isin(labels, req)
            rec.append(len(truth[qi] & set(np.where(probed)[0].tolist())) / 5.0)
            cells.append(len(req))
            scanned.append(probed.mean())
            if len(req) > 1:
                multi += 1
            for drop in req:                      # lazy cloud omits one required cell
                reported = [c for c in req if c != drop]
                omit_tot += 1
                omit_det += (not val.verify_query_boundary(Q[qi], reported)["is_complete"])
        ablation.append({"n_lists": n_lists, "margin_factor": mf,
                         "recall_at_5": round(float(np.mean(rec)), 4),
                         "mean_cells_probed": round(float(np.mean(cells)), 2),
                         "mean_fraction_scanned": round(float(np.mean(scanned)), 4),
                         "queries_with_multiple_required_cells": round(multi / len(queries), 4),
                         "omission_attempts": omit_tot,
                         "omission_detected": omit_det})
results["dbv_ablation"] = ablation
results["n_queries"] = len(queries)

# DBV latency (6 lists, 1.5)
val = IVFBoundaryValidator(1.5, os.path.join(tmp, "c6.npy"), os.path.join(tmp, "r6.npy"))
m, p = med_ms(lambda: val.verify_query_boundary(Q[0], val.required_clusters(Q[0])[0]), 2000)
results["dbv_check_latency"] = {"median_ms": round(m, 4), "p95_ms": round(p, 4)}

# =========================== 3. attack tests on the real code ======================
user_recs = make_records(60, 6, org="org_t1")
cm = CloudMMR(org_id="org_t1").build_from_chunks(user_recs)
r0 = user_recs[7]
good = cm.get_two_tier_proof(r0.user_id, r0.chunk_id)
ct = bytearray(base64.b64decode(r0.encrypted_content))


def aead_ok(ciphertext_b64, nonce_b64, aad=None):
    try:
        decrypt_aes_gcm(ciphertext_b64, nonce_b64, KEY, aad)
        return True
    except Exception:
        return False


attacks = {}
# A1 ciphertext bit-flip
flipped = bytearray(ct)
flipped[10] ^= 1
attacks["A1_ciphertext_bitflip"] = {
    # the MMR leaf commits to the plaintext hash, never to the ciphertext, so Gate 2 cannot see this attack
    "mmr_rejects": False,
    "aead_rejects": not aead_ok(base64.b64encode(bytes(flipped)).decode(), r0.nonce)}
# A2 tampered chunk_hash in the proof
bad = good.model_copy(deep=True)
bad.chunk_hash = sha256_hex("forged plaintext")
attacks["A2_forged_chunk_hash"] = {"mmr_rejects": not verify_two_tier_proof(bad)}
# A3 modified sibling hash in the proof path
bad = good.model_copy(deep=True)
st = bad.user_proof.sibling_path
if st:
    st[0].sibling_hash = sha256_hex("x")
    attacks["A3_modified_sibling_hash"] = {"mmr_rejects": not verify_two_tier_proof(bad)}
else:
    bad.cloud_proof.cloud_sibling_path[0].sibling_hash = sha256_hex("x")
    attacks["A3_modified_sibling_hash"] = {"mmr_rejects": not verify_two_tier_proof(bad)}
# A4 swap ciphertext between two chunks (the cloud keeps both proofs and metadata intact)
r1 = user_recs[19]
attacks["A4_ciphertext_swap_between_chunks"] = {
    "aead_rejects": not aead_ok(r1.encrypted_content, r1.nonce)}
attacks["A4_ciphertext_swap_between_chunks"]["note"] = (
    "AES-GCM is called with associated_data=None and the plaintext hash is not re-checked "
    "after decryption, so a swapped (ciphertext, nonce) pair is accepted")
# A5 re-keyed forgery: attacker builds its own MMR and attaches its own root to the forged chunk
forged_text = "attacker supplied content"
enc_f = encrypt_aes_gcm(forged_text, KEY)
frec = EncryptedChunkRecord(point_id="f", chunk_id="chk_f", doc_id="doc_f", org_id="org_t1", user_id="user_00",
                            chunk_index=0, encrypted_content=enc_f.ciphertext_b64, nonce=enc_f.nonce_b64,
                            chunk_hash=sha256_hex(forged_text), metadata=ChunkMetadata())
fmmr = CloudMMR(org_id="org_t1").build_from_chunks([frec])
fproof = fmmr.get_two_tier_proof("user_00", "chk_f")
attacks["A5_self_consistent_forgery"] = {
    "accepted_with_proof_embedded_root": verify_two_tier_proof(fproof),
    "accepted_against_trusted_root": verify_two_tier_proof(fproof, expected_cloud_root=cm.cloud_root),
    "note": "the query path passes expected_cloud_root=proof.expected_cloud_root, i.e. the root carried by the response itself"}
# A6 omission is covered in the ablation table; A7 cross-tenant via AuthService
import core.auth_service as A  # noqa: E402
A.DB_DIR = tempfile.mkdtemp()
A.DB_PATH = os.path.join(A.DB_DIR, "auth.db")
A.AuthService.init_db()
A.AuthService.create_user_direct("u_a", "u_a", "pw-aaaaaa", "A", "org_enterprise_01", status="approved")
A.AuthService.create_user_direct("u_b", "u_b", "pw-bbbbbb", "B", "org_finance_corp", status="approved")
A.AuthService.create_user_direct("u_p", "u_p", "pw-pppppp", "P", "org_enterprise_01", status="pending")
attacks["A7_cross_tenant_access"] = {
    "own_org_allowed": A.AuthService.is_user_authorized_for_org("u_a", "org_enterprise_01")[0],
    "other_org_blocked": not A.AuthService.is_user_authorized_for_org("u_a", "org_finance_corp")[0],
    "other_org_blocked_2": not A.AuthService.is_user_authorized_for_org("u_b", "org_enterprise_01")[0],
    "pending_user_blocked": not A.AuthService.is_user_authorized_for_org("u_p", "org_enterprise_01")[0],
    "unknown_user_blocked": not A.AuthService.is_user_authorized_for_org("ghost", "org_enterprise_01")[0]}
results["attack_tests"] = attacks

# per-query client-side verification cost for K retrieved chunks (Gate 2 + Gate 4 + DBV)
for K in (5, 22, 50):
    sample = [user_recs[i % len(user_recs)] for i in range(K)]
    pr = [cm.get_two_tier_proof(r.user_id, r.chunk_id) for r in sample]

    def full():
        for r, p in zip(sample, pr):
            verify_two_tier_proof(p)
            decrypt_aes_gcm(r.encrypted_content, r.nonce, KEY)
        val.verify_query_boundary(Q[0], val.required_clusters(Q[0])[0])
    m, p95 = med_ms(full, 300)
    results.setdefault("per_query_verification", []).append(
        {"retrieved_chunks": K, "median_ms": round(m, 3), "p95_ms": round(p95, 3)})

json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=2)
print(json.dumps(results, indent=2))
