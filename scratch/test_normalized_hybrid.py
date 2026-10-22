import sys
import re
import math
sys.path.insert(0, ".")

from database.qdrant_repository import QdrantRepository
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings
from core.embedding_service import EmbeddingService

STOPWORDS = {
    "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by", "of",
    "from", "as", "is", "was", "are", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "should", "would", "may", "might", "must", "shall",
    "what", "when", "where", "who", "whom", "which", "why", "how", "this", "that", "these",
    "those", "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
    "my", "your", "his", "their", "our", "its", "tell", "give", "show", "list", "about",
    "question", "questions", "me", "please", "can", "you", "pdf", "file", "document"
}

def extract_salient_terms(query: str):
    words = re.findall(r"[a-zA-Z0-9]+", query.lower())
    salient = [w for w in words if w not in STOPWORDS and len(w) > 1]
    return salient if salient else words

def compute_bm25_scores(query: str, documents, k1: float = 1.5, b: float = 0.75):
    q_terms = extract_salient_terms(query)
    if not q_terms:
        return [0.0] * len(documents)

    doc_tokens = [re.findall(r"[a-zA-Z0-9]+", d.get("text", "").lower()) for d in documents]
    doc_lens = [len(tokens) for tokens in doc_tokens]
    avgdl = sum(doc_lens) / len(documents) if documents else 1.0
    n_docs = len(documents)

    df = {}
    for tokens in doc_tokens:
        for w in set(tokens):
            df[w] = df.get(w, 0) + 1

    idf = {}
    for w in q_terms:
        freq = df.get(w, 0)
        idf[w] = math.log(1.0 + (n_docs - freq + 0.5) / (freq + 0.5))

    q_bigrams = [f"{q_terms[i]} {q_terms[i+1]}" for i in range(len(q_terms) - 1)]

    scores = []
    for i, (d, tokens) in enumerate(zip(documents, doc_tokens)):
        text_lower = d.get("text", "").lower()
        tf = {}
        for w in tokens:
            tf[w] = tf.get(w, 0) + 1

        bm25 = 0.0
        dl = doc_lens[i]
        for w in q_terms:
            if w in tf:
                term_tf = tf[w]
                num = idf[w] * term_tf * (k1 + 1.0)
                den = term_tf + k1 * (1.0 - b + b * (dl / avgdl))
                bm25 += num / den

        phrase_bonus = 0.0
        for bg in q_bigrams:
            if bg in text_lower:
                phrase_bonus += 3.5

        scores.append(bm25 + phrase_bonus)
    return scores

def hybrid_rerank(documents, dense_scores, bm25_scores, alpha: float = 0.65):
    """
    Normalized Hybrid Score:
    alpha * BM25_norm + (1 - alpha) * Dense_norm
    """
    max_bm25 = max(bm25_scores) if bm25_scores else 1.0
    min_dense = min(dense_scores) if dense_scores else 0.0
    max_dense = max(dense_scores) if dense_scores else 1.0
    dense_range = (max_dense - min_dense) if (max_dense - min_dense) > 1e-6 else 1.0

    scored_docs = []
    for i, doc in enumerate(documents):
        bm = bm25_scores[i]
        dn = dense_scores[i]

        bm_norm = (bm / max_bm25) if max_bm25 > 0 else 0.0
        dn_norm = (dn - min_dense) / dense_range

        if max_bm25 > 0 and bm > 0:
            final_score = alpha * bm_norm + (1.0 - alpha) * dn_norm
        elif max_bm25 > 0:
            # Query has keywords, but this doc matched NONE of them
            final_score = (1.0 - alpha) * dn_norm * 0.4
        else:
            final_score = dn_norm

        d_copy = dict(doc)
        d_copy["score"] = round(float(final_score), 5)
        d_copy["bm25_score"] = round(float(bm), 3)
        d_copy["dense_score"] = round(float(dn), 3)
        scored_docs.append(d_copy)

    scored_docs.sort(key=lambda x: x["score"], reverse=True)
    return scored_docs

def stitch_candidates(top_docs, all_docs_by_idx, max_window: int = 2):
    """
    Expand top candidates by stitching adjacent neighbor chunk if split across boundaries.
    """
    result = []
    seen_indices = set()

    for doc in top_docs:
        c_idx = doc.get("chunk_index")
        if c_idx in seen_indices:
            continue

        merged_doc = dict(doc)
        cur_text = merged_doc.get("text", "")

        # Look for next neighbor c_idx + 1
        nxt = all_docs_by_idx.get(c_idx + 1)
        if nxt and (len(cur_text) < 800):
            nxt_text = nxt.get("text", "")
            # check overlap
            merged = False
            for ov_len in range(min(100, len(cur_text), len(nxt_text)), 15, -1):
                suffix = cur_text[-ov_len:]
                if nxt_text.startswith(suffix):
                    cur_text = cur_text + nxt_text[ov_len:]
                    merged = True
                    break
            if not merged:
                cur_text = cur_text + "\n" + nxt_text
            seen_indices.add(c_idx + 1)

        merged_doc["text"] = cur_text
        seen_indices.add(c_idx)
        result.append(merged_doc)

    return result

# Run test on all queries
repo = QdrantRepository()
settings = get_settings()
chunks = repo.extract_all_org_chunks('org_enterprise_01')
all_docs = [
    {
        'text': decrypt_aes_gcm(c.encrypted_content, c.nonce, settings.DEFAULT_ENCRYPTION_KEY),
        'chunk_index': c.chunk_index,
        'doc_id': c.doc_id,
    }
    for c in chunks
]
all_by_idx = {d['chunk_index']: d for d in all_docs}

test_queries = [
    'what is multithreading in java give solution for this question',
    'tell the project listed in the resume',
    'give company information',
    'what was the system design question',
    'tell about the career objective'
]

for q in test_queries:
    q_vec = EmbeddingService.embed(q)
    dense_scores = [float(q_vec.dot(EmbeddingService.embed(d['text'][:250]))) for d in all_docs]
    bm25_scores = compute_bm25_scores(q, all_docs)
    reranked = hybrid_rerank(all_docs, dense_scores, bm25_scores, alpha=0.7)
    stitched = stitch_candidates(reranked[:5], all_by_idx)

    print("=" * 75)
    print(f"QUERY: {q}")
    max_bm = max(bm25_scores) if bm25_scores else 0
    print(f"Max BM25 across corpus: {max_bm:.3f}")
    for rank, it in enumerate(stitched[:3]):
        c_idx = it.get('chunk_index')
        score = it.get('score')
        bm = it.get('bm25_score')
        dn = it.get('dense_score')
        txt = it['text'].replace('\n', ' ')[:95]
        print(f"  #{rank+1} [Chunk {c_idx}] Score={score:.4f} (BM25={bm}, Dense={dn})")
        print(f"      {txt}...")
