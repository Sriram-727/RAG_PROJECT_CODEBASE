import math, re
from database.qdrant_repository import QdrantRepository
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings

def simple_stem(w):
    w = w.lower()
    if len(w) > 3 and w.endswith('ies'):
        return w[:-3] + 'y'
    if len(w) > 4 and w.endswith('ing'):
        return w[:-3]
    if len(w) > 4 and w.endswith('es'):
        return w[:-2]
    if len(w) > 3 and w.endswith('ed'):
        return w[:-2]
    if len(w) > 3 and w.endswith('s') and not w.endswith('ss'):
        return w[:-1]
    return w

STOPWORDS = {'a', 'an', 'the', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of', 'from', 'tell', 'give', 'show', 'please'}

repo = QdrantRepository()
settings = get_settings()
chunks = repo.extract_all_org_chunks('org_enterprise_01')
decrypted = [{'text': decrypt_aes_gcm(c.encrypted_content, c.nonce, settings.DEFAULT_ENCRYPTION_KEY), 'idx': c.chunk_index} for c in chunks]

q = 'tell the project listed in the resume'
q_words = [w.lower() for w in re.findall(r'[a-zA-Z0-9]+', q) if w.lower() not in STOPWORDS]
q_stems = [simple_stem(w) for w in q_words]
print('q_stems:', q_stems)

doc_stems = [[simple_stem(w) for w in re.findall(r'[a-zA-Z0-9]+', d['text'].lower())] for d in decrypted]
df = {}
for tokens in doc_stems:
    for w in set(tokens):
        df[w] = df.get(w, 0) + 1

n_docs = len(decrypted)
idf = {w: math.log(1.0 + (n_docs - df.get(w, 0) + 0.5) / (df.get(w, 0) + 0.5)) for w in q_stems}

scores = []
avgdl = sum(len(t) for t in doc_stems) / n_docs
for i, tokens in enumerate(doc_stems):
    tf = {}
    for w in tokens:
        tf[w] = tf.get(w, 0) + 1
    bm25 = 0.0
    dl = len(tokens)
    for w in q_stems:
        if w in tf:
            term_tf = tf[w]
            num = idf[w] * term_tf * 2.5
            den = term_tf + 1.5 * (0.25 + 0.75 * (dl / avgdl))
            bm25 += num / den
    # Section header boost: if a chunk has 'project' in first 15 tokens
    if 'project' in tokens[:15]:
        bm25 += 5.0
    scores.append(bm25)

for d, s in sorted(zip(decrypted, scores), key=lambda x: x[1], reverse=True)[:6]:
    txt = d['text'].replace('\n', ' ')[:75]
    print(f"Chunk {d['idx']} Score={s:.3f}: {txt}...")
