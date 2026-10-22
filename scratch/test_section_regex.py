import re

with open("scratch/print_full_chunks.py") as f:
    pass

from database.qdrant_client import QdrantClientManager
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings

client = QdrantClientManager.get_client()
key = get_settings().DEFAULT_ENCRYPTION_KEY
pts, _ = client.scroll("org_chunks_org_enterprise_01", limit=100, with_payload=True)
pts.sort(key=lambda p: (p.payload.get("doc_id", ""), p.payload.get("chunk_index", 0)))

corpus_chunks = []
for p in pts:
    pl = p.payload
    txt = decrypt_aes_gcm(pl['encrypted_content'], pl['nonce'], key) if 'encrypted_content' in pl else pl.get('text', '')
    corpus_chunks.append(txt)

full_corpus = "\n\n".join(corpus_chunks)

all_major_sections = [
    r"CAREER OBJECTIVES?", r"OBJECTIVES?", r"PROFESSIONAL SUMMARY", r"SUMMARY", r"PROFILE", r"ABOUT ME",
    r"AREAS OF INTEREST", r"CERTIFICATIONS?", r"ACHIEVEMENTS?", r"PROJECTS?", r"TECHNICAL SKILLS", r"SKILLS",
    r"EDUCATION", r"EXPERIENCE", r"WORK HISTORY", r"WORK EXPERIENCE", r"PUBLICATIONS?", r"AWARDS?",
    r"INTERESTS?", r"DECLARATION", r"REFERENCES"
]

def test_extract(target, sec_header_pattern):
    filtered_major = [s for s in all_major_sections if not re.search(sec_header_pattern, s, re.I)]
    major_sections = "|".join(filtered_major)
    
    # Require section header to appear at start of line
    pattern = (
        r"(?:^|\n)\s*(?:[#*-]*\s*)?(?:\d+[\.\)]\s*)?(?:key\s+|academic\s+|personal\s+|technical\s+|core\s+)?\b("
        + sec_header_pattern
        + r")\b\s*[:.\n-]*\s*\n*(.*?)(?=(?:\n\s*(?:[#*-]*\s*)?(?:"
        + major_sections
        + r")\b\s*[:\n]|\Z))"
    )
    m = re.search(pattern, full_corpus, re.I | re.DOTALL)
    if m:
        content = m.group(2).strip()
        print(f"=== TARGET: {target} ===")
        print(f"Header: {m.group(1)}")
        print(f"Content preview:\n{content[:200]}")
        print()
    else:
        print(f"=== TARGET: {target} === NOT FOUND")

test_extract("objective", r"(?:career\s+)?objectives?|about\s+me|summary")
test_extract("skill", r"(?:technical\s+)?(?:skills?|technologies|tech\s+stack)")
test_extract("project", r"projects?")
test_extract("education", r"(?:academic\s+)?(?:education|qualifications?)")
test_extract("interest", r"areas\s+of\s+interest|interests?")
test_extract("achievement", r"achievements?|awards?|honors?")
test_extract("certification", r"certifications?|licenses?")
