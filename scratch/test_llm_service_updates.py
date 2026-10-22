import sys
import re
sys.path.insert(0, ".")

from core.llm_service import LLMService
from database.qdrant_repository import QdrantRepository
from crypto.encryption import decrypt_aes_gcm
from config.settings import get_settings

repo = QdrantRepository()
settings = get_settings()
chunks = repo.extract_all_org_chunks('org_enterprise_01')
full_text = "\n".join([decrypt_aes_gcm(c.encrypted_content, c.nonce, settings.DEFAULT_ENCRYPTION_KEY) for c in chunks])

print("Testing Projects extraction...")
print("Projects:", LLMService._extract_section_answer("tell the project listed in the resume", full_text))

print("\nTesting Career Objective query...")
print("Objective:", LLMService._extract_section_answer("tell about the career objective", full_text))
