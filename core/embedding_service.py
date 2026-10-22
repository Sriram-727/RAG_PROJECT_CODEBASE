import logging
import numpy as np
import hashlib
import os
import re
from typing import List, Any

logger = logging.getLogger("EmbeddingService")

try:
    from sentence_transformers import SentenceTransformer
    HAS_ST = True
except Exception:
    HAS_ST = False


class EmbeddingService:
    """
    Wraps a real sentence-embedding model so query text and document text
    are embedded semantically instead of via hash-seeded random vectors.
    Loaded once and reused (lazy singleton) to avoid reloading weights per request.
    """
    _model: Any = None
    _model_name: str = "all-MiniLM-L6-v2"  # 384-dim, fast, CPU-friendly

    @classmethod
    def get_model(cls):
        if cls._model is None:
            force_mock = os.environ.get("USE_MOCK_EMBEDDINGS", "").lower() in ("true", "1", "yes")
            if HAS_ST and not force_mock:
                try:
                    logger.info(f"Loading embedding model '{cls._model_name}'...")
                    # Try offline cache first to avoid remote HTTP roundtrips
                    try:
                        cls._model = SentenceTransformer(cls._model_name, local_files_only=True)
                    except Exception:
                        cls._model = SentenceTransformer(cls._model_name)
                    logger.info("Embedding model loaded successfully.")
                except Exception as e:
                    logger.warning(f"Could not load SentenceTransformer ({e}). Using feature-hash fallback.")
                    cls._model = "MOCK_MODEL"
            else:
                logger.info("Using feature-hash embeddings.")
                cls._model = "MOCK_MODEL"
        return cls._model

    @classmethod
    def embed(cls, text: str) -> np.ndarray:
        """Embed a single string. Returns a normalized float32 vector."""
        if cls.get_model() == "MOCK_MODEL":
            return cls._mock_embed(text)
        vec = cls.get_model().encode(text, normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)

    @classmethod
    def embed_batch(cls, texts: List[str]) -> np.ndarray:
        """Embed a list of strings. Returns an (N, dim) normalized float32 matrix."""
        if cls.get_model() == "MOCK_MODEL":
            return np.array([cls._mock_embed(t) for t in texts], dtype=np.float32)
        vecs = cls.get_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)

    @classmethod
    def dimension(cls) -> int:
        return 384

    @classmethod
    def _mock_embed(cls, text: str) -> np.ndarray:
        """
        Deterministic semantic bag-of-words feature hashing for 384-dim space.
        Aligns query keywords (e.g. 'projects', 'resume') with matching document chunk keywords,
        ensuring positive semantic similarity rather than random MD5 white noise.
        """
        dim = 384
        vec = np.zeros(dim, dtype=np.float32)
        words = re.findall(r"[a-zA-Z0-9]+", text.lower())
        if not words:
            words = ["empty"]
        for w in words:
            # Deterministic hash bucket
            h = int(hashlib.md5(w.encode("utf-8")).hexdigest()[:8], 16)
            idx = h % dim
            sign = 1.0 if ((h >> 16) & 1) else -1.0
            vec[idx] += sign

        norm = np.linalg.norm(vec)
        if norm > 1e-9:
            vec /= norm
        return vec