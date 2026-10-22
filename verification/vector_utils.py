"""
vector_utils.py

Small pure-Python vector helpers (no numpy dependency, consistent with
the rest of this module). Vectors are just lists of floats -- in the
real system these would be masked embeddings; here they're mocked as
low-dimensional points so clustering behavior is easy to see/debug.
"""

import math
from crypto_utils import sha256_hex


def euclidean(a: list, b: list) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def mean_vector(vectors: list) -> list:
    n = len(vectors)
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / n for i in range(dim)]


def vector_hash(v: list) -> str:
    """Deterministic hash of a vector, for committing centroids into an MMR."""
    rounded = [round(x, 6) for x in v]
    return sha256_hex(str(rounded).encode("utf-8"))
