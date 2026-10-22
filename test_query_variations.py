from core.embedding_service import EmbeddingService
import numpy as np

chunk2 = """PROJECTS:
1. Real-time Fraud Detection Engine: High-throughput stream processing system built with Apache Kafka and Flink, classifying 50,000 financial transactions per second with sub-10ms inference."""

chunk0 = """JOHN DOE - SOFTWARE ENGINEER
Email: john.doe@example.com | Phone: 555-0199
PROFESSIONAL SUMMARY: Accomplished Full-Stack Engineer."""

v2 = EmbeddingService.embed(chunk2)
v0 = EmbeddingService.embed(chunk0)

queries = [
    "what is the name of projects in the resume pdf",
    "what is the name of projects",
    "projects in the resume",
    "projects",
    "what are the projects",
    "list projects",
    "projects developed by the candidate"
]

for q in queries:
    qv = EmbeddingService.embed(q)
    s2 = np.dot(qv, v2)
    s0 = np.dot(qv, v0)
    print(f"Query: '{q}'")
    print(f"   -> Sim with Chunk 2 (Projects): {s2:.4f}")
    print(f"   -> Sim with Chunk 0 (Summary):  {s0:.4f}")
    print(f"   -> Winner: {'CHUNK 2 (PROJECTS)' if s2 > s0 else 'CHUNK 0 (SUMMARY)'}\n")
