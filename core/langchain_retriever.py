"""
Modular LangChain Local RAG Retrieval Pipeline.
100% Offline execution with zero cloud API dependencies.

Implements:
1. Qdrant Vector Store connection (Docker at localhost:6333).
2. HuggingFace Embeddings (sentence-transformers/all-MiniLM-L6-v2, 384-dimensional).
3. Local MultiQueryRetriever expansion via Ollama (llama3.2:1b).
4. Local FlashRank cross-encoder reranking (Top 10 -> Top 3).
"""
from __future__ import annotations

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import logging
import warnings
from functools import lru_cache
from typing import Any, List, Optional

from qdrant_client import QdrantClient

warnings.filterwarnings("ignore", category=DeprecationWarning)
logger = logging.getLogger("LangChainRetriever")


@lru_cache(maxsize=1)
def _get_qdrant_client(qdrant_url: str = "http://localhost:6333") -> QdrantClient:
    return QdrantClient(url=qdrant_url)


@lru_cache(maxsize=1)
def _get_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_compressor():
    try:
        from langchain.retrievers.document_compressors import FlashrankRerank
    except ImportError:
        from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank

    return FlashrankRerank(
        model="ms-marco-TinyBERT-L-2-v2",
        top_n=3,
    )


def _resolve_document_text(
    doc: Any,
    client: Optional[QdrantClient] = None,
    collection_name: str = "org_1_collection",
) -> str:
    """Extract raw plaintext from Document, supporting decrypted payloads if present."""
    if hasattr(doc, "page_content") and doc.page_content:
        return doc.page_content.strip()

    metadata = getattr(doc, "metadata", {})
    if isinstance(metadata, dict):
        if "text" in metadata and metadata["text"]:
            return str(metadata["text"]).strip()

        point_id = metadata.get("_id")
        if point_id and client:
            try:
                from crypto.encryption import decrypt_aes_gcm
                from config.settings import get_settings

                pts = client.retrieve(
                    collection_name=collection_name,
                    ids=[point_id],
                    with_payload=True,
                )
                if pts and pts[0].payload:
                    payload = pts[0].payload
                    if "text" in payload and payload["text"]:
                        return str(payload["text"]).strip()
                    ciphertext = payload.get("encrypted_content")
                    nonce = payload.get("nonce")
                    if ciphertext and nonce:
                        return decrypt_aes_gcm(
                            ciphertext,
                            nonce,
                            get_settings().DEFAULT_ENCRYPTION_KEY,
                        ).strip()
            except Exception as e:
                logger.debug(f"Payload extraction note: {e}")

    return str(getattr(doc, "page_content", "")).strip()


def retrieve_optimized_chunks(
    user_query: str,
    collection_name: str = "org_1_collection",
    qdrant_url: str = "http://localhost:6333",
    ollama_model: str = "llama3.2:1b",
    ollama_base_url: str = "http://localhost:11434",
) -> List[str]:
    """
    Executes offline multi-query expansion, Qdrant retrieval, and FlashRank
    reranking. Returns the raw text of the top 3 chunks.

    Args:
        user_query: The incoming natural language question or search query.
        collection_name: Name of the target Qdrant collection (default: "org_1_collection").
        qdrant_url: URL for the local Qdrant instance.
        ollama_model: Name of the local Ollama LLM to use for query expansion.
        ollama_base_url: URL for the local Ollama instance.

    Returns:
        List of 3 raw text strings [chunk1, chunk2, chunk3].
    """
    try:
        from langchain.retrievers.multi_query import MultiQueryRetriever
    except ImportError:
        from langchain_classic.retrievers import MultiQueryRetriever

    from langchain_ollama import ChatOllama
    from langchain_qdrant import QdrantVectorStore

    client = _get_qdrant_client(qdrant_url)
    embeddings = _get_embeddings()

    # 1. Connect to existing Qdrant collection
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=collection_name,
        embedding=embeddings,
        content_payload_key="text",
    )
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # 2. Local Multi-Query Expansion via Ollama
    candidate_docs = []
    try:
        local_llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.0,
        )
        multi_query_retriever = MultiQueryRetriever.from_llm(
            retriever=base_retriever,
            llm=local_llm,
            include_original=True,
        )
        candidate_docs = multi_query_retriever.invoke(user_query)
    except Exception as e:
        logger.info(
            f"Ollama expansion note ({e}). Proceeding directly with base vector retrieval."
        )
        candidate_docs = base_retriever.invoke(user_query)

    # 3. Ensure document page_content is populated before reranking
    for doc in candidate_docs:
        if not getattr(doc, "page_content", None):
            doc.page_content = _resolve_document_text(
                doc, client=client, collection_name=collection_name
            )

    # 4. Local FlashRank Cross-Encoder Reranking (top 10 -> top 3)
    compressor = _get_compressor()
    ranked_documents = compressor.compress_documents(
        documents=candidate_docs,
        query=user_query,
    )

    # 5. Extract raw string texts
    final_chunks = [
        _resolve_document_text(doc, client=client, collection_name=collection_name)
        for doc in ranked_documents
    ]

    return [chunk for chunk in final_chunks if chunk][:3]
