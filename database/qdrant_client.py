"""Qdrant client connection manager supporting in-memory, local Docker, and Qdrant Cloud."""
from __future__ import annotations

import logging
from typing import Optional
from qdrant_client import QdrantClient
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class QdrantClientManager:
    """Manages Qdrant client lifecycle and connectivity."""

    _instance: Optional[QdrantClient] = None
    _mode: Optional[str] = None

    @classmethod
    def get_client(cls, settings: Optional[Settings] = None, force_new: bool = False) -> QdrantClient:
        """
        Get or create a configured QdrantClient instance.
        Automatically handles connection modes: 'memory', 'local', 'cloud'.
        """
        cfg = settings or get_settings()
        mode = cfg.QDRANT_MODE.lower()

        # If previously fell back to in-memory, but user configured local, test if local Docker has become available
        if cls._instance is not None and not force_new:
            if cls._mode == "memory" and mode == "local":
                try:
                    candidate = QdrantClient(
                        host=cfg.QDRANT_HOST,
                        port=cfg.QDRANT_PORT,
                        grpc_port=cfg.QDRANT_GRPC_PORT,
                        api_key=cfg.QDRANT_API_KEY,
                        timeout=1.5,
                        prefer_grpc=cfg.QDRANT_PREFER_GRPC,
                    )
                    candidate.get_collections()
                    logger.info("Local Qdrant Docker is now online. Upgraded from in-memory fallback to Docker instance.")
                    cls._instance = candidate
                    cls._mode = "local"
                    return candidate
                except Exception:
                    return cls._instance
            return cls._instance

        logger.info(f"Initializing Qdrant client in mode: {mode}")

        try:
            if mode == "memory":
                client = QdrantClient(location=":memory:")
                logger.info("Initialized in-memory Qdrant instance.")
            elif mode == "cloud":
                if not cfg.QDRANT_URL:
                    raise ValueError("QDRANT_URL is required for Qdrant Cloud mode.")
                client = QdrantClient(
                    url=cfg.QDRANT_URL,
                    api_key=cfg.QDRANT_API_KEY,
                    timeout=cfg.QDRANT_TIMEOUT,
                    prefer_grpc=cfg.QDRANT_PREFER_GRPC,
                )
                logger.info(f"Connected to Qdrant Cloud at {cfg.QDRANT_URL}")
            elif mode == "local":
                client = QdrantClient(
                    host=cfg.QDRANT_HOST,
                    port=cfg.QDRANT_PORT,
                    grpc_port=cfg.QDRANT_GRPC_PORT,
                    api_key=cfg.QDRANT_API_KEY,
                    timeout=cfg.QDRANT_TIMEOUT,
                    prefer_grpc=cfg.QDRANT_PREFER_GRPC,
                )
                # Verify connectivity
                client.get_collections()
                logger.info(f"Connected to local Qdrant at {cfg.QDRANT_HOST}:{cfg.QDRANT_PORT}")
            else:
                raise ValueError(f"Unsupported QDRANT_MODE: {mode}")

            cls._instance = client
            cls._mode = mode
            return client

        except Exception as e:
            if mode == "local":
                logger.warning(
                    f"Failed to connect to local Qdrant ({e}). Falling back to in-memory mode for development."
                )
                client = QdrantClient(location=":memory:")
                cls._instance = client
                cls._mode = "memory"
                return client
            logger.error(f"Error initializing Qdrant client: {e}")
            raise

    @classmethod
    def reset(cls) -> None:
        """Reset singleton client (useful for unit tests)."""
        cls._instance = None
        cls._mode = None
