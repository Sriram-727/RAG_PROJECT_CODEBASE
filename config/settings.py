"""Centralized application settings and environment variable management."""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Settings
    APP_NAME: str = "Enterprise Cloud-Side MMR Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # Qdrant Deployment Settings
    # Modes: 'memory' (in-memory embedded), 'local' (self-hosted docker/service), 'cloud' (Qdrant Cloud)
    QDRANT_MODE: Literal["memory", "local", "cloud"] = "local"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_URL: str | None = None
    QDRANT_API_KEY: str | None = None
    QDRANT_PREFER_GRPC: bool = False
    QDRANT_TIMEOUT: float = 10.0
    DEFAULT_COLLECTION_PREFIX: str = "org_chunks_"

    # MMR Storage Settings
    # Storage modes: 'blob' (local filesystem/blob object store) or 'qdrant' (embedded payload metadata)
    MMR_STORAGE_TYPE: Literal["blob", "qdrant"] = "blob"
    BLOB_STORAGE_DIR: str = "./storage_data/mmr_blobs"

    # Cryptography & Security
    # 256-bit (32 bytes) hex-encoded key for AES-GCM encryption
    DEFAULT_ENCRYPTION_KEY: str = (
        "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
    )

    # Dynamic Boundary & Token Limits
    MAX_TOKENS_PER_CHUNK: int = 2048
    MAX_CHUNKS_PER_REQUEST: int = 50

    # LLM Settings (for grounded inference and hallucination checking)
    LLM_API_KEY: str | None = None
    LLM_MODEL: str = "gpt-4o-mini"
    ENABLE_MOCK_LLM: bool = True

    # Google OAuth 2.0 / OpenID Connect (OIDC) Credentials
    GOOGLE_CLIENT_ID: str | None = Field(default=None, description="Google OAuth 2.0 Client ID from GCP Console")
    GOOGLE_CLIENT_SECRET: str | None = Field(default=None, description="Google OAuth 2.0 Client Secret from GCP Console")
    GOOGLE_REDIRECT_URI: str = Field(
        default="http://127.0.0.1:8000/api/v1/auth/google/callback",
        description="Authorized redirect URI registered in GCP OAuth client credentials"
    )


@lru_cache()
def get_settings() -> Settings:
    """Return cached application settings singleton."""
    return Settings()
