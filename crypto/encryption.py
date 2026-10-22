"""AES-256-GCM authenticated encryption and decryption."""
from __future__ import annotations

import base64
import os
import secrets
from pydantic import BaseModel
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


class EncryptedPayload(BaseModel):
    """Container for AES-256-GCM encrypted data."""
    ciphertext_b64: str
    nonce_b64: str
    tag_b64: str | None = None  # AESGCM appends tag to ciphertext in cryptography library

    def serialize_full(self) -> str:
        """Serialize into combined base64 format."""
        return self.ciphertext_b64


def generate_aes_key() -> str:
    """Generate a random 256-bit (32-byte) hex-encoded encryption key."""
    return secrets.token_hex(32)


def _normalize_key(key: bytes | str) -> bytes:
    """Ensure key is exactly 32 bytes."""
    if isinstance(key, str):
        # Try hex decode first
        if len(key) == 64:
            try:
                return bytes.fromhex(key)
            except ValueError:
                pass
        key_bytes = key.encode("utf-8")
    else:
        key_bytes = key

    if len(key_bytes) == 32:
        return key_bytes
    # If key is not 32 bytes, derive via SHA-256
    import hashlib
    return hashlib.sha256(key_bytes).digest()


def encrypt_aes_gcm(plaintext: str | bytes, key: bytes | str) -> EncryptedPayload:
    """
    Encrypt plaintext using AES-256-GCM with a fresh 12-byte nonce.
    Returns EncryptedPayload with base64-encoded ciphertext and nonce.
    """
    if isinstance(plaintext, str):
        data = plaintext.encode("utf-8")
    else:
        data = plaintext

    raw_key = _normalize_key(key)
    aesgcm = AESGCM(raw_key)
    nonce = os.urandom(12)  # 96-bit standard nonce for AES-GCM
    # aesgcm.encrypt produces ciphertext + 16-byte tag
    ciphertext = aesgcm.encrypt(nonce, data, None)

    return EncryptedPayload(
        ciphertext_b64=base64.b64encode(ciphertext).decode("utf-8"),
        nonce_b64=base64.b64encode(nonce).decode("utf-8"),
        tag_b64=base64.b64encode(ciphertext[-16:]).decode("utf-8"),
    )


def decrypt_aes_gcm(
    ciphertext_b64: str,
    nonce_b64: str,
    key: bytes | str,
    associated_data: bytes | None = None,
) -> str:
    """
    Decrypt AES-256-GCM ciphertext using the provided nonce and key.
    Raises Exception if authentication tag validation fails.
    """
    raw_key = _normalize_key(key)
    aesgcm = AESGCM(raw_key)
    ciphertext = base64.b64decode(ciphertext_b64)
    nonce = base64.b64decode(nonce_b64)

    decrypted_bytes = aesgcm.decrypt(nonce, ciphertext, associated_data)
    return decrypted_bytes.decode("utf-8")


class AESGCMCipher:
    """Wrapper class providing static methods for AES-GCM cipher operations."""
    @staticmethod
    def encrypt(plaintext: str | bytes, key: bytes | str) -> EncryptedPayload:
        return encrypt_aes_gcm(plaintext, key)

    @staticmethod
    def decrypt(
        ciphertext_b64: str,
        nonce_b64: str,
        key: bytes | str,
        associated_data: bytes | None = None,
    ) -> str:
        return decrypt_aes_gcm(ciphertext_b64, nonce_b64, key, associated_data)
