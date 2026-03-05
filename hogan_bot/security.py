"""Security module for Hogan — Phase 7 (CryptoSwift-inspired).

Provides:
  1. AES-256-CBC encrypted storage for API keys at rest.
     API keys from ``.env`` are optionally encrypted using a master password
     on first run.  Encrypted keys are stored in SQLite ``api_keys`` table.

  2. HMAC-SHA256 signature verification for incoming webhook payloads.
     Every webhook endpoint should call ``verify_webhook_signature`` before
     processing the payload.

  3. A ``KeyVault`` class that loads API keys — trying encrypted SQLite first,
     then falling back to plaintext ``.env`` values.

Usage::

    from hogan_bot.security import KeyVault, sign_webhook, verify_webhook_signature

    # Encrypt and store API keys on first run:
    vault = KeyVault(master_password="my-secret-password")
    vault.store("kraken_api_key", "MY_KEY_VALUE")
    vault.store("kraken_api_secret", "MY_SECRET_VALUE")

    # Retrieve at runtime:
    key = vault.get("kraken_api_key")

    # Webhook HMAC:
    signature = sign_webhook(payload_bytes, secret="webhook-secret")
    verify_webhook_signature(payload_bytes, signature, secret="webhook-secret")
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import secrets
import sqlite3
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_VAULT_TABLE = "api_keys"


# ---------------------------------------------------------------------------
# AES-256-CBC encryption helpers
# ---------------------------------------------------------------------------

def _derive_key_iv(password: str, salt: bytes) -> tuple[bytes, bytes]:
    """Derive a 32-byte AES key and 16-byte IV from a password using PBKDF2."""
    from hashlib import pbkdf2_hmac
    key_iv = pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations=200_000, dklen=48)
    return key_iv[:32], key_iv[32:48]


def encrypt_aes256(plaintext: str, password: str) -> str:
    """Encrypt *plaintext* with AES-256-CBC and return a base64-encoded string.

    Format: ``<salt_hex>$<iv_hex>$<ciphertext_hex>``
    """
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding
    except ImportError:
        raise RuntimeError(
            "cryptography package not installed. Run: pip install cryptography"
        )

    salt = secrets.token_bytes(16)
    key, iv = _derive_key_iv(password, salt)

    padder = padding.PKCS7(128).padder()
    padded = padder.update(plaintext.encode("utf-8")) + padder.finalize()

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    encryptor = cipher.encryptor()
    ciphertext = encryptor.update(padded) + encryptor.finalize()

    return f"{salt.hex()}${iv.hex()}${ciphertext.hex()}"


def decrypt_aes256(encrypted: str, password: str) -> str:
    """Decrypt an AES-256-CBC encrypted string. Returns plaintext."""
    try:
        from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
        from cryptography.hazmat.backends import default_backend
        from cryptography.hazmat.primitives import padding
    except ImportError:
        raise RuntimeError(
            "cryptography package not installed. Run: pip install cryptography"
        )

    parts = encrypted.split("$")
    if len(parts) != 3:
        raise ValueError("Invalid encrypted format (expected salt$iv$ciphertext)")

    salt = bytes.fromhex(parts[0])
    iv = bytes.fromhex(parts[1])
    ciphertext = bytes.fromhex(parts[2])

    key, _ = _derive_key_iv(password, salt)

    cipher = Cipher(algorithms.AES(key), modes.CBC(iv), backend=default_backend())
    decryptor = cipher.decryptor()
    padded = decryptor.update(ciphertext) + decryptor.finalize()

    unpadder = padding.PKCS7(128).unpadder()
    plaintext = unpadder.update(padded) + unpadder.finalize()
    return plaintext.decode("utf-8")


# ---------------------------------------------------------------------------
# Key Vault
# ---------------------------------------------------------------------------

class KeyVault:
    """Encrypted-at-rest API key storage.

    On first use, encrypts and stores API keys in the SQLite ``api_keys``
    table.  On subsequent runs, loads from SQLite and decrypts with the
    master password.  Falls back to plaintext env vars if no encrypted key
    exists for a given name.
    """

    def __init__(
        self,
        db_path: str = "data/hogan.db",
        master_password: str | None = None,
    ) -> None:
        self.db_path = db_path
        self._password = master_password or os.getenv("HOGAN_VAULT_PASSWORD", "")
        self._ensure_table()

    def _ensure_table(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {_VAULT_TABLE} (
                key_name   TEXT PRIMARY KEY,
                ciphertext TEXT NOT NULL,
                created_at REAL NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()

    def store(self, key_name: str, plaintext_value: str) -> None:
        """Encrypt and store *plaintext_value* under *key_name*."""
        if not self._password:
            raise ValueError("Master password required for key storage. Set HOGAN_VAULT_PASSWORD.")
        ciphertext = encrypt_aes256(plaintext_value, self._password)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            f"INSERT OR REPLACE INTO {_VAULT_TABLE} (key_name, ciphertext, created_at) VALUES (?,?,?)",
            (key_name, ciphertext, time.time()),
        )
        conn.commit()
        conn.close()
        logger.info("KeyVault: stored encrypted key '%s'", key_name)

    def get(self, key_name: str, fallback_env: str | None = None) -> str | None:
        """Retrieve and decrypt *key_name*.

        Falls back to the env var named *fallback_env* (or ``key_name.upper()``)
        if no encrypted value is found.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute(
                f"SELECT ciphertext FROM {_VAULT_TABLE} WHERE key_name=?", (key_name,)
            )
            row = cur.fetchone()
            conn.close()

            if row:
                if not self._password:
                    logger.warning(
                        "KeyVault: encrypted key '%s' found but HOGAN_VAULT_PASSWORD not set.",
                        key_name,
                    )
                    return None
                return decrypt_aes256(row[0], self._password)
        except Exception as exc:
            logger.warning("KeyVault.get('%s') failed: %s — using env fallback", key_name, exc)

        # Env var fallback
        env_name = fallback_env or key_name.upper()
        return os.getenv(env_name, None)

    def encrypt_from_env(self, mappings: dict[str, str]) -> dict[str, bool]:
        """Bulk-encrypt env vars into the vault.

        ``mappings`` is ``{key_name: env_var_name}`` e.g.::

            {"kraken_api_key": "KRAKEN_API_KEY", "kraken_secret": "KRAKEN_SECRET"}

        Returns dict of ``{key_name: success_bool}``.
        """
        results = {}
        for key_name, env_var in mappings.items():
            value = os.getenv(env_var, "")
            if not value:
                logger.warning("KeyVault: env var %s is empty — skipping.", env_var)
                results[key_name] = False
                continue
            try:
                self.store(key_name, value)
                results[key_name] = True
            except Exception as exc:
                logger.error("KeyVault: failed to store %s: %s", key_name, exc)
                results[key_name] = False
        return results


# ---------------------------------------------------------------------------
# HMAC webhook verification
# ---------------------------------------------------------------------------

def sign_webhook(payload: bytes, secret: str) -> str:
    """Compute HMAC-SHA256 signature for *payload*.

    Returns a hex digest string.
    """
    return hmac.new(secret.encode("utf-8"), payload, hashlib.sha256).hexdigest()


def verify_webhook_signature(
    payload: bytes,
    signature: str,
    secret: str | None = None,
) -> bool:
    """Verify that *signature* matches HMAC-SHA256 of *payload*.

    Uses ``HOGAN_WEBHOOK_SECRET`` env var if *secret* is None.
    Returns True if valid, False otherwise.
    Raises ``ValueError`` if no secret is configured.
    """
    secret = secret or os.getenv("HOGAN_WEBHOOK_SECRET", "")
    if not secret:
        raise ValueError(
            "No webhook secret configured. Set HOGAN_WEBHOOK_SECRET or pass secret= explicitly."
        )
    expected = sign_webhook(payload, secret)
    return hmac.compare_digest(expected.lower(), signature.lower().removeprefix("sha256="))


# ---------------------------------------------------------------------------
# CLI — encrypt existing .env API keys into the vault
# ---------------------------------------------------------------------------
def _main() -> None:
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Hogan Key Vault CLI")
    p.add_argument("--db", default="data/hogan.db")
    sub = p.add_subparsers(dest="cmd")

    enc = sub.add_parser("encrypt-env", help="Encrypt API keys from .env into the vault")
    enc.add_argument("--password", required=True, help="Master encryption password")

    dec = sub.add_parser("get", help="Retrieve and decrypt a key")
    dec.add_argument("key_name")
    dec.add_argument("--password", required=True)

    args = p.parse_args()
    if args.cmd is None:
        p.print_help()
        return

    if args.cmd == "encrypt-env":
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
        vault = KeyVault(db_path=args.db, master_password=args.password)
        results = vault.encrypt_from_env({
            "kraken_api_key": "KRAKEN_API_KEY",
            "kraken_api_secret": "KRAKEN_SECRET",
            "coingecko_key": "COINGECKO_KEY",
            "cryptoquant_key": "CRYPTOQUANT_KEY",
            "glassnode_key": "GLASSNODE_KEY",
            "santiment_key": "SANTIMENT_KEY",
            "cryptopanic_key": "CRYPTOPANIC_KEY",
            "telegram_token": "HOGAN_TELEGRAM_TOKEN",
        })
        print(json.dumps(results, indent=2))

    elif args.cmd == "get":
        vault = KeyVault(db_path=args.db, master_password=args.password)
        val = vault.get(args.key_name)
        print(val or "(not found)")


if __name__ == "__main__":
    _main()
