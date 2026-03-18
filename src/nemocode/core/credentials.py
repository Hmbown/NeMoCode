# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Credential management — keyring-first, env-fallback.

Phase 1: Provides secure storage via system keyring (macOS Keychain,
GNOME Keyring, Windows Credential Locker) with fallback to environment variables.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)

_SERVICE_NAME = "nemocode"

# Known credential keys and their descriptions
KNOWN_KEYS: dict[str, str] = {
    "NVIDIA_API_KEY": "NVIDIA API Key (build.nvidia.com / NIM API Catalog)",
    "OPENROUTER_API_KEY": "OpenRouter API Key",
    "TOGETHER_API_KEY": "Together AI API Key",
    "DEEPINFRA_API_KEY": "DeepInfra API Key",
    "FIREWORKS_API_KEY": "Fireworks AI API Key",
}


def _keyring_available() -> bool:
    """Check if keyring library is available and functional."""
    try:
        import keyring

        # Test that the backend is usable
        keyring.get_keyring()
        return True
    except Exception:
        return False


def get_credential(key_name: str) -> str | None:
    """Get a credential value. Checks keyring first, then environment."""
    # Try keyring
    if _keyring_available():
        try:
            import keyring

            value = keyring.get_password(_SERVICE_NAME, key_name)
            if value:
                return value
        except Exception as e:
            logger.debug("Keyring read failed for %s: %s", key_name, e)

    # Fall back to environment
    return os.environ.get(key_name)


def set_credential(key_name: str, value: str) -> tuple[bool, str]:
    """Store a credential. Returns (success, storage_method)."""
    if _keyring_available():
        try:
            import keyring

            keyring.set_password(_SERVICE_NAME, key_name, value)
            return True, "keyring"
        except Exception as e:
            logger.warning("Keyring write failed for %s: %s", key_name, e)

    # Can't store without keyring — user needs to use env vars
    return False, "none"


def delete_credential(key_name: str) -> bool:
    """Remove a credential from keyring."""
    if _keyring_available():
        try:
            import keyring

            keyring.delete_password(_SERVICE_NAME, key_name)
            return True
        except Exception:
            return False
    return False


def list_credentials() -> dict[str, dict[str, Any]]:
    """List all known credentials and their status."""
    result = {}
    for key_name, description in KNOWN_KEYS.items():
        value = get_credential(key_name)
        source = "none"
        if value:
            # Determine where it came from
            if _keyring_available():
                try:
                    import keyring

                    kr_val = keyring.get_password(_SERVICE_NAME, key_name)
                    if kr_val:
                        source = "keyring"
                    else:
                        source = "env"
                except Exception:
                    source = "env"
            else:
                source = "env"

        result[key_name] = {
            "description": description,
            "configured": value is not None,
            "source": source,
            "masked_value": _mask(value) if value else None,
        }
    return result


def _mask(value: str) -> str:
    """Mask a credential value for display."""
    if len(value) <= 8:
        return "*" * len(value)
    return value[:4] + "*" * (len(value) - 8) + value[-4:]


async def test_credential(key_name: str) -> dict[str, Any]:
    """Test if a credential works by making a lightweight API call."""
    import httpx

    value = get_credential(key_name)
    if not value:
        return {"status": "missing", "key": key_name}

    # Test based on which API the key is for
    test_urls: dict[str, str] = {
        "NVIDIA_API_KEY": "https://integrate.api.nvidia.com/v1/models",
        "OPENROUTER_API_KEY": "https://openrouter.ai/api/v1/models",
        "TOGETHER_API_KEY": "https://api.together.xyz/v1/models",
        "DEEPINFRA_API_KEY": "https://api.deepinfra.com/v1/openai/models",
        "FIREWORKS_API_KEY": "https://api.fireworks.ai/inference/v1/models",
    }

    url = test_urls.get(key_name)
    if not url:
        return {"status": "unknown", "key": key_name, "message": "No test URL for this key"}

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers={"Authorization": f"Bearer {value}"})
            if resp.status_code == 200:
                return {"status": "ok", "key": key_name}
            elif resp.status_code == 401:
                return {"status": "invalid", "key": key_name, "message": "Authentication failed"}
            else:
                return {"status": "error", "key": key_name, "message": f"HTTP {resp.status_code}"}
    except Exception as e:
        return {"status": "error", "key": key_name, "message": str(e)}


def storage_backend() -> str:
    """Return the name of the active storage backend."""
    if _keyring_available():
        try:
            import keyring

            return type(keyring.get_keyring()).__name__
        except Exception:
            pass
    return "environment-only"
