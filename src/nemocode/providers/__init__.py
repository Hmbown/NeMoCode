# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""NIM providers — shared base class and provider registry."""

from __future__ import annotations

from nemocode.config.schema import Endpoint


class NIMProviderBase:
    """Base class for all NIM providers. Handles endpoint config, auth headers, base URL."""

    def __init__(self, endpoint: Endpoint, api_key: str | None = None) -> None:
        self.endpoint = endpoint
        self._api_key = api_key
        self._base_url = endpoint.base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **self.endpoint.extra_headers,
        }
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers
