# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Thin HTTP clients for NVIDIA NeMo Microservices.

Supports Data Designer, Evaluator, and Safe Synthesizer with env-var overrides
for base URLs and API keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterator

import httpx

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _api_key() -> str | None:
    """Return the best available API key, preferring NIM_API_KEY."""
    return os.environ.get("NIM_API_KEY") or os.environ.get("NGC_CLI_API_KEY")


def _auth_headers() -> dict[str, str]:
    key = _api_key()
    if key:
        return {"Authorization": f"Bearer {key}"}
    return {}


# ---------------------------------------------------------------------------
# Base client
# ---------------------------------------------------------------------------


@dataclass
class _BaseClient:
    """Minimal wrapper around httpx for a single NeMo microservice."""

    base_url: str
    timeout: float = 60.0
    _client: httpx.Client = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=_auth_headers(),
            timeout=self.timeout,
        )

    def close(self) -> None:
        self._client.close()

    def get(self, path: str, **kwargs: Any) -> httpx.Response:
        resp = self._client.get(path, **kwargs)
        resp.raise_for_status()
        return resp

    def post(self, path: str, **kwargs: Any) -> httpx.Response:
        resp = self._client.post(path, **kwargs)
        resp.raise_for_status()
        return resp

    def delete(self, path: str, **kwargs: Any) -> httpx.Response:
        resp = self._client.delete(path, **kwargs)
        resp.raise_for_status()
        return resp

    def stream_post(self, path: str, **kwargs: Any) -> Iterator[str]:
        """POST with streaming response — yields lines."""
        with self._client.stream("POST", path, **kwargs) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    yield line

    def health(self) -> bool:
        try:
            resp = self._client.get("/health")
            return resp.status_code == 200
        except httpx.HTTPError:
            return False


# ---------------------------------------------------------------------------
# Data Designer client
# ---------------------------------------------------------------------------


class DataDesignerClient(_BaseClient):
    """Client for NeMo Data Designer REST API.

    Endpoints follow /v1/data-designer/... per the official OpenAPI spec.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        url = base_url or _env("NEMOCODE_DATA_BASE_URL", "http://localhost:8080")
        super().__init__(base_url=url, timeout=timeout)

    # -- Preview -----------------------------------------------------------

    def preview(self, config: dict[str, Any], num_records: int = 5) -> list[dict[str, Any]]:
        """POST /v1/data-designer/preview — generate a small preview batch."""
        import json

        payload = {"config": config, "num_records": num_records}
        records: list[dict[str, Any]] = []
        for line in self.stream_post("/v1/data-designer/preview", json=payload):
            try:
                msg = json.loads(line)
            except (json.JSONDecodeError, TypeError):
                continue
            if msg.get("message_type") == "dataset":
                records.append(msg)
        return records

    # -- Jobs --------------------------------------------------------------

    def create_job(
        self,
        name: str,
        config: dict[str, Any],
        num_records: int = 100,
        description: str = "",
    ) -> dict[str, Any]:
        """POST /v1/data-designer/jobs — create an async generation job."""
        payload: dict[str, Any] = {
            "name": name,
            "spec": {"config": config, "num_records": num_records},
        }
        if description:
            payload["description"] = description
        return self.post("/v1/data-designer/jobs", json=payload).json()

    def list_jobs(self, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        """GET /v1/data-designer/jobs"""
        return self.get(
            "/v1/data-designer/jobs",
            params={"page": page, "page_size": page_size},
        ).json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        """GET /v1/data-designer/jobs/{job_id}"""
        return self.get(f"/v1/data-designer/jobs/{job_id}").json()

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """GET /v1/data-designer/jobs/{job_id}/status"""
        return self.get(f"/v1/data-designer/jobs/{job_id}/status").json()

    def get_job_logs(self, job_id: str, limit: int = 100) -> dict[str, Any]:
        """GET /v1/data-designer/jobs/{job_id}/logs"""
        return self.get(
            f"/v1/data-designer/jobs/{job_id}/logs",
            params={"limit": limit},
        ).json()

    def get_job_results(self, job_id: str) -> dict[str, Any]:
        """GET /v1/data-designer/jobs/{job_id}/results"""
        return self.get(f"/v1/data-designer/jobs/{job_id}/results").json()

    def download_dataset(self, job_id: str) -> bytes:
        """GET /v1/data-designer/jobs/{job_id}/results/dataset/download"""
        return self.get(f"/v1/data-designer/jobs/{job_id}/results/dataset/download").content

    def download_analysis(self, job_id: str) -> dict[str, Any]:
        """GET /v1/data-designer/jobs/{job_id}/results/analysis/download"""
        return self.get(f"/v1/data-designer/jobs/{job_id}/results/analysis/download").json()


# ---------------------------------------------------------------------------
# Evaluator client
# ---------------------------------------------------------------------------


class EvaluatorClient(_BaseClient):
    """Client for NeMo Evaluator REST API.

    Endpoints follow /v2/evaluation/jobs/... per the official OpenAPI spec.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        url = base_url or _env("NEMOCODE_EVAL_BASE_URL", "http://localhost:8080")
        super().__init__(base_url=url, timeout=timeout)

    def create_job(self, spec: dict[str, Any]) -> dict[str, Any]:
        """POST /v2/evaluation/jobs"""
        return self.post("/v2/evaluation/jobs", json=spec).json()

    def list_jobs(self, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        """GET /v2/evaluation/jobs"""
        return self.get(
            "/v2/evaluation/jobs",
            params={"page": page, "page_size": page_size},
        ).json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        """GET /v2/evaluation/jobs/{job_id}"""
        return self.get(f"/v2/evaluation/jobs/{job_id}").json()

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """GET /v2/evaluation/jobs/{job_id}/status"""
        return self.get(f"/v2/evaluation/jobs/{job_id}/status").json()

    def get_job_logs(self, job_id: str, limit: int = 100) -> dict[str, Any]:
        """GET /v2/evaluation/jobs/{job_id}/logs"""
        return self.get(
            f"/v2/evaluation/jobs/{job_id}/logs",
            params={"limit": limit},
        ).json()

    def get_job_results(self, job_id: str) -> dict[str, Any]:
        """GET /v2/evaluation/jobs/{job_id}/results"""
        return self.get(f"/v2/evaluation/jobs/{job_id}/results").json()

    def download_results(self, job_id: str) -> dict[str, Any]:
        """GET /v2/evaluation/jobs/{job_id}/results/evaluation-results/download"""
        return self.get(f"/v2/evaluation/jobs/{job_id}/results/evaluation-results/download").json()


# ---------------------------------------------------------------------------
# Safe Synthesizer client
# ---------------------------------------------------------------------------


class SafeSynthesizerClient(_BaseClient):
    """Client for NeMo Safe Synthesizer REST API.

    Endpoints follow /v1beta1/safe-synthesizer/... per the official OpenAPI spec.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float = 120.0,
    ) -> None:
        url = base_url or _env("NEMOCODE_SAFE_SYNTH_BASE_URL", "http://localhost:8080")
        super().__init__(base_url=url, timeout=timeout)

    def create_job(self, spec: dict[str, Any]) -> dict[str, Any]:
        """POST /v1beta1/safe-synthesizer/jobs"""
        return self.post("/v1beta1/safe-synthesizer/jobs", json=spec).json()

    def list_jobs(self, page: int = 1, page_size: int = 20) -> dict[str, Any]:
        """GET /v1beta1/safe-synthesizer/jobs"""
        return self.get(
            "/v1beta1/safe-synthesizer/jobs",
            params={"page": page, "page_size": page_size},
        ).json()

    def get_job(self, job_id: str) -> dict[str, Any]:
        """GET /v1beta1/safe-synthesizer/jobs/{job_id}"""
        return self.get(f"/v1beta1/safe-synthesizer/jobs/{job_id}").json()

    def get_job_status(self, job_id: str) -> dict[str, Any]:
        """GET /v1beta1/safe-synthesizer/jobs/{job_id}/status"""
        return self.get(f"/v1beta1/safe-synthesizer/jobs/{job_id}/status").json()

    def get_job_logs(self, job_id: str, limit: int = 100) -> dict[str, Any]:
        """GET /v1beta1/safe-synthesizer/jobs/{job_id}/logs"""
        return self.get(
            f"/v1beta1/safe-synthesizer/jobs/{job_id}/logs",
            params={"limit": limit},
        ).json()

    def get_job_results(self, job_id: str) -> dict[str, Any]:
        """GET /v1beta1/safe-synthesizer/jobs/{job_id}/results"""
        return self.get(f"/v1beta1/safe-synthesizer/jobs/{job_id}/results").json()

    def download_synthetic_data(self, job_id: str) -> bytes:
        """GET /v1beta1/safe-synthesizer/jobs/{job_id}/results/synthetic_data/download"""
        return self.get(
            f"/v1beta1/safe-synthesizer/jobs/{job_id}/results/synthetic_data/download"
        ).content
