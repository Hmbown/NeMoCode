# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""CustomizerClient — REST client for NVIDIA NeMo Customizer (LoRA fine-tuning).

Communicates with the NeMo Customizer service to create, monitor, and retrieve
LoRA fine-tuning jobs for Nemotron 3 Nano 4B.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CUSTOMIZER_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "nvidia/nemotron-3-nano-4b-v1.1"
DEFAULT_LORA_RANK = 16


class CustomizerClient:
    """REST client for the NeMo Customizer API (v2).

    Base URL is read from ``NEMOCODE_CUSTOMIZER_BASE_URL`` (default:
    ``http://localhost:8000``).  Auth uses ``NVIDIA_API_KEY`` when set.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self._base_url = (
            base_url
            or os.environ.get("NEMOCODE_CUSTOMIZER_BASE_URL", DEFAULT_CUSTOMIZER_BASE_URL)
        ).rstrip("/")
        self._api_key = api_key or os.environ.get("NVIDIA_API_KEY")
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _handle_response(self, resp: httpx.Response) -> dict[str, Any]:
        """Raise on HTTP errors, return parsed JSON."""
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise CustomizerAPIError(resp.status_code, detail)
        return resp.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_job(
        self,
        dataset_path: str,
        *,
        model: str = DEFAULT_MODEL,
        lora_rank: int = DEFAULT_LORA_RANK,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        output_model: str | None = None,
    ) -> dict[str, Any]:
        """Submit a new LoRA fine-tuning job.

        ``dataset_path`` should point to a JSONL file on the server's filesystem
        (or accessible via the customizer's data volume).
        """
        body: dict[str, Any] = {
            "model": model,
            "training_data": str(Path(dataset_path)),
            "hyperparameters": {
                "lora_rank": lora_rank,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            },
        }
        if output_model:
            body["output_model"] = output_model

        with httpx.Client(timeout=self._timeout) as client:
            resp = client.post(
                self._url("/v2/nemo/customization/jobs"),
                json=body,
                headers=self._headers(),
            )
        return self._handle_response(resp)

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all customization jobs."""
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                self._url("/v2/nemo/customization/jobs"),
                headers=self._headers(),
            )
        data = self._handle_response(resp)
        # The API may return {"jobs": [...]} or a bare list
        if isinstance(data, list):
            return data
        return data.get("jobs", [])

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Get the status of a customization job."""
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                self._url(f"/v2/nemo/customization/jobs/{job_id}"),
                headers=self._headers(),
            )
        return self._handle_response(resp)

    def get_logs(self, job_id: str) -> dict[str, Any]:
        """Get training logs for a customization job."""
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                self._url(f"/v2/nemo/customization/jobs/{job_id}/logs"),
                headers=self._headers(),
            )
        return self._handle_response(resp)

    def get_results(self, job_id: str) -> dict[str, Any]:
        """Get results / download info for a completed customization job."""
        with httpx.Client(timeout=self._timeout) as client:
            resp = client.get(
                self._url(f"/v2/nemo/customization/jobs/{job_id}/results"),
                headers=self._headers(),
            )
        return self._handle_response(resp)


class CustomizerAPIError(Exception):
    """Raised when the Customizer API returns an error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")
