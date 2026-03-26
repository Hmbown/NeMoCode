# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""CustomizerClient — REST client for NVIDIA NeMo Customizer.

Communicates with the NeMo Customizer service to create, monitor, and retrieve
fine-tuning jobs for Nemotron 3 Nano 4B.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_CUSTOMIZER_BASE_URL = "http://localhost:8000"
DEFAULT_MODEL = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"
DEFAULT_LORA_RANK = 16
DEFAULT_TRAINING_TYPE = "sft"
DEFAULT_FINETUNING_TYPE = "lora"
DEFAULT_LORA_DROPOUT = 0.01


def _resolve_api_key(explicit_api_key: str | None) -> str | None:
    if explicit_api_key:
        return explicit_api_key

    for env_name in ("NGC_CLI_API_KEY", "NVIDIA_API_KEY", "NIM_API_KEY"):
        value = os.environ.get(env_name)
        if value:
            return value
    return None


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
        self._api_key = _resolve_api_key(api_key)
        self._timeout = timeout

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _headers(self, extra_headers: dict[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        if extra_headers:
            headers.update(extra_headers)
        return headers

    def _url(self, path: str) -> str:
        return f"{self._base_url}{path}"

    def _handle_response(self, resp: httpx.Response) -> dict[str, Any] | list[Any]:
        """Raise on HTTP errors, return parsed JSON."""
        if resp.status_code >= 400:
            detail = resp.text
            try:
                detail = resp.json().get("detail", detail)
            except Exception:
                pass
            raise CustomizerAPIError(resp.status_code, detail)
        return resp.json()

    def _request_json(
        self,
        method: str,
        paths: list[str],
        *,
        extra_headers: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any] | list[Any]:
        """Try one or more API paths and return parsed JSON.

        The NeMo platform and older local wrappers expose slightly different
        route prefixes, so we probe the modern route first and fall back to the
        legacy path when needed.
        """
        last_resp: httpx.Response | None = None
        with httpx.Client(timeout=self._timeout) as client:
            for path in paths:
                request_kwargs = {
                    "headers": self._headers(extra_headers),
                    **kwargs,
                }
                if method.upper() == "GET":
                    resp = client.get(self._url(path), **request_kwargs)
                elif method.upper() == "POST":
                    resp = client.post(self._url(path), **request_kwargs)
                elif method.upper() == "DELETE":
                    resp = client.delete(self._url(path), **request_kwargs)
                else:
                    resp = client.request(method, self._url(path), **request_kwargs)
                last_resp = resp
                if resp.status_code == 404 and path != paths[-1]:
                    continue
                return self._handle_response(resp)

        if last_resp is None:
            raise RuntimeError("No response received from the Customizer service.")
        return self._handle_response(last_resp)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_job(
        self,
        dataset_path: str | None = None,
        *,
        model: str = DEFAULT_MODEL,
        lora_rank: int = DEFAULT_LORA_RANK,
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 1e-4,
        output_model: str | None = None,
        config: str | None = None,
        dataset_name: str | None = None,
        dataset_namespace: str = "default",
        training_type: str = DEFAULT_TRAINING_TYPE,
        finetuning_type: str = DEFAULT_FINETUNING_TYPE,
        max_seq_length: int | None = None,
        lora_dropout: float = DEFAULT_LORA_DROPOUT,
        wandb_api_key: str | None = None,
    ) -> dict[str, Any]:
        """Submit a new customization job.

        Preferred platform mode uses ``config`` plus a Data Store dataset entity
        via ``dataset_name`` / ``dataset_namespace`` and calls the modern
        ``/v1/customization/jobs`` API.

        Legacy local-dev mode accepts ``dataset_path`` and calls the older
        local-file job endpoint.
        """
        if config and dataset_name:
            hyperparameters: dict[str, Any] = {
                "training_type": training_type,
                "finetuning_type": finetuning_type,
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
            }
            if max_seq_length is not None:
                hyperparameters["max_seq_length"] = max_seq_length
            if finetuning_type == "lora":
                hyperparameters["lora"] = {
                    "adapter_dim": lora_rank,
                    "adapter_dropout": lora_dropout,
                }

            extra_headers: dict[str, str] = {}
            if wandb_api_key:
                extra_headers["wandb-api-key"] = wandb_api_key

            body = {
                "config": config,
                "dataset": {
                    "name": dataset_name,
                    "namespace": dataset_namespace,
                },
                "hyperparameters": hyperparameters,
            }
            return self._request_json(
                "POST",
                ["/v1/customization/jobs"],
                json=body,
                extra_headers=extra_headers,
            )

        if not dataset_path:
            raise ValueError("Provide either dataset_path or config + dataset_name.")

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

        return self._request_json(
            "POST",
            ["/v2/nemo/customization/jobs"],
            json=body,
        )

    def list_configs(
        self,
        *,
        base_model: str | None = None,
        training_type: str | None = DEFAULT_TRAINING_TYPE,
        finetuning_type: str | None = DEFAULT_FINETUNING_TYPE,
        enabled: bool = True,
        page: int = 1,
        page_size: int = 100,
    ) -> list[dict[str, Any]]:
        """List available customization configs from the platform API."""
        params: dict[str, Any] = {
            "page": page,
            "page_size": page_size,
            "sort": "-created_at",
        }
        if base_model:
            params["filter[base_model]"] = base_model
        if training_type:
            params["filter[training_type]"] = training_type
        if finetuning_type:
            params["filter[finetuning_type]"] = finetuning_type
        params["filter[enabled]"] = str(enabled).lower()

        data = self._request_json(
            "GET",
            ["/v1/customization/configs"],
            params=params,
        )
        if isinstance(data, list):
            return data
        return data.get("data", [])

    def list_jobs(self) -> list[dict[str, Any]]:
        """List all customization jobs."""
        data = self._request_json(
            "GET",
            ["/v1/customization/jobs", "/v2/nemo/customization/jobs"],
        )
        # The API may return {"jobs": [...]} or a bare list
        if isinstance(data, list):
            return data
        if "data" in data:
            return data.get("data", [])
        return data.get("jobs", [])

    def get_status(self, job_id: str) -> dict[str, Any]:
        """Get the status of a customization job."""
        return self._request_json(
            "GET",
            [
                f"/v1/customization/jobs/{job_id}",
                f"/v2/nemo/customization/jobs/{job_id}",
            ],
        )

    def get_logs(self, job_id: str) -> dict[str, Any]:
        """Get training logs for a customization job."""
        return self._request_json(
            "GET",
            [
                f"/v1/customization/jobs/{job_id}/logs",
                f"/v2/nemo/customization/jobs/{job_id}/logs",
            ],
        )

    def get_results(self, job_id: str) -> dict[str, Any]:
        """Get results / download info for a completed customization job."""
        return self._request_json(
            "GET",
            [
                f"/v1/customization/jobs/{job_id}/results",
                f"/v2/nemo/customization/jobs/{job_id}/results",
            ],
        )


class CustomizerAPIError(Exception):
    """Raised when the Customizer API returns an error response."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"HTTP {status_code}: {detail}")
