"""Google Health API read-only tools.

This module registers Hermes tools for querying user health data from the
Google Health API. Authentication uses Google OAuth 2.0 credentials and a
refresh token provided out-of-band via environment variables or a local token
file. The tools are intentionally read-only; they only call documented GET
endpoints.

Credential sources, checked at call time:
- ``GOOGLE_HEALTH_TOKEN_FILE`` JSON with client_id, client_secret, refresh_token
  (default: ``$HERMES_HOME/google_health_token.json``)
- or env vars: ``GOOGLE_HEALTH_CLIENT_ID``, ``GOOGLE_HEALTH_CLIENT_SECRET``,
  ``GOOGLE_HEALTH_REFRESH_TOKEN``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from pydantic import BaseModel, Field, ValidationError, field_validator

from hermes_constants import get_hermes_home
from tools.registry import registry, tool_error

logger = logging.getLogger(__name__)

GOOGLE_HEALTH_BASE_URL = "https://health.googleapis.com/v4"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
DEFAULT_TIMEOUT_SECONDS = 30
DATA_TYPE_RE = re.compile(r"^[a-z][a-z0-9-]*$")

READONLY_SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
    "https://www.googleapis.com/auth/googlehealth.location.readonly",
    "https://www.googleapis.com/auth/googlehealth.nutrition.readonly",
    "https://www.googleapis.com/auth/googlehealth.profile.readonly",
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
]


class GoogleHealthCredentials(BaseModel):
    """OAuth credentials loaded from env vars or a local token file."""

    client_id: str
    client_secret: str
    refresh_token: str
    access_token: str | None = None
    expires_at: float | None = None
    token_file: Path | None = None

    @field_validator("client_id", "client_secret", "refresh_token")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must not be empty")
        return value.strip()

    def has_fresh_access_token(self, skew_seconds: int = 60) -> bool:
        if not self.access_token:
            return False
        if not self.expires_at:
            return True
        return self.expires_at > time.time() + skew_seconds


class ListDataPointsRequest(BaseModel):
    """External boundary model for google_health_list_data_points."""

    data_type: str = Field(description="Google Health API data type in endpoint/kebab-case form, e.g. steps, weight, sleep")
    filter: str | None = Field(default=None, description="Optional AIP-160 filter expression from Google Health API docs")
    page_size: int | None = Field(default=None, ge=1, le=10000)
    page_token: str | None = None

    @field_validator("data_type")
    @classmethod
    def _valid_data_type(cls, value: str) -> str:
        value = value.strip()
        if not DATA_TYPE_RE.fullmatch(value):
            raise ValueError("data_type must be a kebab-case identifier such as 'steps' or 'heart-rate'")
        return value


class LatestDataPointsRequest(BaseModel):
    """External boundary model for google_health_latest_data_points."""

    data_type: str
    filter: str | None = None
    limit: int = Field(default=10, ge=1, le=100)
    page_size: int = Field(default=100, ge=1, le=10000)

    @field_validator("data_type")
    @classmethod
    def _valid_data_type(cls, value: str) -> str:
        value = value.strip()
        if not DATA_TYPE_RE.fullmatch(value):
            raise ValueError("data_type must be a kebab-case identifier such as 'steps' or 'heart-rate'")
        return value


def _default_token_file() -> Path:
    return Path(get_hermes_home()) / "google_health_token.json"


def _configured_token_file() -> Path:
    raw = os.getenv("GOOGLE_HEALTH_TOKEN_FILE")
    if raw:
        return Path(raw).expanduser()
    return _default_token_file()


def _load_credentials() -> GoogleHealthCredentials:
    token_file = _configured_token_file()
    file_data: dict[str, Any] = {}
    if token_file.exists():
        try:
            file_data = json.loads(token_file.read_text())
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in GOOGLE_HEALTH_TOKEN_FILE {token_file}: {exc}") from exc

    data = {
        "client_id": os.getenv("GOOGLE_HEALTH_CLIENT_ID") or file_data.get("client_id"),
        "client_secret": os.getenv("GOOGLE_HEALTH_CLIENT_SECRET") or file_data.get("client_secret"),
        "refresh_token": os.getenv("GOOGLE_HEALTH_REFRESH_TOKEN") or file_data.get("refresh_token"),
        "access_token": os.getenv("GOOGLE_HEALTH_ACCESS_TOKEN") or file_data.get("access_token"),
        "expires_at": file_data.get("expires_at"),
        "token_file": token_file if token_file.exists() else None,
    }
    try:
        return GoogleHealthCredentials.model_validate(data)
    except ValidationError as exc:
        raise ValueError(
            "Google Health API credentials are not configured. Set "
            "GOOGLE_HEALTH_CLIENT_ID, GOOGLE_HEALTH_CLIENT_SECRET, and "
            "GOOGLE_HEALTH_REFRESH_TOKEN, or create ~/.hermes/google_health_token.json."
        ) from exc


def _save_refreshed_token(creds: GoogleHealthCredentials) -> None:
    """Persist refreshed access token only when credentials came from a file."""
    if not creds.token_file:
        return
    payload = {
        "client_id": creds.client_id,
        "client_secret": creds.client_secret,
        "refresh_token": creds.refresh_token,
        "access_token": creds.access_token,
        "expires_at": creds.expires_at,
        "scopes": READONLY_SCOPES,
    }
    creds.token_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = creds.token_file.with_suffix(creds.token_file.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(creds.token_file)


def _refresh_access_token(creds: GoogleHealthCredentials) -> str:
    if creds.has_fresh_access_token():
        return str(creds.access_token)

    response = requests.post(
        GOOGLE_TOKEN_URL,
        data={
            "client_id": creds.client_id,
            "client_secret": creds.client_secret,
            "refresh_token": creds.refresh_token,
            "grant_type": "refresh_token",
        },
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Token refresh failed: HTTP {response.status_code}: {response.text[:500]}")
    payload = response.json()
    access_token = payload.get("access_token")
    if not access_token:
        raise RuntimeError("Token refresh response did not contain access_token")

    expires_in = int(payload.get("expires_in") or 3600)
    creds.access_token = access_token
    creds.expires_at = time.time() + expires_in
    _save_refreshed_token(creds)
    return access_token


def _health_get(path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    creds = _load_credentials()
    token = _refresh_access_token(creds)
    url = f"{GOOGLE_HEALTH_BASE_URL}/{path.lstrip('/')}"
    response = requests.get(
        url,
        headers={"Authorization": f"Bearer {token}", "Accept": "application/json"},
        params={k: v for k, v in (params or {}).items() if v not in (None, "")},
        timeout=DEFAULT_TIMEOUT_SECONDS,
    )
    if response.status_code >= 400:
        raise RuntimeError(f"Google Health API request failed: HTTP {response.status_code}: {response.text[:1000]}")
    return response.json()


def _extract_observation_time(data_point: dict[str, Any]) -> str:
    """Best-effort sort key over documented DataPoint time shapes."""
    candidates = [
        data_point.get("sampleTime", {}).get("physicalTime"),
        data_point.get("sample_time", {}).get("physical_time"),
        data_point.get("interval", {}).get("endTime"),
        data_point.get("interval", {}).get("end_time"),
        data_point.get("interval", {}).get("startTime"),
        data_point.get("interval", {}).get("start_time"),
        data_point.get("date"),
    ]
    for value in candidates:
        if isinstance(value, str) and value:
            return value
    return ""


def _credential_status() -> dict[str, Any]:
    token_file = _configured_token_file()
    env_configured = all(
        os.getenv(name)
        for name in ("GOOGLE_HEALTH_CLIENT_ID", "GOOGLE_HEALTH_CLIENT_SECRET", "GOOGLE_HEALTH_REFRESH_TOKEN")
    )
    file_configured = token_file.exists()
    return {
        "configured": env_configured or file_configured,
        "env_configured": env_configured,
        "token_file": str(token_file),
        "token_file_exists": file_configured,
        "base_url": GOOGLE_HEALTH_BASE_URL,
        "readonly_scopes": READONLY_SCOPES,
        "setup_script": "scripts/google_health_oauth.py",
    }


def _handle_status(args: dict, **kw: Any) -> str:
    return json.dumps({"result": _credential_status()})


def _handle_list_data_points(args: dict, **kw: Any) -> str:
    try:
        request = ListDataPointsRequest.model_validate(args)
        result = _health_get(
            f"users/me/dataTypes/{request.data_type}/dataPoints",
            params={
                "pageSize": request.page_size,
                "pageToken": request.page_token,
                "filter": request.filter,
            },
        )
        return json.dumps({"result": result})
    except (ValidationError, ValueError) as exc:
        return tool_error(str(exc))
    except Exception as exc:
        logger.exception("google_health_list_data_points failed")
        return tool_error(str(exc))


def _handle_latest_data_points(args: dict, **kw: Any) -> str:
    try:
        request = LatestDataPointsRequest.model_validate(args)
        result = _health_get(
            f"users/me/dataTypes/{request.data_type}/dataPoints",
            params={"pageSize": request.page_size, "filter": request.filter},
        )
        points = list(result.get("dataPoints") or [])
        points.sort(key=_extract_observation_time, reverse=True)
        return json.dumps(
            {
                "result": {
                    "data_type": request.data_type,
                    "count_returned_by_api": len(result.get("dataPoints") or []),
                    "next_page_token": result.get("nextPageToken"),
                    "data_points": points[: request.limit],
                }
            }
        )
    except (ValidationError, ValueError) as exc:
        return tool_error(str(exc))
    except Exception as exc:
        logger.exception("google_health_latest_data_points failed")
        return tool_error(str(exc))


def _check_google_health_available() -> bool:
    try:
        _load_credentials()
        return True
    except Exception:
        return False


GOOGLE_HEALTH_STATUS_SCHEMA = {
    "name": "google_health_status",
    "description": "Check local Google Health API credential configuration and documented read-only OAuth scopes without reading health data.",
    "parameters": {"type": "object", "properties": {}, "required": []},
}

GOOGLE_HEALTH_LIST_DATA_POINTS_SCHEMA = {
    "name": "google_health_list_data_points",
    "description": (
        "Read Google Health API data points for a documented data type. "
        "This is a raw read-only wrapper around GET /v4/users/me/dataTypes/{dataType}/dataPoints. "
        "Pass data_type in kebab-case, e.g. steps, weight, heart-rate, sleep. "
        "Use filter only with exact fields documented by Google Health API."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_type": {"type": "string", "description": "Data type in endpoint/kebab-case form, e.g. steps, weight, sleep."},
            "filter": {"type": "string", "description": "Optional AIP-160 filter expression from Google Health API docs."},
            "page_size": {"type": "integer", "description": "Optional max data points to return; Google caps at 10000 and caps exercise/sleep at 25."},
            "page_token": {"type": "string", "description": "nextPageToken from a previous response."},
        },
        "required": ["data_type"],
    },
}

GOOGLE_HEALTH_LATEST_DATA_POINTS_SCHEMA = {
    "name": "google_health_latest_data_points",
    "description": (
        "Return the latest Google Health API data points for a documented data type by sorting returned points by observed time. "
        "Use an explicit Google Health API filter to bound the query when possible."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "data_type": {"type": "string", "description": "Data type in endpoint/kebab-case form, e.g. steps, weight, sleep."},
            "filter": {"type": "string", "description": "Optional AIP-160 filter expression from Google Health API docs."},
            "limit": {"type": "integer", "description": "Number of latest points to return, 1-100. Default 10."},
            "page_size": {"type": "integer", "description": "Page size for the underlying Google Health API list call. Default 100."},
        },
        "required": ["data_type"],
    },
}

registry.register(
    name="google_health_status",
    toolset="google_health",
    schema=GOOGLE_HEALTH_STATUS_SCHEMA,
    handler=_handle_status,
    check_fn=lambda: True,
    emoji="🩺",
)

registry.register(
    name="google_health_list_data_points",
    toolset="google_health",
    schema=GOOGLE_HEALTH_LIST_DATA_POINTS_SCHEMA,
    handler=_handle_list_data_points,
    check_fn=_check_google_health_available,
    emoji="🩺",
)

registry.register(
    name="google_health_latest_data_points",
    toolset="google_health",
    schema=GOOGLE_HEALTH_LATEST_DATA_POINTS_SCHEMA,
    handler=_handle_latest_data_points,
    check_fn=_check_google_health_available,
    emoji="🩺",
)
