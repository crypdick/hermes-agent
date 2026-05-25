#!/usr/bin/env python3
"""Create a local Google Health API OAuth token file for Hermes.

Usage:
  uv run python scripts/google_health_oauth.py --client-id CLIENT_ID

The client secret is read from GOOGLE_HEALTH_CLIENT_SECRET or prompted with hidden input.
By default this follows the Google Health API docs' web OAuth pattern using
https://www.google.com as redirect URI: open the printed URL, approve access,
then paste the final redirected URL back into the prompt. The script exchanges
the authorization code for tokens and writes ~/.hermes/google_health_token.json
with mode 0600.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import stat
import time
import urllib.parse
from pathlib import Path

import requests

DEFAULT_SCOPES = [
    "https://www.googleapis.com/auth/googlehealth.activity_and_fitness.readonly",
    "https://www.googleapis.com/auth/googlehealth.health_metrics_and_measurements.readonly",
    "https://www.googleapis.com/auth/googlehealth.location.readonly",
    "https://www.googleapis.com/auth/googlehealth.nutrition.readonly",
    "https://www.googleapis.com/auth/googlehealth.profile.readonly",
    "https://www.googleapis.com/auth/googlehealth.sleep.readonly",
]
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"


def default_output_path() -> Path:
    hermes_home = Path(os.getenv("HERMES_HOME", Path.home() / ".hermes"))
    return hermes_home / "google_health_token.json"


def extract_code(value: str) -> str:
    value = value.strip()
    parsed = urllib.parse.urlparse(value)
    if parsed.query:
        query = urllib.parse.parse_qs(parsed.query)
        codes = query.get("code")
        if codes:
            return codes[0]
    return value


def write_private_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.chmod(tmp, stat.S_IRUSR | stat.S_IWUSR)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Authorize Hermes for Google Health API read-only access")
    parser.add_argument("--client-id", required=True, help="OAuth 2.0 client ID from Google Cloud Console")
    parser.add_argument(
        "--client-secret",
        help="OAuth 2.0 client secret. Prefer GOOGLE_HEALTH_CLIENT_SECRET or the hidden prompt to avoid shell history.",
    )
    parser.add_argument("--redirect-uri", default="https://www.google.com", help="Authorized redirect URI configured in Google Cloud")
    parser.add_argument("--output", type=Path, default=default_output_path(), help="Token JSON output path")
    parser.add_argument("--scope", action="append", dest="scopes", help="OAuth scope to request; repeat to override defaults")
    args = parser.parse_args()

    scopes = args.scopes or DEFAULT_SCOPES
    client_secret = args.client_secret or os.getenv("GOOGLE_HEALTH_CLIENT_SECRET")
    if not client_secret:
        client_secret = getpass.getpass("OAuth client secret: ")
    if not client_secret.strip():
        print("OAuth client secret is required.")
        return 1

    params = {
        "client_id": args.client_id,
        "redirect_uri": args.redirect_uri,
        "response_type": "code",
        "scope": " ".join(scopes),
        "access_type": "offline",
        "prompt": "consent",
        "include_granted_scopes": "true",
    }
    authorization_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
    print("Open this URL in a browser and approve Google Health API access:\n")
    print(authorization_url)
    print("\nAfter approval, paste the full redirected URL or just the code parameter.")
    code = extract_code(input("Authorization code or URL: "))

    response = requests.post(
        TOKEN_URL,
        data={
            "client_id": args.client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": args.redirect_uri,
            "grant_type": "authorization_code",
        },
        timeout=30,
    )
    if response.status_code >= 400:
        print(f"Token exchange failed: HTTP {response.status_code}")
        print(response.text)
        return 1

    token = response.json()
    refresh_token = token.get("refresh_token")
    if not refresh_token:
        print("Token response did not include refresh_token. Re-run with prompt=consent or revoke/re-approve access.")
        print(json.dumps({k: v for k, v in token.items() if k != "access_token"}, indent=2))
        return 1

    payload = {
        "client_id": args.client_id,
        "client_secret": client_secret,
        "refresh_token": refresh_token,
        "access_token": token.get("access_token"),
        "expires_at": time.time() + int(token.get("expires_in") or 3600),
        "scopes": scopes,
        "token_type": token.get("token_type"),
    }
    write_private_json(args.output.expanduser(), payload)
    print(f"Wrote Google Health API token file: {args.output.expanduser()}")
    print("Restart Hermes/gateway or start a new session so the google_health tools are discovered with credentials.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
