import json
import time

import tools.google_health_tool as google_health


class DummyResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def test_check_available_from_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_HEALTH_CLIENT_ID", "client")
    monkeypatch.setenv("GOOGLE_HEALTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_HEALTH_REFRESH_TOKEN", "refresh")

    assert google_health._check_google_health_available() is True


def test_check_unavailable_without_credentials(monkeypatch, tmp_path):
    monkeypatch.delenv("GOOGLE_HEALTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_HEALTH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GOOGLE_HEALTH_REFRESH_TOKEN", raising=False)
    monkeypatch.setenv("GOOGLE_HEALTH_TOKEN_FILE", str(tmp_path / "missing.json"))

    assert google_health._check_google_health_available() is False


def test_list_data_points_refreshes_token_and_calls_documented_endpoint(monkeypatch):
    monkeypatch.setenv("GOOGLE_HEALTH_CLIENT_ID", "client")
    monkeypatch.setenv("GOOGLE_HEALTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_HEALTH_REFRESH_TOKEN", "refresh")
    calls = []

    def fake_post(url, data, timeout):
        calls.append(("post", url, data, timeout))
        return DummyResponse(payload={"access_token": "access", "expires_in": 3600})

    def fake_get(url, headers, params, timeout):
        calls.append(("get", url, headers, params, timeout))
        return DummyResponse(payload={"dataPoints": [{"date": "2026-05-25"}], "nextPageToken": ""})

    monkeypatch.setattr(google_health.requests, "post", fake_post)
    monkeypatch.setattr(google_health.requests, "get", fake_get)

    raw = google_health._handle_list_data_points(
        {"data_type": "weight", "page_size": 10, "filter": 'weight.sample_time.physical_time >= "2026-05-01T00:00:00Z"'}
    )
    result = json.loads(raw)

    assert result["result"]["dataPoints"] == [{"date": "2026-05-25"}]
    assert calls[0][0] == "post"
    assert calls[0][1] == google_health.GOOGLE_TOKEN_URL
    assert calls[1][0] == "get"
    assert calls[1][1] == "https://health.googleapis.com/v4/users/me/dataTypes/weight/dataPoints"
    assert calls[1][2]["Authorization"] == "Bearer access"
    assert calls[1][3]["pageSize"] == 10
    assert "filter" in calls[1][3]


def test_rejects_invalid_data_type_before_http(monkeypatch):
    def fail_get(*args, **kwargs):
        raise AssertionError("HTTP should not be called for invalid data_type")

    monkeypatch.setattr(google_health.requests, "get", fail_get)

    raw = google_health._handle_list_data_points({"data_type": "../weight"})
    result = json.loads(raw)

    assert "error" in result
    assert "data_type" in result["error"]


def test_latest_sorts_by_observation_time(monkeypatch):
    monkeypatch.setenv("GOOGLE_HEALTH_CLIENT_ID", "client")
    monkeypatch.setenv("GOOGLE_HEALTH_CLIENT_SECRET", "secret")
    monkeypatch.setenv("GOOGLE_HEALTH_REFRESH_TOKEN", "refresh")

    def fake_post(url, data, timeout):
        return DummyResponse(payload={"access_token": "access", "expires_in": 3600})

    def fake_get(url, headers, params, timeout):
        return DummyResponse(
            payload={
                "dataPoints": [
                    {"date": "2026-05-20", "value": 1},
                    {"sampleTime": {"physicalTime": "2026-05-25T12:00:00Z"}, "value": 2},
                    {"interval": {"endTime": "2026-05-24T12:00:00Z"}, "value": 3},
                ]
            }
        )

    monkeypatch.setattr(google_health.requests, "post", fake_post)
    monkeypatch.setattr(google_health.requests, "get", fake_get)

    raw = google_health._handle_latest_data_points({"data_type": "weight", "limit": 2})
    result = json.loads(raw)["result"]

    assert [point["value"] for point in result["data_points"]] == [2, 3]
    assert result["count_returned_by_api"] == 3
