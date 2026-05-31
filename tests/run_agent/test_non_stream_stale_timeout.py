"""Regression tests for non-streaming stale timeout behavior."""

import threading
from types import SimpleNamespace

from run_agent import AIAgent
from agent.chat_completion_helpers import interruptible_api_call


def _fake_agent():
    def stale_base():
        return 300.0, True

    return SimpleNamespace(
        provider="openai-codex",
        model="gpt-5.5",
        _base_url="https://chatgpt.com/backend-api/codex",
        base_url=None,
        _resolved_api_call_stale_timeout_base=stale_base,
    )


def _codex_kwargs(chars: int) -> dict:
    return {
        "model": "gpt-5.5",
        "instructions": "system prompt",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "x" * chars},
                ],
            }
        ],
        "store": False,
    }


def test_chat_completions_messages_still_count_toward_non_stream_stale_timeout():
    """The transport-shape fix must preserve chat-completions scaling."""
    agent = _fake_agent()
    kwargs = {
        "model": "gpt-5.5",
        "messages": [{"role": "user", "content": "x" * 240_000}],
    }

    timeout = AIAgent._compute_non_stream_stale_timeout(agent, kwargs)  # type: ignore[arg-type]

    assert timeout == 450.0


def test_codex_responses_input_counts_toward_non_stream_stale_timeout():
    """Large Responses API payloads must not be treated as context~=0."""
    agent = _fake_agent()

    timeout = AIAgent._compute_non_stream_stale_timeout(agent, _codex_kwargs(240_000))  # type: ignore[arg-type]

    assert timeout == 450.0


def test_codex_responses_input_count_is_available_for_stale_logging():
    """The same estimator used for timeout scaling should feed context logging."""
    tokens = AIAgent._estimate_non_stream_context_tokens(_codex_kwargs(240_000))

    assert tokens > 50_000


def test_codex_responses_stream_activity_prevents_outer_non_stream_stale_timeout(monkeypatch):
    """Codex streams internally; provider events must reset the outer stale timer."""
    import agent.chat_completion_helpers as helpers

    events_seen = []
    statuses = []
    fake_now = {"value": 1_000.0}
    monkeypatch.setattr(helpers.time, "time", lambda: fake_now["value"])

    def run_codex_stream(api_kwargs, client=None, on_first_delta=None, on_stream_event=None):
        # Advance fake provider time beyond the stale threshold while emitting
        # stream activity. This avoids wall-clock races while still proving that
        # Codex SSE events, not only final response completion, keep the outer
        # non-streaming wrapper alive.
        for _ in range(4):
            fake_now["value"] += 0.14
            on_stream_event()
            events_seen.append(fake_now["value"])
        threading.Event().wait(0.5)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    agent = SimpleNamespace(
        api_mode="codex_responses",
        _codex_on_first_delta=None,
        _interrupt_requested=False,
        _create_request_openai_client=lambda **kwargs: SimpleNamespace(close=lambda: None),
        _close_request_openai_client=lambda *args, **kwargs: None,
        _run_codex_stream=run_codex_stream,
        _compute_non_stream_stale_timeout=lambda payload: 0.25,
        _estimate_non_stream_context_tokens=AIAgent._estimate_non_stream_context_tokens,
        _touch_activity=lambda desc: None,
        _emit_status=statuses.append,
    )

    response = interruptible_api_call(agent, _codex_kwargs(1_000))

    assert response.choices[0].message.content == "ok"
    assert len(events_seen) == 4
    assert events_seen[-1] - 1_000.0 > 0.25
    assert statuses == []
