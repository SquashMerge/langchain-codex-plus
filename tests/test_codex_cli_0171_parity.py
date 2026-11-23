"""Parity with Codex CLI 0.171.0 (from 0.167.1).

Each test pins one difference between the two Codex releases that changes
what this library sends or how it reads a response:

* the client version it presents (``?client_version=`` and User-Agent);
* an HTTP error's code is read from ``detail.code``, then ``error.code``,
  then the base64 ``x-error-json`` header (upstream ``api_bridge.rs``,
  bb360e5db8, 4b345a8de9, 5b98eaafeb), so 403s such as
  ``invalid_workspace_selected`` are identifiable;
* terminal provider codes end the stream at once, even when a
  ``response.completed`` follows (upstream ``error.rs`` / ``sse/responses.rs``,
  0811df3976, 1885417cd4, d9446b4b01, 65f2709052, 0ed9d6fa4b). The library
  already stopped on the first error event; these tests pin it for every code
  0.171 makes terminal.

The ChatGPT token lifecycle (auth.json schema, refresh triggers, refresh
request and error classification, request auth headers) is unchanged between
the two releases, so it has no test here; ``test_codex_cli_0167_parity.py``
still covers it.
"""

from __future__ import annotations

import base64
import json

import pytest
from langchain_core.messages import HumanMessage

from langchain_codex_plus.codex_protocol import (
    CodexResponseError,
    SseEvent,
    consume_events,
    error_code_from_x_error_json,
    parse_error_body,
)
from tests.conftest import (
    _AsyncCaptureTransport,
    _CaptureTransport,
    _make_llm,
    _sse_bytes,
)

# ─── Presented client version ──────────────────────────────────────────


def test_presents_codex_cli_0171(auth_file, monkeypatch):
    monkeypatch.delenv("CODEX_PLUS_CLIENT_VERSION", raising=False)
    transport = _CaptureTransport(body=_sse_bytes([
        ("response.created", {"response": {"id": "r"}}),
        ("response.completed", {"response": {"id": "r", "status": "completed"}}),
    ]))
    llm = _make_llm(auth_file, transport=transport)
    assert llm._request_url().endswith("?client_version=0.171.0")
    llm.invoke([HumanMessage("hi")])
    sent = transport.last_request
    assert sent.headers["user-agent"].startswith("codex_cli_rs/0.171.0")
    assert sent.url.params["client_version"] == "0.171.0"


# ─── HTTP error code: detail.code, error.code, x-error-json ────────────


def _x_error_json(code: str) -> str:
    return base64.b64encode(json.dumps({"error": {"code": code}}).encode()).decode()


def test_structured_detail_supplies_code_and_message():
    err = parse_error_body(
        b'{"detail": {"code": "invalid_workspace_selected", "message": "pick a workspace"}}'
    )
    assert err.code == "invalid_workspace_selected"
    assert err.message == "pick a workspace"


def test_detail_code_wins_over_error_code():
    err = parse_error_body(json.dumps({
        "detail": {"code": "codex_workspace_access_denied"},
        "error": {"code": "other", "message": "m"},
    }).encode())
    assert err.code == "codex_workspace_access_denied"


def test_string_detail_keeps_its_message():
    err = parse_error_body(b'{"detail": "Unsupported content type"}')
    assert err.message == "Unsupported content type"
    assert err.code is None


@pytest.mark.parametrize("body", [b"{}", b"not json", b'{"detail": "denied"}'])
def test_x_error_json_supplies_code_when_body_has_none(body):
    headers = {"X-Error-Json": _x_error_json("access_program_not_enabled")}
    assert parse_error_body(body, headers).code == "access_program_not_enabled"


def test_body_code_wins_over_x_error_json():
    headers = {"x-error-json": _x_error_json("from_header")}
    err = parse_error_body(b'{"error": {"code": "from_body", "message": "m"}}', headers)
    assert err.code == "from_body"


@pytest.mark.parametrize("value", ["", "%%%not-base64", base64.b64encode(b"[1]").decode(),
                                   base64.b64encode(b'{"error": {"code": 7}}').decode()])
def test_malformed_x_error_json_is_ignored(value):
    assert error_code_from_x_error_json({"x-error-json": value}) is None
    assert parse_error_body(b"{}", {"x-error-json": value}).code is None


def test_http_403_carries_x_error_json_code(auth_file):
    transport = _CaptureTransport(
        status_code=403, body=b"{}",
        headers={"x-error-json": _x_error_json("codex_workspace_access_denied")},
    )
    llm = _make_llm(auth_file, transport=transport)
    with pytest.raises(CodexResponseError) as caught:
        llm.invoke([HumanMessage("hi")])
    assert caught.value.status_code == 403
    assert caught.value.code == "codex_workspace_access_denied"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "headers", "code"),
    [
        (b'{"detail": {"code": "invalid_workspace_selected", "message": "m"}}', {},
         "invalid_workspace_selected"),
        (b"{}", {"x-error-json": _x_error_json("codex_workspace_access_denied")},
         "codex_workspace_access_denied"),
    ],
    ids=["detail_code", "x_error_json"],
)
async def test_async_http_403_carries_code(auth_file, body, headers, code):
    transport = _AsyncCaptureTransport(status_code=403, body=body, headers=headers)
    llm = _make_llm(auth_file, transport=transport)
    with pytest.raises(CodexResponseError) as caught:
        await llm.ainvoke([HumanMessage("hi")])
    assert caught.value.status_code == 403
    assert caught.value.code == code


# ─── Terminal provider codes end the stream ────────────────────────────

_TERMINAL_CODES = [
    "flex_unavailable",
    "inference_hook_denied",
    "context_length_exceeded",
    "invalid_access_program",
    "unsupported_access_program",
    "content_policy_violation",
    "access_program_not_enabled",
    "budget_reached",
]


def _terminal(kind: str, code: str):
    if kind == "error":
        return ("error", {"type": "error", "code": code, "message": f"{code} happened"})
    return ("response.failed", {
        "type": "response.failed",
        "response": {"id": "r", "status": "failed",
                     "error": {"code": code, "message": f"{code} happened"}},
    })


def _stream(kind: str, code: str):
    # A response.completed after the failure must not win: 0.171 ends the
    # stream on the terminal event and ignores what follows.
    return [
        ("response.created", {"response": {"id": "r"}}),
        _terminal(kind, code),
        ("response.output_text.delta", {"delta": "late"}),
        ("response.completed", {"response": {"id": "r", "status": "completed"}}),
    ]


@pytest.mark.parametrize("kind", ["error", "response.failed"])
@pytest.mark.parametrize("code", _TERMINAL_CODES)
def test_terminal_code_ends_consume_events(kind, code):
    events = [SseEvent(event=name, data=data) for name, data in _stream(kind, code)]
    with pytest.raises(CodexResponseError) as caught:
        consume_events(events)
    assert caught.value.code == code


@pytest.mark.parametrize("kind", ["error", "response.failed"])
@pytest.mark.parametrize("code", _TERMINAL_CODES)
def test_terminal_code_ends_invoke_and_stream(auth_file, kind, code):
    body = _sse_bytes(_stream(kind, code))
    llm = _make_llm(auth_file, transport=_CaptureTransport(body=body))
    with pytest.raises(CodexResponseError) as caught:
        llm.invoke([HumanMessage("hi")])
    assert caught.value.code == code
    chunks = []
    with pytest.raises(CodexResponseError):
        for chunk in llm.stream([HumanMessage("hi")]):
            chunks.append(chunk)
    assert "late" not in "".join(str(c.content) for c in chunks)


@pytest.mark.asyncio
@pytest.mark.parametrize("code", ["context_length_exceeded", "budget_reached"])
async def test_terminal_code_ends_astream(auth_file, code):
    body = _sse_bytes(_stream("error", code))
    llm = _make_llm(auth_file, transport=_AsyncCaptureTransport(body=body))
    with pytest.raises(CodexResponseError) as caught:
        async for _ in llm.astream([HumanMessage("hi")]):
            pass
    assert caught.value.code == code
