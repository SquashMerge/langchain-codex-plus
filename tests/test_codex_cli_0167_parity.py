"""Parity with Codex CLI 0.167.1 (from 0.159.1).

Each test pins one difference between the two Codex releases that changes
what this library sends or how it reads a response:

* request field order: ``model``, ``stream``, ``service_tier`` first;
* ``response.failed`` and ``response.incomplete`` are errors, not a quiet
  end of stream;
* token refresh follows ``tokens.oauth_provenance`` (issuer + client_id)
  and reports ``refresh_reason``, and a refresh write-back keeps the
  provenance fields Codex records.
"""

from __future__ import annotations

import json

import httpx
import pytest
from langchain_core.messages import HumanMessage

from langchain_codex_plus import CODEX_OAUTH_CLIENT_ID, load_codex_auth, refresh_codex_auth
from langchain_codex_plus import codex_auth as _ca
from langchain_codex_plus.codex_auth import REFRESH_TOKEN_URL, _refresh_body
from langchain_codex_plus.codex_protocol import (
    CodexResponseError,
    SseEvent,
    build_request_body,
    consume_events,
)
from tests.conftest import _AsyncCaptureTransport, _CaptureTransport, _make_llm, _sse_bytes

ISSUER = "https://auth.example.test/"
PROVENANCE_CLIENT = "app_provenance_client"


# ─── Request body ──────────────────────────────────────────────────────


def test_routing_fields_lead_the_body_and_extra_still_applies():
    body = build_request_body(
        [HumanMessage("hi")],
        model="gpt-5.4",
        extra={"service_tier": "flex", "temperature": 0},
    )
    assert list(body)[:4] == ["model", "stream", "service_tier", "input"]
    assert body["service_tier"] == "flex"
    assert body["temperature"] == 0
    assert "instructions" not in body


# ─── Terminal events ───────────────────────────────────────────────────

_FAILED = ("response.failed", {
    "type": "response.failed",
    "response": {"id": "r", "status": "failed",
                 "error": {"code": "server_error", "message": "upstream exploded"}},
})
_INCOMPLETE = ("response.incomplete", {
    "type": "response.incomplete",
    "response": {"id": "r", "status": "incomplete",
                 "incomplete_details": {"reason": "content_filter"}},
})


def _stream(*terminal):
    return [
        ("response.created", {"response": {"id": "r"}}),
        ("response.output_text.delta", {"delta": "partial"}),
        *terminal,
    ]


@pytest.mark.parametrize(
    ("terminal", "code", "kind", "text"),
    [
        (_FAILED, "server_error", "response_failed", "upstream exploded"),
        (_INCOMPLETE, "content_filter", "response_incomplete", "content_filter"),
    ],
)
def test_consume_events_raises_on_failed_and_incomplete(terminal, code, kind, text):
    events = [SseEvent(event=name, data=data) for name, data in _stream(terminal)]
    with pytest.raises(CodexResponseError) as caught:
        consume_events(events)
    assert caught.value.code == code
    assert caught.value.type == kind
    assert text in caught.value.message


@pytest.mark.parametrize("terminal", [_FAILED, _INCOMPLETE], ids=["failed", "incomplete"])
def test_invoke_and_stream_raise_on_failed_and_incomplete(auth_file, terminal):
    body = _sse_bytes(_stream(terminal))
    llm = _make_llm(auth_file, transport=_CaptureTransport(body=body))
    with pytest.raises(CodexResponseError):
        llm.invoke([HumanMessage("hi")])
    with pytest.raises(CodexResponseError):
        list(llm.stream([HumanMessage("hi")]))


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal", [_FAILED, _INCOMPLETE], ids=["failed", "incomplete"])
async def test_astream_raises_on_failed_and_incomplete(auth_file, terminal):
    body = _sse_bytes(_stream(terminal))
    llm = _make_llm(auth_file, transport=_AsyncCaptureTransport(body=body))
    with pytest.raises(CodexResponseError):
        async for _ in llm.astream([HumanMessage("hi")]):
            pass


# ─── Refresh: provenance + refresh_reason ──────────────────────────────


def _auth_doc(**token_extra):
    return {
        "auth_mode": "chatgpt",
        "tokens": {
            "access_token": "atk-old",
            "id_token": "id",
            "refresh_token": "rtk-old",
            "account_id": "acct",
            **token_extra,
        },
        "last_refresh": "2026-05-21T00:00:00Z",
    }


class _TokenEndpoint(httpx.BaseTransport):
    def __init__(self):
        self.requests: list[httpx.Request] = []

    def handle_request(self, request):
        request.read()
        self.requests.append(request)
        return httpx.Response(
            200, json={"access_token": "atk-new", "refresh_token": "rtk-new"},
            request=request,
        )


@pytest.mark.parametrize(
    ("provenance", "expected"),
    [
        ({"issuer": ISSUER, "client_id": PROVENANCE_CLIENT}, (ISSUER, PROVENANCE_CLIENT)),
        ({"issuer": ISSUER}, (None, None)),
        ({"client_id": PROVENANCE_CLIENT}, (None, None)),
        ("not-a-dict", (None, None)),
    ],
    ids=["complete", "issuer-only", "client-only", "malformed"],
)
def test_load_reads_only_complete_provenance(tmp_path, provenance, expected):
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_auth_doc(oauth_provenance=provenance)))
    auth = load_codex_auth(path)
    assert (auth.oauth_issuer, auth.oauth_client_id) == expected


def test_refresh_uses_provenance_issuer_client_and_reason(tmp_path, monkeypatch):
    monkeypatch.delenv(_ca.REFRESH_TOKEN_URL_OVERRIDE_ENV_VAR, raising=False)
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_auth_doc(
        oauth_provenance={"issuer": ISSUER, "client_id": PROVENANCE_CLIENT},
    )))
    endpoint = _TokenEndpoint()
    refresh_codex_auth(
        load_codex_auth(path), path=path, write_back=False,
        http_client=httpx.Client(transport=endpoint),
        refresh_reason="unauthorized_recovery",
    )
    (req,) = endpoint.requests
    assert str(req.url) == "https://auth.example.test/oauth/token"
    assert json.loads(req.content) == {
        "client_id": PROVENANCE_CLIENT,
        "grant_type": "refresh_token",
        "refresh_token": "rtk-old",
        "refresh_reason": "unauthorized_recovery",
    }


def test_refresh_without_provenance_keeps_default_endpoint_and_body(tmp_path, monkeypatch):
    monkeypatch.delenv(_ca.REFRESH_TOKEN_URL_OVERRIDE_ENV_VAR, raising=False)
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_auth_doc()))
    endpoint = _TokenEndpoint()
    refresh_codex_auth(
        load_codex_auth(path), path=path, write_back=False,
        http_client=httpx.Client(transport=endpoint),
    )
    (req,) = endpoint.requests
    assert str(req.url) == REFRESH_TOKEN_URL
    assert json.loads(req.content) == {
        "client_id": CODEX_OAUTH_CLIENT_ID,
        "grant_type": "refresh_token",
        "refresh_token": "rtk-old",
    }


def test_override_env_beats_provenance(tmp_path, monkeypatch):
    monkeypatch.setenv(_ca.REFRESH_TOKEN_URL_OVERRIDE_ENV_VAR, "https://override.test/token")
    path = tmp_path / "auth.json"
    path.write_text(json.dumps(_auth_doc(
        oauth_provenance={"issuer": ISSUER, "client_id": PROVENANCE_CLIENT},
    )))
    endpoint = _TokenEndpoint()
    refresh_codex_auth(
        load_codex_auth(path), path=path, write_back=False,
        http_client=httpx.Client(transport=endpoint),
    )
    assert str(endpoint.requests[0].url) == "https://override.test/token"


def test_unknown_refresh_reason_is_rejected():
    with pytest.raises(ValueError):
        _refresh_body("rtk", refresh_reason="because")


def test_chat_model_401_refresh_reports_reason_and_keeps_provenance(tmp_path, monkeypatch):
    """End to end: a 401 refreshes with ``refresh_reason=unauthorized_recovery``
    against the recorded issuer, and the write-back keeps the provenance
    fields Codex CLI records in ``auth.json``."""
    monkeypatch.delenv(_ca.REFRESH_TOKEN_URL_OVERRIDE_ENV_VAR, raising=False)
    path = tmp_path / "auth.json"
    provenance = {"issuer": ISSUER, "client_id": PROVENANCE_CLIENT}
    path.write_text(json.dumps(_auth_doc(
        oauth_provenance=provenance, login_owner_id="login-123",
    )))

    class _TwoShot(httpx.BaseTransport):
        def __init__(self):
            self.calls = 0

        def handle_request(self, request):
            self.calls += 1
            if self.calls == 1:
                return httpx.Response(401, content=b'{"detail":"expired"}', request=request)
            return httpx.Response(200, content=_sse_bytes([
                ("response.created", {"response": {"id": "r"}}),
                ("response.output_text.delta", {"delta": "ok"}),
                ("response.completed", {"response": {"id": "r", "status": "completed"}}),
            ]), request=request)

    endpoint = _TokenEndpoint()
    real = _ca._do_refresh_sync
    monkeypatch.setattr(
        _ca, "_do_refresh_sync",
        lambda _c, url, body: real(httpx.Client(transport=endpoint), url, body),
    )
    llm = _make_llm(path, transport=_TwoShot())
    assert llm.invoke([HumanMessage("hi")]).content == "ok"

    (req,) = endpoint.requests
    assert str(req.url) == "https://auth.example.test/oauth/token"
    sent = json.loads(req.content)
    assert sent["refresh_reason"] == "unauthorized_recovery"
    assert sent["client_id"] == PROVENANCE_CLIENT
    tokens = json.loads(path.read_text())["tokens"]
    assert tokens["access_token"] == "atk-new"
    assert tokens["oauth_provenance"] == provenance
    assert tokens["login_owner_id"] == "login-123"
