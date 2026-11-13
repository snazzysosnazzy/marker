from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from marker.services.claude import ClaudeService


class Answer(BaseModel):
    text: str


def response(stop_reason="end_turn", parsed=True, model="claude-sonnet-5-5"):
    return SimpleNamespace(
        model=model,
        stop_reason=stop_reason,
        content=[SimpleNamespace(type="text", text='{"text":"converted"}')]
        if parsed
        else [],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=20,
            cache_creation_input_tokens=30,
            cache_read_input_tokens=40,
        ),
    )


@pytest.fixture
def sdk(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)
    factory = MagicMock()
    factory.return_value.messages.create.return_value = response()
    factory.return_value.messages.stream.return_value.__enter__.return_value.get_final_message.return_value = response()
    monkeypatch.setattr("marker.services.claude.anthropic.Anthropic", factory)
    return factory


def service(**config):
    return ClaudeService(
        {
            "claude_api_key": "credential",
            "claude_model_name": "claude-sonnet-5-5",
            **config,
        }
    )


@pytest.mark.parametrize("streaming", [False, True])
def test_request_and_usage(sdk, streaming):
    converter = service(
        claude_streaming=streaming, claude_effort="high", max_claude_tokens=128000
    )
    block = MagicMock()
    assert converter("private document", None, block, Answer, timeout=90) == {
        "text": "converted"
    }
    method = (
        sdk.return_value.messages.stream
        if streaming
        else sdk.return_value.messages.create
    )
    request = method.call_args.kwargs
    assert request == {
        "model": "claude-sonnet-5-5",
        "max_tokens": 128000,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "private document"}]}
        ],
        "timeout": 90,
        "output_config": {
            "effort": "high",
            "format": {
                "type": "json_schema",
                "schema": anthropic.transform_schema(Answer),
            },
        },
        "thinking": {"type": "adaptive"},
    }
    record = converter.usage_records[0]
    assert record["success"] is True
    assert record["stop_reason"] == "end_turn"
    assert record["input_tokens"] == 100
    assert record["output_tokens"] == 20
    assert record["cache_write_tokens"] == 30
    assert record["cache_read_tokens"] == 40
    assert record["estimated_usd"] == pytest.approx(0.000479)
    assert record["seconds"] >= 0
    block.update_metadata.assert_called_once_with(
        llm_tokens_used=190, llm_request_count=1
    )
    assert "private document" not in repr(record)
    assert "credential" not in repr(record)


def test_defaults_and_client_reuse(sdk):
    converter = ClaudeService({"claude_api_key": "credential"})
    converter("text", None, None, Answer)
    converter("text", None, None, Answer)
    sdk.assert_called_once_with(api_key="credential", max_retries=0)
    request = sdk.return_value.messages.create.call_args.kwargs
    assert request["model"] == "claude-opus-4-8"
    assert request["max_tokens"] == 8192
    assert "thinking" not in request
    assert "effort" not in request["output_config"]
    sdk.return_value.messages.stream.assert_not_called()


def test_environment_and_config_precedence(sdk, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "environment-workspace")
    ClaudeService().get_client()
    sdk.assert_called_with(
        api_key="environment-key",
        max_retries=0,
        default_headers={"anthropic-workspace-id": "environment-workspace"},
    )
    configured = ClaudeService(
        {
            "claude_api_key": "configured-key",
            "claude_workspace_id": "configured-workspace",
        }
    )
    configured.get_client()
    sdk.assert_called_with(
        api_key="configured-key",
        max_retries=0,
        default_headers={"anthropic-workspace-id": "configured-workspace"},
    )
    scoped = ClaudeService(
        {
            "ClaudeService_claude_api_key": "scoped-key",
            "ClaudeService_claude_workspace_id": None,
        }
    )
    scoped.get_client()
    sdk.assert_called_with(api_key="scoped-key", max_retries=0)


@pytest.mark.parametrize(
    "reason,parsed", [("refusal", False), ("max_tokens", True), ("end_turn", False)]
)
def test_incomplete_responses_preserve_billable_usage(sdk, reason, parsed):
    sdk.return_value.messages.create.return_value = response(reason, parsed)
    converter = service()
    assert converter("text", None, None, Answer) == {}
    record = converter.usage_records[0]
    assert record["success"] is False
    assert record["stop_reason"] == reason
    assert record["output_tokens"] == 20
    assert record["estimated_usd"] > 0


def test_failure_redaction(sdk, caplog):
    sdk.return_value.messages.create.side_effect = anthropic.AuthenticationError(
        "credential and private document",
        response=httpx.Response(
            401, request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
        ),
        body={"prompt": "private document"},
    )
    converter = service()
    assert converter("private document", None, None, Answer) == {}
    assert converter.usage_records[0]["error_type"] == "AuthenticationError"
    assert converter.usage_records[0]["status_code"] == 401
    assert converter.usage_records[0]["success"] is False
    assert "credential" not in caplog.text + repr(converter.usage_records)
    assert "private document" not in caplog.text + repr(converter.usage_records)
    assert sdk.return_value.messages.create.call_count == 1


def test_explicit_retry_accounts_for_each_attempt(sdk, monkeypatch):
    error = anthropic.APITimeoutError(
        request=httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    )
    sdk.return_value.messages.create.side_effect = [error, response()]
    sleep = MagicMock()
    monkeypatch.setattr("marker.services.claude.time.sleep", sleep)
    converter = service()
    assert converter("text", None, None, Answer, max_retries=1) == {"text": "converted"}
    assert [record["success"] for record in converter.usage_records] == [False, True]
    sleep.assert_called_once_with(converter.retry_wait_time)
    sdk.assert_called_once_with(api_key="credential", max_retries=0)


@pytest.mark.parametrize(
    "model,input_tokens,expected",
    [
        ("claude-sonnet-5-5", 100000, 0.200279),
        ("claude-opus-5-5", 100000, 0.400558),
        ("claude-haiku-5-5", 99930, 0.01000715),
        ("claude-haiku-5-5", 99931, 0.05003625),
        ("unknown", 100000, None),
    ],
)
def test_prices_and_haiku_threshold(model, input_tokens, expected):
    usage = {
        "input_tokens": input_tokens,
        "output_tokens": 20,
        "cache_write_tokens": 30,
        "cache_read_tokens": 40,
    }
    cost = ClaudeService.estimate_cost(model, usage)
    if expected is None:
        assert cost is None
    else:
        assert cost == pytest.approx(expected)


def test_concurrent_usage_and_client_initialization(sdk):
    converter = service()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(
            executor.map(lambda _: converter("text", None, None, Answer), range(30))
        )
    assert all(result == {"text": "converted"} for result in results)
    sdk.assert_called_once()
    assert len(converter.usage_records) == 30
    assert sorted(record["call"] for record in converter.usage_records) == list(
        range(1, 31)
    )


@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize(
    "reason,text,success",
    [
        ("end_turn", '{"text":"converted"}', True),
        ("max_tokens", '{"text":"unfinished', False),
        ("refusal", "Cannot fulfill this request.", False),
        ("end_turn", '{"unexpected":"private document"}', False),
    ],
)
def test_sdk_response_validation_preserves_final_usage(
    streaming, reason, text, success
):
    import json

    usage = {
        "input_tokens": 100,
        "output_tokens": 20,
        "cache_creation_input_tokens": 30,
        "cache_read_input_tokens": 40,
    }
    message = {
        "id": "msg_test",
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-5-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": reason,
        "stop_sequence": None,
        "usage": usage,
    }

    def handle(request):
        body = json.loads(request.content)
        assert (
            body["output_config"]["format"]["schema"]["additionalProperties"] is False
        )
        assert "output_format" not in body
        if not streaming:
            return httpx.Response(200, json=message)
        start = {
            **message,
            "content": [],
            "stop_reason": None,
            "usage": {**usage, "output_tokens": 1},
        }
        events = [
            {"type": "message_start", "message": start},
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": text},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": reason, "stop_sequence": None},
                "usage": {"output_tokens": 20},
            },
            {"type": "message_stop"},
        ]
        content = "".join(
            f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
        )
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, content=content
        )

    converter = service(claude_streaming=streaming)
    with httpx.Client(transport=httpx.MockTransport(handle)) as http_client:
        converter._client = anthropic.Anthropic(
            api_key="credential", max_retries=0, http_client=http_client
        )
        result = converter("private document", None, None, Answer)
    assert result == ({"text": "converted"} if success else {})
    record = converter.usage_records[0]
    assert record["success"] is success
    assert record["stop_reason"] == reason
    assert record["output_tokens"] == 20
    assert record["input_tokens"] == 100
    assert record["cache_write_tokens"] == 30
    assert record["cache_read_tokens"] == 40
    assert record["estimated_usd"] == pytest.approx(0.000479)
    assert "private document" not in repr(record)
