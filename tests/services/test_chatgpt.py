"""HTTP-mocked checks for ChatGPT-authorized structured Responses requests."""

import json
import os
import unittest
from unittest.mock import patch

import httpx
from openai import OpenAI
from PIL import Image
from pydantic import BaseModel

from marker.services.chatgpt import ChatGPTService


class Nested(BaseModel):
    count: int


class Result(BaseModel):
    answer: str
    detail: Nested | None = None


def completed(
    text='{"answer":"ok","detail":null}', status="completed", content=None, usage=True
):
    response = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1,
        "status": status,
        "model": "gpt-6.1-sol",
        "output": [
            {
                "id": "msg_test",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": content
                or [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "usage": {"input_tokens": 12, "output_tokens": 8, "total_tokens": 20}
        if usage
        else None,
    }
    return {"type": "response.completed", "response": response, "sequence_number": 1}


def sse(events):
    return (
        "".join(
            "event: " + item["type"] + "\ndata: " + json.dumps(item) + "\n\n"
            for item in events
        )
        + "data: [DONE]\n\n"
    )


class ChatGPTTests(unittest.TestCase):
    token = "oauth-test-credential-not-a-real-token"

    def run_request(self, events, status=200, config=None, image=None):
        requests = []

        def handler(request):
            requests.append(request)
            if status != 200:
                return httpx.Response(
                    status,
                    json={
                        "error": {"message": self.token, "type": "authentication_error"}
                    },
                )
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, text=sse(events)
            )

        transport = httpx.MockTransport(handler)
        clients = []

        def factory(**kwargs):
            client = OpenAI(**kwargs, http_client=httpx.Client(transport=transport))
            clients.append(client)
            return client

        service = ChatGPTService(
            {
                "chatgpt_access_token": self.token,
                "chatgpt_effort": "high",
                "max_retries": 0,
                **(config or {}),
            }
        )
        with patch("marker.services.chatgpt.OpenAI", side_effect=factory):
            result = service("Read the document.", image, None, Result)
        for client in clients:
            client.close()
        return result, service, requests

    def test_actual_bearer_image_protocol_and_strict_schema(self):
        result, service, requests = self.run_request(
            [completed()], image=Image.new("RGB", (4, 4), "white")
        )
        self.assertEqual(result, {"answer": "ok", "detail": None})
        request = requests[0]
        self.assertEqual(str(request.url), "https://api.openai.com/v1/responses")
        self.assertEqual(request.headers["authorization"], "Bearer " + self.token)
        body = json.loads(request.content)
        self.assertFalse(body["store"])
        self.assertTrue(body["stream"])
        self.assertEqual(body["reasoning"], {"effort": "high"})
        self.assertEqual(body["input"][0]["content"][0]["type"], "input_text")
        self.assertTrue(
            body["input"][0]["content"][1]["image_url"].startswith(
                "data:image/webp;base64,"
            )
        )
        for forbidden in (
            "max_output_tokens",
            "max_tool_calls",
            "metadata",
            "temperature",
            "top_p",
            "conversation",
            "previous_response_id",
        ):
            self.assertNotIn(forbidden, body)
        schema = body["text"]["format"]["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(schema["required"], ["answer", "detail"])
        self.assertFalse(schema["$defs"]["Nested"]["additionalProperties"])
        self.assertTrue(service.usage_records[0]["usage_available"])
        self.assertEqual(service.usage_records[0]["total_tokens"], 20)
        self.assertIsNone(service.usage_records[0]["estimated_usd"])
        self.assertNotIn(self.token, json.dumps(service.usage_records))

    def test_failure_incomplete_interruption_refusal_and_bad_json(self):
        cases = [
            [
                {
                    "type": "response.failed",
                    "response": {"status": "failed"},
                    "sequence_number": 1,
                }
            ],
            [
                {
                    "type": "response.incomplete",
                    "response": {"status": "incomplete"},
                    "sequence_number": 1,
                }
            ],
            [
                {
                    "type": "response.output_text.delta",
                    "delta": '{"answer":"ok"}',
                    "sequence_number": 1,
                    "item_id": "msg",
                    "output_index": 0,
                    "content_index": 0,
                }
            ],
            [completed(status="incomplete")],
            [completed(content=[{"type": "refusal", "refusal": "No"}])],
            [completed(text='{"answer":')],
        ]
        for events in cases:
            with self.subTest(events=events):
                result, service, _ = self.run_request(events)
                self.assertEqual(result, {})
                self.assertFalse(service.usage_records[0]["success"])
                self.assertIn("error_type", service.usage_records[0])

    def test_refusal_delta_waits_for_terminal_usage(self):
        event = {
            "type": "response.refusal.delta",
            "delta": "No",
            "sequence_number": 1,
            "item_id": "msg_test",
            "output_index": 0,
            "content_index": 0,
        }
        result, service, _ = self.run_request([event, completed()])
        self.assertEqual(result, {})
        self.assertEqual(service.usage_records[0]["error_type"], "Refusal")
        self.assertTrue(service.usage_records[0]["usage_available"])
        self.assertEqual(service.usage_records[0]["total_tokens"], 20)

    def test_auth_error_redacts_body_and_token(self):
        with self.assertLogs("marker", level="WARNING") as logs:
            result, service, _ = self.run_request([], status=401)
        self.assertEqual(result, {})
        self.assertEqual(
            service.usage_records[0]["error_type"], "ChatGPTAuthorizationError"
        )
        self.assertNotIn(
            self.token, " ".join(logs.output) + json.dumps(service.usage_records)
        )

    def test_no_api_key_fallback(self):
        with (
            patch.dict(os.environ, {"OPENAI_API_KEY": self.token}, clear=True),
            patch("marker.services.chatgpt.OpenAI") as factory,
        ):
            service = ChatGPTService()
            self.assertEqual(service("Text", None, None, Result), {})
            factory.assert_not_called()

    def test_valid_completion_without_usage(self):
        result, service, _ = self.run_request([completed(usage=False)])
        self.assertEqual(result, {"answer": "ok", "detail": None})
        record = service.usage_records[0]
        self.assertTrue(record["success"])
        self.assertFalse(record["usage_available"])
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            self.assertNotIn(key, record)

    def test_explicit_api_key_is_rejected_before_client_creation(self):
        for source in ("config", "environment"):
            with (
                self.subTest(source=source),
                patch.dict(os.environ, {}, clear=True),
                patch("marker.services.chatgpt.OpenAI") as factory,
            ):
                config = (
                    {"chatgpt_access_token": "sk-test-not-real"}
                    if source == "config"
                    else {}
                )
                if source == "environment":
                    os.environ["CHATGPT_ACCESS_TOKEN"] = "sk-test-not-real"
                service = ChatGPTService(config)
                with self.assertRaisesRegex(ValueError, "API keys are not accepted"):
                    service.get_client()
                factory.assert_not_called()

    def test_environment_token(self):
        with (
            patch.dict(os.environ, {"CHATGPT_ACCESS_TOKEN": self.token}),
            patch("marker.services.chatgpt.OpenAI") as factory,
        ):
            service = ChatGPTService()
            service.get_client()
            self.assertEqual(factory.call_args.kwargs["api_key"], self.token)


if __name__ == "__main__":
    unittest.main(verbosity=2)
