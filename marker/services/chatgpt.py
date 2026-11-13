"""Responses API service using an authorized Sign in with ChatGPT access token."""

import copy
import os
import threading
import time
from typing import Annotated

from openai import OpenAI
from pydantic import BaseModel

from marker.logger import get_logger
from marker.services import BaseService

logger = get_logger()


def strict_schema(response_schema: type[BaseModel]) -> dict:
    """Convert a Pydantic object schema to the strict structured-output shape."""
    schema = copy.deepcopy(response_schema.model_json_schema())

    def visit(node):
        if not isinstance(node, dict):
            return
        node.pop("default", None)
        if node.get("type") == "object" or "properties" in node:
            node["additionalProperties"] = False
            node["required"] = list(node.get("properties", {}))
        for key in ("properties", "$defs", "definitions"):
            for child in node.get(key, {}).values():
                visit(child)
        for key in ("items", "additionalProperties"):
            visit(node.get(key))
        for key in ("anyOf", "oneOf", "allOf"):
            for child in node.get(key, []):
                visit(child)

    visit(schema)
    return schema


class ChatGPTService(BaseService):
    """Use plan-authorized OAuth credentials without API-key fallback.

    Access tokens must carry chatgpt.tokens.use.direct authorization. Login and
    refresh belong to the caller; create a new service after replacing a token.
    """

    chatgpt_access_token: str | None = None
    chatgpt_model: Annotated[str, "ChatGPT-authorized model identifier."] = (
        "gpt-6.1-sol"
    )
    chatgpt_effort: str | None = None

    def __init__(self, config=None):
        super().__init__(config)
        self._client = None
        self._lock = threading.Lock()
        self._call_count = 0
        self.usage_records = []

    def process_images(self, images):
        return [
            {
                "type": "input_image",
                "image_url": "data:image/webp;base64," + self.img_to_base64(image),
            }
            for image in images
        ]

    def get_client(self):
        with self._lock:
            if self._client is None:
                token = self.chatgpt_access_token or os.environ.get(
                    "CHATGPT_ACCESS_TOKEN"
                )
                if not token or not token.strip():
                    raise ValueError("A Sign in with ChatGPT access token is required.")
                if token.strip().startswith("sk-"):
                    raise ValueError(
                        "API keys are not accepted by the ChatGPT plan service."
                    )
                self._client = OpenAI(
                    api_key=token,
                    base_url="https://api.openai.com/v1",
                    max_retries=0,
                )
            return self._client

    def __call__(
        self,
        prompt,
        image,
        block,
        response_schema,
        max_retries=None,
        timeout=None,
    ):
        with self._lock:
            self._call_count += 1
            call_number = self._call_count
        record = {
            "call": call_number,
            "model": self.chatgpt_model,
            "effort": self.chatgpt_effort,
            "streaming": True,
            "success": False,
            "estimated_usd": None,
            "billing": "chatgpt_plan",
        }
        started = time.perf_counter()
        try:
            request = {
                "model": self.chatgpt_model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            *self.format_image_for_llm(image),
                        ],
                    }
                ],
                "store": False,
                "stream": True,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": response_schema.__name__,
                        "strict": True,
                        "schema": strict_schema(response_schema),
                    }
                },
                "timeout": self.timeout if timeout is None else timeout,
            }
            if self.chatgpt_effort is not None:
                request["reasoning"] = {"effort": self.chatgpt_effort}
            # No unsupported output caps, sampling parameters, or conversation state.
            client = self.get_client()
            retries = self.max_retries if max_retries is None else max_retries
            completed = None
            refused = False
            with client.with_options(max_retries=retries).responses.create(
                **request
            ) as stream:
                for event in stream:
                    if event.type in (
                        "response.failed",
                        "response.incomplete",
                        "error",
                    ):
                        record["error_type"] = "IncompleteResponse"
                        return {}
                    if event.type in (
                        "response.refusal.delta",
                        "response.refusal.done",
                    ):
                        refused = True
                    if event.type == "response.completed":
                        if completed is not None:
                            record["error_type"] = "DuplicateCompletion"
                            return {}
                        completed = event.response
            if completed is None or completed.status != "completed":
                record["error_type"] = "InterruptedResponse"
                return {}
            record["response_model"] = completed.model
            record["stop_reason"] = completed.status
            usage = completed.usage
            record["usage_available"] = usage is not None
            if usage is not None:
                record.update(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    total_tokens=usage.total_tokens,
                )
            if block is not None:
                metadata = {"llm_request_count": 1}
                if usage is not None:
                    metadata["llm_tokens_used"] = usage.total_tokens
                block.update_metadata(**metadata)
            if refused:
                record["error_type"] = "Refusal"
                return {}
            parts = []
            for item in completed.output:
                if item.type != "message":
                    continue
                for content in item.content:
                    if content.type == "refusal":
                        record["error_type"] = "Refusal"
                        return {}
                    if content.type == "output_text":
                        parts.append(content.text)
            if not parts:
                record["error_type"] = "MissingStructuredOutput"
                return {}
            result = response_schema.model_validate_json("".join(parts)).model_dump()
            record["success"] = True
            return result
        except Exception as error:  # noqa: BLE001 - Service failures return sanitized metadata and no content.
            record["error_type"] = type(error).__name__
            status = getattr(error, "status_code", None)
            if isinstance(status, int):
                record["status_code"] = status
            if status in (401, 403):
                record["error_type"] = "ChatGPTAuthorizationError"
                logger.warning(
                    "ChatGPT authorization failed; obtain a fresh authorized access token."
                )
            else:
                logger.warning("ChatGPT request failed (%s).", type(error).__name__)
            return {}
        finally:
            record["seconds"] = round(time.perf_counter() - started, 3)
            with self._lock:
                self.usage_records.append(record)
