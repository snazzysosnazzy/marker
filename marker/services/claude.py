import os
import threading
import time
from typing import Annotated

import anthropic
import PIL
from anthropic import APITimeoutError, RateLimitError
from PIL import Image
from pydantic import BaseModel

from marker.logger import get_logger
from marker.schema.blocks import Block
from marker.services import BaseService

logger = get_logger()


class ClaudeService(BaseService):
    claude_model_name: Annotated[
        str, "The name of the Anthropic model to use for the service."
    ] = "claude-opus-4-8"
    claude_api_key: Annotated[str, "The Claude API key to use for the service."] = None
    max_claude_tokens: Annotated[
        int, "The maximum number of tokens to use for a single Claude request."
    ] = 8192
    claude_workspace_id: str | None = None
    claude_effort: str | None = None
    claude_streaming: Annotated[bool, "Stream structured responses."] = False

    def __init__(self, config=None):
        self.claude_api_key = os.environ.get("ANTHROPIC_API_KEY")
        self.claude_workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
        super().__init__(config)
        self._client = None
        self._lock = threading.Lock()
        self._call_count = 0
        self.usage_records = []

    def process_images(self, images: list[Image.Image]) -> list[dict]:
        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/webp",
                    "data": self.img_to_base64(img),
                },
            }
            for img in images
        ]

    def get_client(self):
        with self._lock:
            if self._client is None:
                options = {"api_key": self.claude_api_key, "max_retries": 0}
                if self.claude_workspace_id:
                    options["default_headers"] = {
                        "anthropic-workspace-id": self.claude_workspace_id
                    }
                self._client = anthropic.Anthropic(**options)
            return self._client

    @staticmethod
    def estimate_cost(model, usage):
        """Estimate standard USD charges for supported models; omit unknown rates."""
        rates = {
            "claude-sonnet-5-5": (2.0, 10.0, 2.5, 0.1),
            "claude-opus-5-5": (4.0, 20.0, 5.0, 0.2),
        }.get(model)
        if model == "claude-haiku-5-5":
            input_total = sum(
                usage[key]
                for key in ("input_tokens", "cache_write_tokens", "cache_read_tokens")
            )
            rates = (
                (0.1, 0.5, 0.125, 0.01)
                if input_total <= 100000
                else (0.5, 2.5, 0.625, 0.05)
            )
        if rates is None:
            return None
        return (
            sum(
                usage[key] * rate
                for key, rate in zip(
                    (
                        "input_tokens",
                        "output_tokens",
                        "cache_write_tokens",
                        "cache_read_tokens",
                    ),
                    rates,
                )
            )
            / 1_000_000
        )

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | list[PIL.Image.Image] | None,
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        max_retries = self.max_retries if max_retries is None else max_retries
        timeout = self.timeout if timeout is None else timeout
        total_tries = max_retries + 1
        for attempt in range(1, total_tries + 1):
            with self._lock:
                self._call_count += 1
                call_number = self._call_count
            started = time.perf_counter()
            record = {
                "call": call_number,
                "attempt": attempt,
                "model": self.claude_model_name,
                "effort": self.claude_effort,
                "max_tokens": self.max_claude_tokens,
                "streaming": self.claude_streaming,
                "success": False,
                "stop_reason": None,
            }
            retry = False
            try:
                client = self.get_client()
                request = {
                    "model": self.claude_model_name,
                    "max_tokens": self.max_claude_tokens,
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                *self.format_image_for_llm(image),
                                {"type": "text", "text": prompt},
                            ],
                        }
                    ],
                    "output_config": {
                        "format": {
                            "type": "json_schema",
                            "schema": anthropic.transform_schema(response_schema),
                        }
                    },
                    "timeout": timeout,
                }
                if self.claude_effort is not None:
                    request["output_config"]["effort"] = self.claude_effort
                    request["thinking"] = {"type": "adaptive"}
                if self.claude_streaming:
                    with client.messages.stream(**request) as stream:
                        response = stream.get_final_message()
                else:
                    response = client.messages.create(**request)
                record["response_model"] = response.model
                record["stop_reason"] = response.stop_reason
                usage = response.usage
                if usage is not None:
                    record.update(
                        {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "cache_write_tokens": getattr(
                                usage, "cache_creation_input_tokens", 0
                            )
                            or 0,
                            "cache_read_tokens": getattr(
                                usage, "cache_read_input_tokens", 0
                            )
                            or 0,
                        }
                    )
                    record["estimated_usd"] = self.estimate_cost(response.model, record)
                    if block is not None:
                        block.update_metadata(
                            llm_tokens_used=sum(
                                record[key]
                                for key in (
                                    "input_tokens",
                                    "output_tokens",
                                    "cache_write_tokens",
                                    "cache_read_tokens",
                                )
                            ),
                            llm_request_count=1,
                        )
                if response.stop_reason not in ("end_turn", "stop_sequence"):
                    record["error_type"] = "IncompleteResponse"
                    logger.warning("Claude returned an incomplete response.")
                    return {}
                text = "".join(
                    item.text for item in response.content if item.type == "text"
                )
                if not text:
                    record["error_type"] = "MissingParsedOutput"
                    logger.warning("Claude returned no structured output.")
                    return {}
                # Validate after final usage arrives: SDK eager parsing can fail
                # before the stream's final usage event on truncated JSON.
                result = response_schema.model_validate_json(text).model_dump()
                record["success"] = True
                return result
            except Exception as error:  # noqa: BLE001 - Service contract returns {} on failure.
                record["error_type"] = type(error).__name__
                status = getattr(error, "status_code", None)
                if isinstance(status, int):
                    record["status_code"] = status
                retry = (
                    isinstance(error, (RateLimitError, APITimeoutError))
                    and attempt < total_tries
                )
                logger.warning(
                    "Claude request failed (%s, status=%s, attempt=%s/%s).",
                    type(error).__name__,
                    record.get("status_code"),
                    attempt,
                    total_tries,
                )
            finally:
                record["seconds"] = round(time.perf_counter() - started, 3)
                with self._lock:
                    self.usage_records.append(record)
            if not retry:
                break
            time.sleep(attempt * self.retry_wait_time)
        return {}
