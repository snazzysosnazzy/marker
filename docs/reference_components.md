# Recognition and correction components

This fork extends Marker 2.2.0 with local LightOnOCR recognition, selective visual information extraction, and structured correction services. The package version and existing licensing remain unchanged. The bundled LightOnOCR grounding parser retains its [Apache-2.0 license](../marker/recognition/LICENSE.lightonocr); model checkpoints have their own licenses.

## Local LightOnOCR recognition

Install the optional dependencies from the repository checkout:

```sh
python -m pip install -e ".[lighton]"
```

`marker.recognition.lighton.LightOnRecognitionPredictor` implements Marker's full-page recognition contract. It maps grounded labels and bounding boxes into Marker blocks, preserves code as escaped preformatted HTML, and rejects malformed grounding, invalid boxes, unsupported labels, and empty recognition of nonblank pages.

The local backend supports Qwen-based LightOnOCR-3 0.8B and 4B checkpoints. The default is `lightonai/LightOnOCR-3-0.8B`. The 1B architecture is unsupported. Model loading is lazy; inference uses Hugging Face Transformers in the current process. `lighton_device="auto"` selects CUDA when available and otherwise CPU. Automatic dtype is bfloat16 on CUDA and float32 on CPU; supported devices are CPU and CUDA.

| Configuration | Default | Purpose |
| --- | --- | --- |
| `lighton_model` | `lightonai/LightOnOCR-3-0.8B` | Hub model identifier or local checkpoint directory |
| `lighton_revision` | `None` | Optional checkpoint revision |
| `lighton_device` | `auto` | `auto`, `cpu`, or CUDA device |
| `lighton_dtype` | `auto` | `auto`, `bfloat16`, `float16`, or `float32` |
| `lighton_max_new_tokens` | `16384` | Generated-token limit |
| `lighton_max_image_size` | `2048` | Image-size limit |
| `lighton_temperature` / `lighton_top_p` | `0.2` / `0.9` | Generation sampling settings |
| `lighton_seed` | `0` | Generation seed |
| `lighton_local_files_only` | `False` | Require an existing local checkpoint/cache |
| `lighton_record_traces` | `False` | Include source OCR text in recognition traces |

The following function accepts a source path and returns the normal Marker Markdown output object. Supplying the recognition artifact directly avoids constructing Surya or Indra inference models. `mode="vllm"` selects Marker's full-page routing; it does not start a vLLM server for this backend.

```python
from marker.converters.pdf import PdfConverter
from marker.recognition.lighton import LightOnRecognitionPredictor


def convert_with_lighton(source_path, model_path=None):
    recognition_config = {}
    if model_path is not None:
        recognition_config.update(
            lighton_model=str(model_path),
            lighton_local_files_only=True,
        )
    recognition = LightOnRecognitionPredictor(recognition_config)
    artifacts = {
        "recognition_model": recognition,
        "layout_model": None,
        "fast_layout_model": None,
        "ocr_error_model": None,
        "indra_model": None,
    }
    converter = PdfConverter(
        artifact_dict=artifacts,
        config={"mode": "vllm", "use_llm": False},
    )
    return converter(source_path)
```

The first inference may download the selected checkpoint unless local-only loading is enabled. Marker may also obtain its rendering font. Surya schema classes remain package dependencies, but this artifact configuration does not initialize their inference models.

## Selective visual information

`SelectiveVisualProcessor` and `ProtectedPageCorrectionProcessor` are available from `marker.processors.llm.llm_visual_information`. They are opt-in processors and require an LLM service.

`SelectiveVisualProcessor` examines Picture, Figure, and Diagram blocks with surrounding page text. Its structured decision is one of:

- `omit`: remove a decorative, redundant, or uninformative visual and its associated caption group.
- `retain_data`: replace the visual with useful quantitative data or relationships expressed as semantic HTML.
- `retain_instruction`: retain unique instructions, code, settings, or diagram relationships missing from surrounding text.

Retained content is validated against a restricted HTML vocabulary. Omitted content must be empty. Decisions are recorded under `document.metadata["visual_decisions"]`; these records contain extracted content and reasons, so they are document data rather than anonymous telemetry.

In a custom `processor_list`, replace `LLMImageDescriptionProcessor` with `SelectiveVisualProcessor` and replace `LLMPageCorrectionProcessor` with `ProtectedPageCorrectionProcessor`. Keep selective visual processing before protected correction. Pass fully qualified class names to `PdfConverter`. Protected correction masks decided regions, excludes their block IDs from rewrites, and preserves their positions during reordering. Other custom processors must respect these decisions.

`validate_visual_decisions(document)` checks that every expected visual has exactly one valid decision. A strict integration should invoke it before publishing output and inspect service failure records: the services preserve Marker's convention of returning `{}` for failed requests, and not every processor propagates that failure.

## Claude structured correction

`marker.services.claude.ClaudeService` accepts `claude_api_key` or the `ANTHROPIC_API_KEY` environment variable. An explicit configuration value takes precedence. `claude_workspace_id`, or `ANTHROPIC_WORKSPACE_ID` when the field is omitted, adds the `anthropic-workspace-id` header. No workspace header is sent when neither is configured.

| Configuration | Default | Behavior |
| --- | --- | --- |
| `claude_model_name` | `claude-opus-4-8` | Requested model identifier |
| `max_claude_tokens` | `8192` | Response-token cap |
| `claude_streaming` | `False` | Stream a complete Messages response when enabled |
| `claude_effort` | `None` | Explicit effort setting; also enables adaptive thinking |
| `max_retries` / `retry_wait_time` | `2` / `3` seconds | Explicit rate-limit and timeout retry policy |

A service configuration for Sonnet correction, with credentials supplied through the environment:

```python
from marker.services.claude import ClaudeService

claude = ClaudeService({
    "claude_model_name": "claude-sonnet-5-5",
    "claude_effort": "high",
    "claude_streaming": True,
    "max_claude_tokens": 128000,
})
```

For `PdfConverter`, select `llm_service="marker.services.claude.ClaudeService"` and include the service settings and `use_llm=True` in its configuration. Each service caches its client and disables SDK retries to avoid overlapping retry layers.

The Anthropic SDK transforms the Pydantic schema into `output_config.format`. Validation occurs after the complete response arrives, preserving final token usage even when JSON is truncated or invalid. Refusals, `max_tokens`, missing content, and validation failures return `{}`. This follows the provider's [structured output interface](https://platform.claude.com/docs/en/build-with-claude/structured-outputs).

`usage_records` contains one entry per attempt: model, stop reason, success, elapsed seconds, available input/output/cache token counts, and sanitized error class/status. It never records credentials, prompts, response text, or image payloads. Estimates cover standard Sonnet 5.5, Opus 5.5, and Haiku 5.5 token rates, including Haiku's input-size tier and five-minute cache-write rates; unknown models have no estimate. These are estimates rather than invoices; [Anthropic pricing](https://platform.claude.com/docs/en/about-claude/pricing) governs actual charges.

## Direct ChatGPT plan authorization

`marker.services.chatgpt.ChatGPTService` uses a Sign in with ChatGPT access token authorized for direct plan usage. The caller owns sign-in, token storage, refresh, and account-specific model selection. A subscription alone does not supply that authorization. See the official [token reference](https://developers.openai.com/siwc/token-sharing-open-source/token-reference) and [models and inference guide](https://developers.openai.com/siwc/token-sharing-open-source/models-and-inference).

| Configuration | Default | Behavior |
| --- | --- | --- |
| `chatgpt_access_token` | `None` | OAuth access token; falls back to `CHATGPT_ACCESS_TOKEN` |
| `chatgpt_model` | `gpt-6.1-sol` | Model slug available to the authorized account |
| `chatgpt_effort` | `None` | Optional reasoning effort |

The service calls the public Responses API with `store=False` and `stream=True`. It does not read browser sessions, reuse unrelated credentials, or fall back to API-key billing. An expired token requires caller-managed refresh and a new service instance. No unsupported output-token cap or sampling parameter is sent.

```python
from marker.services.chatgpt import ChatGPTService

chatgpt = ChatGPTService({"chatgpt_model": "gpt-6.1-sol"})
```

For `PdfConverter`, select `llm_service="marker.services.chatgpt.ChatGPTService"` with `use_llm=True`. The service requires a completed terminal response and valid structured output before returning content. Available token usage is recorded; its absence does not invalidate a completed response. Refused, incomplete, interrupted, and invalid responses return `{}`. Its sanitized `usage_records` identify `billing="chatgpt_plan"` and leave `estimated_usd=None`; token counts do not imply an API charge. Known API-key strings are rejected by this OAuth-only service.
