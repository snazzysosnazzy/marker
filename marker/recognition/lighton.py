"""Adapt LightOnOCR grounding blocks to Marker's full-page recognition contract."""

from __future__ import annotations

import html
import re
import time

import markdown2
from PIL import Image, ImageChops
from surya.recognition.schema import BlockOCRResult, PageOCRResult

# Upstream parser; Apache-2.0 license is in LICENSE.lightonocr.
# https://github.com/lightonai/LightOnOCR/blob/main/lightonocr/grounding.py
from marker.recognition.grounding import MARKER, parse_blocks
from marker.util import assign_config

LABELS = {
    "text": "Text",
    "aside_text": "Text",
    "title": "SectionHeader",
    "section": "SectionHeader",
    "list": "ListGroup",
    "caption": "Caption",
    "footnote": "Footnote",
    "page_footnote": "Footnote",
    "header": "PageHeader",
    "page_header": "PageHeader",
    "header_image": "PageHeader",
    "footer": "PageFooter",
    "page_footer": "PageFooter",
    "footer_image": "PageFooter",
    "page_number": "PageFooter",
    "table": "Table",
    "chart": "Figure",
    "formula": "Equation",
    "code": "Code",
    "image": "Picture",
}
MARKER_CANDIDATE = re.compile(r"!\[[A-Za-z_][\w-]*\+?\]\(")
LANGUAGE_HINTS = {
    "python",
    "py",
    "javascript",
    "js",
    "typescript",
    "ts",
    "java",
    "c",
    "cpp",
    "c++",
    "csharp",
    "c#",
    "go",
    "rust",
    "ruby",
    "php",
    "swift",
    "kotlin",
    "scala",
    "bash",
    "sh",
    "shell",
    "powershell",
    "sql",
    "html",
    "css",
    "json",
    "yaml",
    "yml",
    "xml",
    "r",
    "matlab",
    "text",
    "plaintext",
}


def code_html(text: str) -> str:
    """Remove presentation fences or a language hint, preserving code bytes."""
    lines = text.splitlines(keepends=True)
    if lines and re.fullmatch(r"\s*(`{3,}|~{3,})[^\r\n]*\r?\n?", lines[0]):
        opening = re.match(r"\s*(`{3,}|~{3,})", lines[0]).group(1)
        lines = lines[1:]
        if lines and re.fullmatch(
            r"\s*" + re.escape(opening[0]) + "{" + str(len(opening)) + r",}\s*",
            lines[-1],
        ):
            lines = lines[:-1]
        text = "".join(lines).removesuffix("\n").removesuffix("\r")
    elif len(lines) > 1 and lines[0].strip().lower() in LANGUAGE_HINTS:
        text = "".join(lines[1:])
    return "<pre><code>" + html.escape(text, quote=False) + "</code></pre>"


def block_html(label: str, text: str) -> str:
    if label == "Code":
        return code_html(text)
    if label == "Equation":
        if re.search(r"<math\b", text, re.I):
            return text
        for left, right in [("$$", "$$"), ("\\[", "\\]"), ("$", "$")]:
            if (
                text.startswith(left)
                and text.endswith(right)
                and len(text) >= len(left) + len(right)
            ):
                text = text[len(left) : -len(right)].strip()
                break
        return '<math display="block">' + html.escape(text, quote=False) + "</math>"
    if label == "Table" and re.search(r"<table\b", text, re.I):
        return text
    return str(markdown2.markdown(text, extras=["tables", "fenced-code-blocks"]))


class LightOnRecognitionPredictor:
    """Full-page recognizer backed exclusively by the supplied OCR client."""

    lighton_model = "lightonai/LightOnOCR-3-0.8B"
    lighton_revision = None
    lighton_device = "auto"
    lighton_dtype = "auto"
    lighton_max_new_tokens = 16384
    lighton_max_image_size = 2048
    lighton_temperature = 0.2
    lighton_top_p = 0.9
    lighton_seed = 0
    lighton_local_files_only = False
    lighton_record_traces = False

    def __init__(self, config=None, backend=None):
        assign_config(self, config)
        if self.lighton_max_new_tokens <= 0 or self.lighton_max_image_size <= 0:
            raise ValueError("LightOnOCR token and image limits must be positive.")
        if self.lighton_temperature < 0 or not 0 < self.lighton_top_p <= 1:
            raise ValueError("LightOnOCR requires temperature >= 0 and 0 < top_p <= 1.")
        self.model_id = str(self.lighton_model)
        self.backend = backend if backend is not None else LocalLightOnClient(self)
        self.disable_tqdm = False
        self.pages = []

    @property
    def usage_records(self):
        return getattr(self.backend, "usage_records", [])

    def __call__(self, images, layout_results=None, full_page=True):
        if not full_page or layout_results is not None:
            raise ValueError(
                "LightOnOCR adapter supports full-page recognition only; block fallback is unavailable."
            )
        results = []
        for image in images:
            page_index = len(self.pages)
            raw = self.backend.infer(image, "grounding")
            if not isinstance(raw, str):
                raise TypeError("LightOnOCR backend must return a grounding string.")
            trace = {
                "page_index": page_index,
                "image_size": list(image.size),
                "blocks": [],
            }
            if self.lighton_record_traces:
                trace["raw_grounding"] = raw
            self.pages.append(trace)
            try:
                if not raw.strip() and not is_blank_image(image):
                    raise ValueError(
                        "LightOnOCR returned empty OCR for a nonblank image."
                    )
                result = self._parse_page(raw, image.size, trace)
            except Exception as error:
                trace["error"] = str(error)
                raise
            results.append(result)
        return results

    def _parse_page(self, raw, size, trace):
        markers = list(MARKER.finditer(raw))
        if raw.strip() and not markers:
            raise ValueError("LightOnOCR output contains no valid grounding markers.")
        if markers and raw[: markers[0].start()].strip():
            raise ValueError(
                "LightOnOCR output contains ungrounded text before the first block."
            )
        starts = {m.start() for m in markers}
        for candidate in MARKER_CANDIDATE.finditer(raw):
            if candidate.start() not in starts:
                raise ValueError(
                    "Malformed LightOnOCR grounding marker: " + candidate.group(0)
                )
        width, height = size
        results = []
        for order, block in enumerate(parse_blocks(raw)):
            if block.label not in LABELS:
                raise ValueError("Unsupported LightOnOCR label: " + block.label)
            x1, y1, x2, y2 = block.bbox
            if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
                raise ValueError(
                    f"Invalid LightOnOCR box for {block.label}: {block.bbox}"
                )
            if block.continues and order == 0:
                raise ValueError(
                    "First LightOnOCR block cannot continue a preceding block."
                )
            label = LABELS[block.label]
            if block.label == "chart" and re.search(r"<table\b", block.text, re.I):
                label = "Table"
            px1, py1, px2, py2 = block.to_pixels(width, height)
            polygon = [[px1, py1], [px2, py1], [px2, py2], [px1, py2]]
            skipped = label in ("Picture", "Figure")
            rendered = "" if skipped else block_html(label, block.text)
            raw_label = block.label + ("+" if block.continues else "")
            results.append(
                BlockOCRResult(
                    polygon=polygon,
                    label=label,
                    raw_label=raw_label,
                    reading_order=order,
                    html=rendered,
                    skipped=skipped,
                    error=False,
                )
            )
            trace["blocks"].append(
                {
                    "reading_order": order,
                    "canonical_label": label,
                    "raw_label": raw_label,
                    "raw_text": block.text if self.lighton_record_traces else None,
                    "normalized_bbox": list(block.bbox),
                    "pixel_polygon": polygon,
                    "continues": block.continues,
                    "skipped": skipped,
                }
            )
        return PageOCRResult(blocks=results, image_bbox=[0, 0, width, height])


def is_blank_image(image):
    """Accept empty recognition only for an exactly uniform white image."""
    rgb = image.convert("RGB")
    return (
        ImageChops.difference(rgb, Image.new("RGB", rgb.size, "white")).getbbox()
        is None
    )


class LocalLightOnClient:
    """Load and run native Qwen-based LightOnOCR checkpoints on demand."""

    def __init__(self, settings):
        self.settings = settings
        self.model_id = settings.model_id
        self.usage_records = []
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoProcessor,
                Qwen3_5ForConditionalGeneration,
            )
        except ImportError as error:
            raise ImportError(
                "LightOnOCR requires the lighton optional dependencies (Transformers >= 5.5.4 and accelerate)."
            ) from error
        settings = self.settings
        options = {"local_files_only": settings.lighton_local_files_only}
        if settings.lighton_revision is not None:
            options["revision"] = settings.lighton_revision
        config = AutoConfig.from_pretrained(self.model_id, **options)
        if config.model_type != "qwen3_5":
            raise ValueError(
                "Only Qwen-based LightOnOCR-3 0.8B/4B checkpoints are supported; the 1B architecture is not supported."
            )
        device = settings.lighton_device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("LightOnOCR supports CPU and CUDA devices.")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("LightOnOCR CUDA was requested but CUDA is unavailable.")
        dtype_name = settings.lighton_dtype
        if dtype_name == "auto":
            dtype_name = "bfloat16" if self.device.type == "cuda" else "float32"
        dtypes = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        if dtype_name not in dtypes:
            raise ValueError(
                "lighton_dtype must be auto, bfloat16, float16, or float32."
            )
        self.dtype = dtypes[dtype_name]
        started = time.perf_counter()
        model, loading = Qwen3_5ForConditionalGeneration.from_pretrained(
            self.model_id,
            config=config,
            dtype=self.dtype,
            device_map=str(self.device),
            attn_implementation="sdpa",
            output_loading_info=True,
            **options,
        )
        failures = {
            key: loading[key]
            for key in (
                "missing_keys",
                "unexpected_keys",
                "mismatched_keys",
                "error_msgs",
            )
            if loading.get(key)
        }
        if failures:
            raise RuntimeError(
                f"LightOnOCR checkpoint has unmatched weights: {failures}"
            )
        processor = AutoProcessor.from_pretrained(self.model_id, **options)
        model.eval()
        self.model, self.processor, self.torch = model, processor, torch
        self.context_limit = model.config.text_config.max_position_embeddings
        self.load_seconds = time.perf_counter() - started

    def infer(self, image, label):
        if label != "grounding":
            raise ValueError(
                "LightOnOCR recognition supports only the grounding prompt."
            )
        self._load()
        torch, settings = self.torch, self.settings
        original_size = image.size
        resized = image.convert("RGB")
        resized.thumbnail(
            (settings.lighton_max_image_size, settings.lighton_max_image_size),
            Image.Resampling.LANCZOS,
        )
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": resized},
                    {"type": "text", "text": "grounding"},
                ],
            }
        ]
        started = time.perf_counter()
        inputs = self.processor.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False,
        )
        inputs = {
            key: value.to(device=self.device, dtype=self.dtype)
            if value.is_floating_point()
            else value.to(self.device)
            for key, value in inputs.items()
        }
        input_count = inputs["input_ids"].shape[1]
        allowance = min(
            settings.lighton_max_new_tokens, self.context_limit - input_count - 1
        )
        if allowance <= 0:
            raise RuntimeError(
                "The image prompt exceeds the LightOnOCR context window."
            )
        generation = {
            "max_new_tokens": allowance,
            "do_sample": settings.lighton_temperature > 0,
            "use_cache": True,
        }
        if settings.lighton_temperature > 0:
            generation.update(
                temperature=settings.lighton_temperature,
                top_p=settings.lighton_top_p,
                top_k=0,
            )
        # fork_rng restores the caller's random state after deterministic OCR sampling.
        devices = (
            [
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            ]
            if self.device.type == "cuda"
            else []
        )
        with torch.random.fork_rng(devices=devices), torch.inference_mode():
            torch.manual_seed(settings.lighton_seed)
            generated = self.model.generate(**inputs, **generation)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        tokens = generated[0, input_count:]
        text = self.processor.decode(tokens, skip_special_tokens=True)
        eos = self.model.generation_config.eos_token_id
        eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        ended = bool(tokens.numel()) and tokens[-1].item() in eos_ids
        record = {
            "call": len(self.usage_records) + 1,
            "label": label,
            "original_pixels": original_size,
            "model_pixels": resized.size,
            "input_tokens": input_count,
            "output_tokens": tokens.numel(),
            "output_allowance": allowance,
            "ended_with_eos": ended,
            "limit_reached": tokens.numel() >= allowance and not ended,
            "truncated": not ended,
            "seconds": time.perf_counter() - started,
        }
        self.usage_records.append(record)
        if not ended:
            raise RuntimeError(
                "LightOnOCR output did not terminate with EOS; truncated recognition is rejected."
            )
        return text
