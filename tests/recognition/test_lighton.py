import html
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from marker.recognition.lighton import LightOnRecognitionPredictor, LocalLightOnClient


class Backend:
    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    def infer(self, image, label):
        self.calls.append((image.size, label))
        return self.raw


class AdapterTests(unittest.TestCase):
    def convert(self, raw):
        backend = Backend(raw)
        adapter = LightOnRecognitionPredictor(
            {"lighton_record_traces": True}, backend=backend
        )
        result = adapter([Image.new("RGB", (2000, 1000))])[0]
        self.assertEqual(backend.calls, [((2000, 1000), "grounding")])
        return result, adapter

    def test_coordinates_order_continuation(self):
        result, adapter = self.convert(
            "![text](100,200,400,700) Alpha\n![text+](500,100,900,800) beta"
        )
        self.assertEqual(result.image_bbox, [0, 0, 2000, 1000])
        self.assertEqual(
            result.blocks[0].polygon, [[200, 200], [800, 200], [800, 700], [200, 700]]
        )
        self.assertEqual([b.reading_order for b in result.blocks], [0, 1])
        self.assertEqual(result.blocks[1].raw_label, "text+")
        self.assertTrue(adapter.pages[0]["blocks"][1]["continues"])
        self.assertEqual(adapter.pages[0]["blocks"][1]["raw_text"], "beta")

    def test_code(self):
        source = (
            "Map<String, List<T>> values;\nif (x < 2 && y > 1) {\n    return values;\n}"
        )
        for body in [source, "java\n" + source, "```java\n" + source + "\n```"]:
            with self.subTest(body=body):
                result, _ = self.convert("![code](0,0,1000,1000) " + body)
                self.assertEqual(
                    result.blocks[0].html,
                    "<pre><code>" + html.escape(source, quote=False) + "</code></pre>",
                )

    def test_tables_lists_and_math(self):
        table = "<table>\n<tr><th>A</th><th>B</th></tr>\n<tr><td>3</td><td>4</td></tr>\n</table>"
        result, _ = self.convert(
            "![table](0,0,1000,200) "
            + table
            + "\n![list](0,200,1000,400) - First\n- Second\n![formula](0,400,1000,600) $$x < y$$\n![chart](0,600,1000,900) "
            + table
        )
        self.assertEqual(result.blocks[0].html, table)
        self.assertEqual(result.blocks[1].label, "ListGroup")
        self.assertIn("<li>First</li>", result.blocks[1].html)
        self.assertIn("<li>Second</li>", result.blocks[1].html)
        self.assertEqual(result.blocks[2].html, '<math display="block">x &lt; y</math>')
        self.assertEqual(result.blocks[3].label, "Table")
        self.assertEqual(result.blocks[3].html, table)
        self.assertFalse(result.blocks[3].skipped)

    def test_visuals_and_header_footer(self):
        result, adapter = self.convert(
            "![image](10,20,300,400) Raw image description\n![chart](400,20,900,400) Raw chart description\n![header_image](10,0,300,20) Logo\n![footer_image](10,900,300,1000) Seal\n![aside_text](0,400,500,700) Note"
        )
        self.assertEqual(
            [b.label for b in result.blocks],
            ["Picture", "Figure", "PageHeader", "PageFooter", "Text"],
        )
        for block in result.blocks[:2]:
            self.assertTrue(block.skipped)
            self.assertEqual(block.html, "")
            self.assertGreater(block.polygon[2][0], block.polygon[0][0])
        self.assertEqual(
            adapter.pages[0]["blocks"][0]["raw_text"], "Raw image description"
        )

    def test_reject_invalid_output(self):
        invalid = [
            "![mystery](0,0,100,100) unknown",
            "![text](400,0,100,100) inverted",
            "![text](0,0,1001,100) outside",
            "![text](0,0,0,100) empty",
            "![text](-1,0,100,100) negative",
            "![text](0,0,100,100) valid\n![text](-1,0,100,100) negative",
            "![text](0,0,100,100) valid\n![text](0.5,0,100,100) fractional",
            "![text](0,0,100,100) valid\n![text](0,0,100 malformed",
            "ungrounded text",
            "prefix\n![text](0,0,100,100) content",
            "![text+](0,0,100,100) orphan",
        ]
        for raw in invalid:
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.convert(raw)

    def test_full_page_guard_and_trace(self):
        backend = Backend("![text](0,0,100,100) Saved")
        adapter = LightOnRecognitionPredictor(
            {"lighton_record_traces": True}, backend=backend
        )
        with self.assertRaisesRegex(ValueError, "full-page"):
            adapter([], full_page=False)
        self.assertEqual(backend.calls, [])
        adapter([Image.new("RGB", (2000, 1000))])
        self.assertEqual(adapter.pages[0]["raw_grounding"], backend.raw)

    def test_empty_nonblank_rejected(self):
        predictor = LightOnRecognitionPredictor(backend=Backend(""))
        with self.assertRaisesRegex(ValueError, "nonblank"):
            predictor([Image.new("RGB", (5, 5), "black")])
        self.assertEqual(predictor([Image.new("RGB", (5, 5), "white")])[0].blocks, [])

    def test_constructor_does_not_load(self):
        with patch.object(
            LocalLightOnClient, "_load", side_effect=AssertionError("Unexpected load")
        ):
            predictor = LightOnRecognitionPredictor(
                {"lighton_model": "/models/local-checkpoint"}
            )
            self.assertEqual(predictor.model_id, "/models/local-checkpoint")
            self.assertEqual(predictor.usage_records, [])

    def test_native_loading_contract(self):
        import torch

        config = SimpleNamespace(
            model_type="qwen3_5",
            text_config=SimpleNamespace(max_position_embeddings=32768),
        )
        model = Mock(config=config)
        config_loader = Mock(return_value=config)
        model_loader = Mock(return_value=(model, {}))
        processor_loader = Mock(return_value=Mock())
        module = SimpleNamespace(
            AutoConfig=SimpleNamespace(from_pretrained=config_loader),
            AutoProcessor=SimpleNamespace(from_pretrained=processor_loader),
            Qwen3_5ForConditionalGeneration=SimpleNamespace(
                from_pretrained=model_loader
            ),
        )
        predictor = LightOnRecognitionPredictor(
            {
                "lighton_device": "cpu",
                "lighton_revision": "pinned",
                "lighton_local_files_only": True,
            }
        )
        with patch.dict(sys.modules, {"transformers": module}):
            predictor.backend._load()
        kwargs = model_loader.call_args.kwargs
        self.assertEqual(kwargs["device_map"], "cpu")
        self.assertEqual(kwargs["dtype"], torch.float32)
        self.assertEqual(kwargs["revision"], "pinned")
        self.assertTrue(kwargs["local_files_only"])
        self.assertTrue(kwargs["output_loading_info"])
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertEqual(
            processor_loader.call_args.kwargs,
            {"revision": "pinned", "local_files_only": True},
        )
        model.eval.assert_called_once()
        # Unsupported architecture fails before any weights are requested.
        config.model_type = "mistral3"
        model_loader.reset_mock()
        with (
            patch.dict(sys.modules, {"transformers": module}),
            self.assertRaisesRegex(ValueError, "1B"),
        ):
            LightOnRecognitionPredictor().backend._load()
        model_loader.assert_not_called()
        config.model_type = "qwen3_5"
        model_loader.return_value = (model, {"missing_keys": ["unloaded.weight"]})
        with (
            patch.dict(sys.modules, {"transformers": module}),
            self.assertRaisesRegex(RuntimeError, "unmatched weights"),
        ):
            LightOnRecognitionPredictor({"lighton_device": "cpu"}).backend._load()

    def test_generation_eos_and_truncation(self):
        import torch

        predictor = LightOnRecognitionPredictor(
            {"lighton_device": "cpu", "lighton_max_new_tokens": 2}
        )
        client = predictor.backend
        client.torch = torch
        client.device = torch.device("cpu")
        client.dtype = torch.float32
        client.context_limit = 100
        client.model = Mock(generation_config=SimpleNamespace(eos_token_id=[9]))
        client.model.generate.return_value = torch.tensor([[1, 2, 7, 9]])
        client.processor = Mock()
        client.processor.apply_chat_template.return_value = {
            "input_ids": torch.tensor([[1, 2]])
        }
        client.processor.decode.return_value = "![text](0,0,100,100) Content"
        output = client.infer(Image.new("RGB", (5, 5)), "grounding")
        self.assertIn("Content", output)
        self.assertTrue(client.usage_records[-1]["ended_with_eos"])
        self.assertFalse(client.usage_records[-1]["truncated"])
        self.assertFalse(
            client.processor.apply_chat_template.call_args.kwargs["enable_thinking"]
        )
        client.model.generate.return_value = torch.tensor([[1, 2, 7, 8]])
        with self.assertRaisesRegex(RuntimeError, "truncated"):
            client.infer(Image.new("RGB", (5, 5)), "grounding")
        self.assertTrue(client.usage_records[-1]["limit_reached"])
        self.assertTrue(client.usage_records[-1]["truncated"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
