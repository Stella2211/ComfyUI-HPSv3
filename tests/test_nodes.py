import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import torch
from PIL import Image


class FakeModel:
    model_name = "fake-model"

    def __init__(self, scores=None, captions=None):
        self.scores = scores if scores is not None else [0.25]
        self.captions = captions if captions is not None else ["caption"]
        self.score_calls = []
        self.caption_calls = []

    def score(self, images, prompts):
        self.score_calls.append((images, prompts))
        return self.scores

    def caption(self, images, max_new_tokens):
        self.caption_calls.append((images, max_new_tokens))
        return self.captions


class NodesTests(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.TemporaryDirectory()
        self.disable_metadata = False
        self.save_args = []
        self.save_counter = 0

        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_output_directory = lambda: self.output_dir.name

        def get_save_image_path(prefix, output_dir, width=0, height=0):
            self.save_args.append((prefix, output_dir, width, height))
            path = Path(output_dir) / "nested"
            path.mkdir(parents=True, exist_ok=True)
            counter = self.save_counter
            self.save_counter += 1
            return str(path), Path(prefix).name, counter, "nested", Path(prefix).name

        folder_paths.get_save_image_path = get_save_image_path
        cli_args = types.ModuleType("comfy.cli_args")
        cli_args.args = types.SimpleNamespace(disable_metadata=False)
        comfy = types.ModuleType("comfy")
        comfy.cli_args = cli_args

        backend = types.ModuleType("hpsv3_test.backend")
        backend.HPSv3PPModel = lambda name: FakeModel()
        backend.list_models = lambda: ["fake-model"]
        package = types.ModuleType("hpsv3_test")
        package.__path__ = [str(Path(__file__).parents[1])]
        sys.modules.update({
            "folder_paths": folder_paths,
            "comfy": comfy,
            "comfy.cli_args": cli_args,
            "hpsv3_test": package,
            "hpsv3_test.backend": backend,
        })

        path = Path(__file__).parents[1] / "nodes.py"
        spec = importlib.util.spec_from_file_location("hpsv3_test.nodes", path)
        self.nodes = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.nodes
        spec.loader.exec_module(self.nodes)
        self.nodes.args = cli_args.args

    def tearDown(self):
        self.output_dir.cleanup()
        for name in ("hpsv3_test.nodes", "hpsv3_test.backend", "hpsv3_test", "folder_paths", "comfy", "comfy.cli_args"):
            sys.modules.pop(name, None)

    @staticmethod
    def image(value, height=2, width=3):
        return torch.full((1, height, width, 3), value, dtype=torch.float32)

    def test_score_pairs_prompts_and_writes_metadata_per_image(self):
        model = FakeModel(scores=[1.5, 2.5])
        result = self.nodes.HPSv3PPScore().score(
            [model], [self.image(0.1), self.image(0.2)], ["first", "second"], ["metadata"], ["scores/out"],
            {"workflow": "yes"}, {"extra": 7},
        )
        self.assertEqual(model.score_calls[0][1], ["first", "second"])
        self.assertEqual(result["result"][1], [1.5, 2.5])
        files = sorted(Path(self.output_dir.name).rglob("*.png"))
        self.assertEqual(len(files), 2)
        for path, prompt, score in zip(files, ("first", "second"), (1.5, 2.5)):
            with Image.open(path) as image:
                data = json.loads(image.info["hpsv3pp"])
                self.assertEqual(json.loads(image.info["prompt"]), {"workflow": "yes"})
                self.assertEqual(json.loads(image.info["extra"]), 7)
            self.assertEqual(data, {"model": "fake-model", "score": score, "prompt": prompt})

    def test_banner_keeps_original_pixels_and_adds_height(self):
        model = FakeModel(scores=[3.0])
        original = self.image(0.4, height=4, width=5)
        result = self.nodes.HPSv3PPScore().score([model], [original], ["prompt"], ["banner"], ["banner"])
        rendered = result["result"][0][0]
        self.assertEqual(rendered.ndim, 4)
        rendered = rendered[0]
        self.assertEqual(rendered.shape[1], 5)
        self.assertGreater(rendered.shape[0], 4)
        torch.testing.assert_close(rendered[-4:], original[0], atol=1 / 255, rtol=0)
        self.assertEqual(self.save_args[0][2:4], (5, rendered.shape[0]))
        saved = next(Path(self.output_dir.name).rglob("*.png"))
        with Image.open(saved) as image:
            self.assertEqual(image.size, (5, rendered.shape[0]))

    def test_both_mode_renders_banner_and_preserves_score_metadata(self):
        model = FakeModel(scores=[4.25])
        original = self.image(0.3, height=3, width=6)
        result = self.nodes.HPSv3PPScore().score([model], [original], ["both prompt"], ["both"], ["both"])
        rendered = result["result"][0][0]
        self.assertEqual(rendered.ndim, 4)
        rendered = rendered[0]
        self.assertEqual(rendered.shape[1], 6)
        self.assertGreater(rendered.shape[0], 3)
        torch.testing.assert_close(rendered[-3:], original[0], atol=1 / 255, rtol=0)
        saved = next(Path(self.output_dir.name).rglob("*.png"))
        with Image.open(saved) as image:
            self.assertEqual(image.size, (6, rendered.shape[0]))
            self.assertEqual(json.loads(image.info["hpsv3pp"]), {
                "model": "fake-model", "score": 4.25, "prompt": "both prompt",
            })

    def test_disabled_metadata_blocks_metadata_before_model_but_banner_is_allowed(self):
        self.nodes.args.disable_metadata = True
        model = FakeModel(scores=[1.0])
        with self.assertRaisesRegex(ValueError, "Metadata is disabled"):
            self.nodes.HPSv3PPScore().score([model], [self.image(0.1)], ["p"], ["metadata"], ["x"])
        with self.assertRaisesRegex(ValueError, "Metadata is disabled"):
            self.nodes.HPSv3PPScore().score([model], [self.image(0.1)], ["p"], ["both"], ["x"])
        self.assertEqual(model.score_calls, [])
        self.nodes.HPSv3PPScore().score([model], [self.image(0.1)], ["p"], ["banner"], ["x"])
        self.assertEqual(len(model.score_calls), 1)
        saved = next(Path(self.output_dir.name).rglob("*.png"))
        with Image.open(saved) as image:
            self.assertNotIn("hpsv3pp", image.info)

    def test_count_and_finite_errors_happen_before_saving(self):
        model = FakeModel(scores=[1.0])
        with self.assertRaisesRegex(ValueError, "images but 3 prompts"):
            self.nodes.HPSv3PPScore().score([model], [self.image(0.1), self.image(0.2)], ["a", "b", "c"], ["metadata"], ["x"])
        model.scores = [float("nan")]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.nodes.HPSv3PPScore().score([model], [self.image(0.1)], ["a"], ["metadata"], ["x"])
        self.assertEqual(list(Path(self.output_dir.name).rglob("*.png")), [])

    def test_caption_returns_one_caption_per_image_and_forwards_token_limit(self):
        model = FakeModel(captions=["one", "two"])
        result = self.nodes.HPSv3PPCaption().caption(model, torch.cat([self.image(0.1), self.image(0.2)]), 128)
        self.assertEqual(result, (["one", "two"],))
        self.assertEqual(len(model.caption_calls[0][0]), 2)
        self.assertEqual(model.caption_calls[0][1], 128)


if __name__ == "__main__":
    unittest.main()
