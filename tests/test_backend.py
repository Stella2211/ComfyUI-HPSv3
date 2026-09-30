import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

ROOT = Path(__file__).parents[1]


class FamilyBackendTests:
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models" / self.family
        self.models.mkdir(parents=True)
        folder = types.ModuleType("folder_paths")
        folder.models_dir = str(self.models.parent)
        folder.add_model_folder_path = mock.Mock()
        folder.get_folder_paths = lambda name: [str(self.models.parent / name)]
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: __import__("torch").device("cuda")
        mm.throw_exception_if_processing_interrupted = mock.Mock()
        mm.free_memory = mock.Mock()
        mm.soft_empty_cache = mock.Mock()
        comfy = types.ModuleType("comfy")
        comfy.model_management = mm
        sys.modules.update({"folder_paths": folder, "comfy": comfy, "comfy.model_management": mm})
        package = types.ModuleType("hpsv3_test_backend")
        package.__path__ = [str(ROOT)]
        inference = types.ModuleType("hpsv3_test_backend.inference")
        inference.run_inference = mock.Mock()
        sys.modules.update({package.__name__: package, inference.__name__: inference})
        spec = importlib.util.spec_from_file_location("hpsv3_test_backend.backend", ROOT / "backend.py")
        self.backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.backend
        spec.loader.exec_module(self.backend)
        self.mm = mm
        self.model_class = getattr(self.backend, self.class_name)
        self.folder = folder

    def tearDown(self):
        for name in ("hpsv3_test_backend.backend", "hpsv3_test_backend.inference", "hpsv3_test_backend", "folder_paths", "comfy", "comfy.model_management"):
            sys.modules.pop(name, None)
        self.tmp.cleanup()

    def make_model(self, path=None):
        path = path or self.models / "model"
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": self.model_type, "quantization_config": {"quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4", "load_in_4bit": True}}), encoding="utf-8")
        for name in ("reward_config.json", "tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
            (path / name).write_text("{}", encoding="utf-8")
        (path / "model.safetensors").write_bytes(b"weights")
        return path

    def hf_modules(self, hub):
        utils = types.ModuleType("huggingface_hub.utils")
        class Tqdm:
            def update(self, n=1):
                return None
        utils.tqdm = Tqdm
        hub.utils = utils
        return {"huggingface_hub": hub, "huggingface_hub.utils": utils}

    def test_download_uses_direct_huggingface_snapshot_and_publishes_after_validation(self):
        staging = self.models / self.model_class.staging_name()
        hub = types.ModuleType("huggingface_hub")
        def snapshot_download(**kwargs):
            self.assertEqual(kwargs["repo_id"], self.model_class.repo)
            self.assertEqual(kwargs["max_workers"], 1)
            self.assertNotIn("*.py", kwargs["allow_patterns"])
            self.make_model(Path(kwargs["local_dir"]))
            return kwargs["local_dir"]
        hub.snapshot_download = snapshot_download
        with mock.patch.dict(sys.modules, self.hf_modules(hub)):
            result = self.model_class._download_default_model()
        self.assertEqual(result, self.models / self.model_class.default_name)
        self.assertFalse(staging.exists())

    def test_download_checks_cancellation_before_and_after(self):
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda **kw: self.make_model(Path(kw["local_dir"]))
        with mock.patch.dict(sys.modules, self.hf_modules(hub)):
            self.model_class._download_default_model()
        self.assertEqual(self.mm.throw_exception_if_processing_interrupted.call_count, 2)

    def test_failed_download_keeps_staging_for_retry(self):
        staging = self.models / self.model_class.staging_name()
        staging.mkdir()
        (staging / "partial.safetensors").write_bytes(b"partial")
        hub = types.ModuleType("huggingface_hub")
        def fail(**kwargs):
            raise RuntimeError("network failure")
        hub.snapshot_download = fail
        with mock.patch.dict(sys.modules, self.hf_modules(hub)), self.assertRaisesRegex(RuntimeError, "network failure"):
            self.model_class._download_default_model()
        self.assertTrue((staging / "partial.safetensors").is_file())

    def test_progress_cancellation_aborts_before_publication(self):
        staging = self.models / self.model_class.staging_name()
        hub = types.ModuleType("huggingface_hub")
        def download(**kwargs):
            staging.mkdir(parents=True)
            (staging / "partial.safetensors").write_bytes(b"partial")
            kwargs["tqdm_class"]().update(1)
        hub.snapshot_download = download
        self.mm.throw_exception_if_processing_interrupted.side_effect = [None, KeyboardInterrupt]
        with mock.patch.dict(sys.modules, self.hf_modules(hub)), self.assertRaises(KeyboardInterrupt):
            self.model_class._download_default_model()
        self.assertTrue(staging.is_dir())
        self.assertFalse((self.models / self.model_class.default_name).exists())

    def test_inference_is_direct_and_never_downloads(self):
        self.make_model()
        image = Image.new("RGB", (2, 2))
        with mock.patch.object(self.backend, "run_inference", return_value=[0.5]) as run:
            model = self.model_class("model")
            self.assertEqual(model.score([image], ["prompt"]), [0.5])
        run.assert_called_once()
        args, kwargs = run.call_args
        self.assertEqual(args[:3], (self.family, (self.models / "model").resolve(), "score"))
        self.assertIs(args[3][0], image)
        self.assertEqual(kwargs["prompts"], ["prompt"])

    def test_model_resolution_rejects_traversal_and_wrong_architecture(self):
        self.make_model()
        with self.assertRaises(ValueError):
            self.model_class.resolve("../outside")
        wrong = self.models / "wrong"
        self.make_model(wrong)
        config = json.loads((wrong / "config.json").read_text(encoding="utf-8"))
        config["model_type"] = self.wrong_model_type
        (wrong / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, self.architecture):
            self.model_class.resolve("wrong")

    def test_model_resolution_rejects_incomplete_shards(self):
        self.make_model()
        broken = self.models / "broken"
        self.make_model(broken)
        (broken / "model.safetensors").unlink()
        (broken / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "missing.safetensors"}}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.model_class.resolve("broken")
        malformed = self.models / "malformed"
        self.make_model(malformed)
        (malformed / "model.safetensors").unlink()
        (malformed / "model.safetensors.index.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.model_class.resolve("malformed")

    def test_model_resolution_rejects_escape_shard(self):
        escaping = self.models / "escaping"
        self.make_model(escaping)
        (escaping / "model.safetensors").unlink()
        (escaping / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "../outside.safetensors"}}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.model_class.resolve("escaping")

    def test_model_listing_ignores_hidden_staging_directories(self):
        self.make_model()
        self.make_model(self.models / ".partial")
        self.assertEqual(self.model_class.list_models(), [self.model_class.default_name, "model"])

    def test_model_folder_is_registered(self):
        self.folder.add_model_folder_path.assert_any_call(self.family, str(self.models))


class HPSv3PPBackendTests(FamilyBackendTests, unittest.TestCase):
    class_name = "HPSv3PPModel"
    family = "hpsv3pp"
    model_type = "qwen3_vl"
    wrong_model_type = "qwen2_vl"
    architecture = "Qwen3-VL"


class HPSv3BackendTests(FamilyBackendTests, unittest.TestCase):
    class_name = "HPSv3Model"
    family = "hpsv3"
    model_type = "qwen2_vl"
    wrong_model_type = "qwen3_vl"
    architecture = "Qwen2-VL"


if __name__ == "__main__":
    unittest.main()
