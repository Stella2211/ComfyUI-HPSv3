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


class BackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.models = Path(self.tmp.name) / "models" / "hpsv3"
        self.models.mkdir(parents=True)
        folder = types.ModuleType("folder_paths")
        folder.models_dir = str(self.models.parent)
        folder.add_model_folder_path = mock.Mock()
        folder.get_folder_paths = lambda _: [str(self.models)]
        mm = types.ModuleType("comfy.model_management")
        mm.get_torch_device = lambda: __import__("torch").device("cuda")
        mm.throw_exception_if_processing_interrupted = mock.Mock()
        mm.free_memory = mock.Mock()
        mm.soft_empty_cache = mock.Mock()
        comfy = types.ModuleType("comfy")
        comfy.model_management = mm
        sys.modules.update({"folder_paths": folder, "comfy": comfy, "comfy.model_management": mm})
        spec = importlib.util.spec_from_file_location("hpsv3_test_backend_hpsv3", ROOT / "backend_hpsv3.py")
        self.backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.backend
        spec.loader.exec_module(self.backend)
        self.mm = mm

    def tearDown(self):
        for name in ("hpsv3_test_backend_hpsv3", "folder_paths", "comfy", "comfy.model_management"):
            sys.modules.pop(name, None)
        self.tmp.cleanup()

    def make_model(self, path=None):
        path = path or self.models / "model"
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"model_type": "qwen2_vl", "quantization_config": {"quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4", "load_in_4bit": True}}), encoding="utf-8")
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
        staging = self.models / self.backend.DOWNLOAD_STAGING_NAME
        hub = types.ModuleType("huggingface_hub")
        def snapshot_download(**kwargs):
            self.assertEqual(kwargs["repo_id"], self.backend.DEFAULT_MODEL_REPO)
            self.assertEqual(kwargs["max_workers"], 1)
            self.assertNotIn("*.py", kwargs["allow_patterns"])
            self.make_model(Path(kwargs["local_dir"]))
            return kwargs["local_dir"]
        hub.snapshot_download = snapshot_download
        with mock.patch.dict(sys.modules, self.hf_modules(hub)):
            result = self.backend._download_default_model()
        self.assertEqual(result, self.models / self.backend.DEFAULT_MODEL_NAME)
        self.assertFalse(staging.exists())

    def test_download_checks_cancellation_before_and_after(self):
        hub = types.ModuleType("huggingface_hub")
        hub.snapshot_download = lambda **kw: self.make_model(Path(kw["local_dir"]))
        with mock.patch.dict(sys.modules, self.hf_modules(hub)):
            self.backend._download_default_model()
        self.assertEqual(self.mm.throw_exception_if_processing_interrupted.call_count, 2)

    def test_failed_download_keeps_staging_for_retry(self):
        staging = self.models / self.backend.DOWNLOAD_STAGING_NAME
        staging.mkdir()
        (staging / "partial.safetensors").write_bytes(b"partial")
        hub = types.ModuleType("huggingface_hub")
        def fail(**kwargs):
            raise RuntimeError("network failure")
        hub.snapshot_download = fail
        with mock.patch.dict(sys.modules, self.hf_modules(hub)), self.assertRaisesRegex(RuntimeError, "network failure"):
            self.backend._download_default_model()
        self.assertTrue((staging / "partial.safetensors").is_file())

    def test_progress_cancellation_aborts_before_publication(self):
        staging = self.models / self.backend.DOWNLOAD_STAGING_NAME
        hub = types.ModuleType("huggingface_hub")
        def download(**kwargs):
            staging.mkdir(parents=True)
            (staging / "partial.safetensors").write_bytes(b"partial")
            kwargs["tqdm_class"]().update(1)
        hub.snapshot_download = download
        self.mm.throw_exception_if_processing_interrupted.side_effect = [None, KeyboardInterrupt]
        with mock.patch.dict(sys.modules, self.hf_modules(hub)), self.assertRaises(KeyboardInterrupt):
            self.backend._download_default_model()
        self.assertTrue(staging.is_dir())
        self.assertFalse((self.models / self.backend.DEFAULT_MODEL_NAME).exists())

    def test_inference_is_direct_and_never_downloads(self):
        self.make_model()
        fake = mock.Mock(return_value=[0.5])
        image = Image.new("RGB", (2, 2))
        with mock.patch.object(self.backend, "_run_inference", side_effect=fake):
            model = self.backend.HPSv3Model("model")
            self.assertEqual(model.score([image], ["prompt"]), [0.5])
        fake.assert_called_once()
        args, kwargs = fake.call_args
        self.assertEqual(args[:3], ("hpsv3", (self.models / "model").resolve(), "score"))
        self.assertIs(args[3][0], image)

    def test_model_resolution_rejects_traversal_and_wrong_architecture(self):
        self.make_model()
        with self.assertRaises(ValueError):
            self.backend.resolve_model("../outside")
        wrong = self.models / "wrong"
        self.make_model(wrong)
        config = json.loads((wrong / "config.json").read_text(encoding="utf-8"))
        config["model_type"] = "qwen3_vl"
        (wrong / "config.json").write_text(json.dumps(config), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Qwen2-VL"):
            self.backend.resolve_model("wrong")

    def test_model_resolution_rejects_escape_shard(self):
        broken = self.models / "broken"
        self.make_model(broken)
        (broken / "model.safetensors").unlink()
        (broken / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"x": "../outside.safetensors"}}), encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.backend.resolve_model("broken")
        malformed = self.models / "malformed"
        self.make_model(malformed)
        (malformed / "model.safetensors").unlink()
        (malformed / "model.safetensors.index.json").write_text("[]", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.backend.resolve_model("malformed")


if __name__ == "__main__":
    unittest.main()
