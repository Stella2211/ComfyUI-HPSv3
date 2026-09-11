import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch  # Load before patch.dict restores sys.modules; torch cannot be reimported.


ROOT = Path(__file__).parents[1]


class LoaderIntegrationTests(unittest.TestCase):
    def setUp(self):
        workspace = tempfile.TemporaryDirectory()
        self.addCleanup(workspace.cleanup)
        self.models = Path(workspace.name) / "models" / "hpsv3pp"
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.models_dir = str(self.models.parent)
        folder_paths.add_model_folder_path = mock.Mock()
        folder_paths.get_folder_paths = lambda name: [str(self.models)]
        comfy = types.ModuleType("comfy")
        comfy.model_management = types.ModuleType("comfy.model_management")
        comfy.model_management.throw_exception_if_processing_interrupted = mock.Mock()
        cli_args = types.ModuleType("comfy.cli_args")
        cli_args.args = types.SimpleNamespace(disable_metadata=False)
        package = types.ModuleType("hpsv3_loader_integration")
        package.__path__ = [str(ROOT)]
        modules = mock.patch.dict(sys.modules, {
            "folder_paths": folder_paths,
            "comfy": comfy,
            "comfy.model_management": comfy.model_management,
            "comfy.cli_args": cli_args,
            package.__name__: package,
        })
        modules.start()
        self.addCleanup(modules.stop)
        for name in ("backend", "nodes"):
            spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", ROOT / f"{name}.py")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            setattr(self, name, module)

    def test_empty_install_exposes_default_without_starting_download(self):
        choices = self.nodes.HPSv3PPModelLoader.INPUT_TYPES()["required"]["model"][0]
        self.assertIn("HPSv3-PlusPlus-bnb-NF4", choices)
        self.assertFalse(self.models.exists())

    def test_loader_downloads_once_and_returns_reusable_model_socket(self):
        name = "HPSv3-PlusPlus-bnb-NF4"
        model_path = self.models / name

        def download():
            model_path.mkdir(parents=True)
            config = {"model_type": "qwen3_vl", "quantization_config": {
                "quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4", "load_in_4bit": True,
            }}
            (model_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
            for filename in ("reward_config.json", "tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
                (model_path / filename).write_text("{}", encoding="utf-8")
            (model_path / "model.safetensors").write_bytes(b"test weights")
            return model_path

        loader = self.nodes.HPSv3PPModelLoader()
        with mock.patch.object(self.backend, "_download_default_model", side_effect=download) as downloader:
            first, = loader.load(name)
            second, = loader.load([name])
        downloader.assert_called_once()
        self.assertEqual(loader.RETURN_TYPES, ("HPSV3PP_MODEL",))
        self.assertIsInstance(first, self.backend.HPSv3PPModel)
        self.assertEqual(first.model_name, name)
        self.assertEqual(second.model_name, name)


if __name__ == "__main__":
    unittest.main()
