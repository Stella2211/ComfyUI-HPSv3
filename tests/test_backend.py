import importlib.util
import json
import os
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
        self.workspace = tempfile.TemporaryDirectory()
        self.models = Path(self.workspace.name) / "models" / "hpsv3pp"
        self.models.mkdir(parents=True)
        self.folder_paths = types.ModuleType("folder_paths")
        self.folder_paths.models_dir = str(Path(self.workspace.name) / "models")
        self.folder_paths.add_model_folder_path = mock.Mock()
        self.folder_paths.get_folder_paths = lambda name: [str(self.models)]
        self.model_management = types.ModuleType("comfy.model_management")
        self.model_management.get_torch_device = lambda: __import__("torch").device("cuda")
        self.model_management.throw_exception_if_processing_interrupted = mock.Mock()
        self.model_management.free_memory = mock.Mock()
        self.model_management.soft_empty_cache = mock.Mock()
        comfy = types.ModuleType("comfy")
        comfy.model_management = self.model_management
        sys.modules.update({
            "folder_paths": self.folder_paths,
            "comfy": comfy,
            "comfy.model_management": self.model_management,
        })
        spec = importlib.util.spec_from_file_location("hpsv3_test_backend", ROOT / "backend.py")
        self.backend = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = self.backend
        spec.loader.exec_module(self.backend)

    def tearDown(self):
        sys.modules.pop("hpsv3_test_backend", None)
        for name in ("folder_paths", "comfy", "comfy.model_management"):
            sys.modules.pop(name, None)
        self.workspace.cleanup()

    def make_model(self, name="model"):
        path = self.models / name
        path.mkdir(parents=True)
        (path / "config.json").write_text(json.dumps({
            "quantization_config": {
                "quant_method": "bitsandbytes",
                "bnb_4bit_quant_type": "nf4",
                "load_in_4bit": True,
            }
        }), encoding="utf-8")
        (path / "reward_config.json").write_text("{}", encoding="utf-8")
        return path

    def test_list_models_discovers_only_configured_model_folders(self):
        self.make_model("z-model")
        (self.models / "incomplete").mkdir()
        self.assertEqual(self.backend.list_models(), ["z-model"])

    def test_resolve_model_rejects_escape_and_invalid_or_incomplete_models(self):
        self.make_model("valid")
        with self.assertRaisesRegex(ValueError, "inside models/hpsv3pp"):
            self.backend.resolve_model("../outside")
        invalid = self.models / "invalid"
        invalid.mkdir()
        (invalid / "config.json").write_text(json.dumps({"quantization_config": {"load_in_4bit": True}}), encoding="utf-8")
        (invalid / "reward_config.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bitsandbytes NF4"):
            self.backend.resolve_model("invalid")
        incomplete = self.models / "incomplete"
        incomplete.mkdir()
        (incomplete / "config.json").write_text(json.dumps({
            "quantization_config": {
                "quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4", "load_in_4bit": True
            }
        }), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "reward_config.json"):
            self.backend.resolve_model("incomplete")

    def make_process(self, result=b'{"scores": [1]}', returncode=0, timeout_once=False):
        process = mock.Mock()
        process.returncode = returncode
        process.poll.return_value = returncode
        calls = []

        def communicate(timeout=None):
            calls.append(timeout)
            if timeout_once and len(calls) == 1:
                raise __import__("subprocess").TimeoutExpired("worker", timeout)
            return result, None

        process.communicate.side_effect = communicate
        return process

    def test_run_preserves_image_order_options_and_sets_offline_environment(self):
        model = self.make_model()
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        process = self.make_process(timeout_once=True)
        captured = {}

        def popen(command, **kwargs):
            captured.update(command=command, kwargs=kwargs)
            captured["request"] = json.loads(Path(command[2]).read_text(encoding="utf-8"))
            Path(command[3]).write_text('{"result": [1], "warnings": []}', encoding="utf-8")
            return process

        self.backend.RUNTIME_PYTHON = runtime
        image_a = Image.new("RGB", (2, 2), "red")
        image_b = Image.new("RGB", (2, 2), "blue")
        with mock.patch.object(self.backend.subprocess, "Popen", side_effect=popen):
            result = self.backend.HPSv3PPModel("model")._run("score", [image_a, image_b], prompts=["a", "b"])
        self.assertEqual(result, [1])
        request = captured["request"]
        self.assertEqual(request["operation"], "score")
        self.assertEqual(request["prompts"], ["a", "b"])
        self.assertEqual([Path(path).name for path in request["images"]], ["0.png", "1.png"])
        env = captured["kwargs"]["env"]
        self.assertEqual({key: env[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK")},
                         {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"})
        self.assertEqual(process.communicate.call_args_list[0].kwargs, {"timeout": 1})
        self.assertEqual(process.communicate.call_args_list[-1].args, ())
        self.assertFalse(Path(captured["command"][2]).parent.exists())

    def test_run_relays_worker_warnings_once_and_keeps_result_shape(self):
        self.make_model()
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        process = self.make_process()
        captured = {}
        def popen(command, **kwargs):
            captured["command"] = command
            Path(command[3]).write_text('{"result": [0.75], "warnings": ["dtype mismatch"]}', encoding="utf-8")
            return process
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), \
             mock.patch.object(self.backend.subprocess, "Popen", side_effect=popen), \
             mock.patch.object(self.backend.logger, "warning") as warning:
            result = self.backend.HPSv3PPModel("model").score([Image.new("RGB", (1, 1))], ["p"])
        self.assertEqual(result, [0.75])
        warning.assert_called_once_with("HPSv3++: %s", "dtype mismatch")
        self.assertFalse(Path(captured["command"][2]).parent.exists())

    def test_run_reports_worker_failure_and_removes_temporary_data(self):
        self.make_model()
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        process = self.make_process(result=b"traceback", returncode=3)
        captured = {}
        def popen(command, **kwargs):
            captured["command"] = command
            return process
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), \
             mock.patch.object(self.backend.subprocess, "Popen", side_effect=popen):
            with self.assertRaisesRegex(RuntimeError, r"(?s)inference failed:.*traceback"):
                self.backend.HPSv3PPModel("model").score([Image.new("RGB", (1, 1))], ["p"])
        self.assertFalse(Path(captured["command"][2]).parent.exists())

    def test_run_kills_child_when_processing_is_interrupted(self):
        self.make_model()
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        process = self.make_process()
        process.poll.return_value = None
        self.model_management.throw_exception_if_processing_interrupted.side_effect = [None, KeyboardInterrupt]
        captured = {}
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), \
             mock.patch.object(self.backend.subprocess, "Popen", side_effect=lambda command, **kwargs: (captured.update(command=command), process)[1]):
            with self.assertRaises(KeyboardInterrupt):
                self.backend.HPSv3PPModel("model").score([Image.new("RGB", (1, 1))], ["p"])
        process.kill.assert_called_once_with()
        process.communicate.assert_called_once_with()
        self.assertFalse(Path(captured["command"][2]).parent.exists())


class WorkerTests(unittest.TestCase):
    def test_run_scores_pairs_independently_and_forwards_caption_options(self):
        calls = []
        class Scorer:
            @classmethod
            def from_merged_dir(cls, **kwargs):
                calls.append(("init", kwargs))
                return cls()
            def score(self, images, prompts, iter_step):
                calls.append(("score", images, prompts, iter_step))
                return [len(images) + len(prompts)]
            def caption(self, images, max_new_tokens):
                calls.append(("caption", images, max_new_tokens))
                return ["ok"]
        evaluation = types.ModuleType("evaluation")
        quantized = types.ModuleType("evaluation.hpsv3pp_quantized")
        quantized.HPSv3PPQuantizedInferencer = Scorer
        sys.modules.update({"evaluation": evaluation, "evaluation.hpsv3pp_quantized": quantized})
        spec = importlib.util.spec_from_file_location("hpsv3_test_worker", ROOT / "worker.py")
        worker = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = worker
        spec.loader.exec_module(worker)
        request = {"model": "m", "device": "cuda", "operation": "score", "images": ["i1", "i2"], "prompts": ["p1", "p2"]}
        self.assertEqual(worker.run(request), [2, 2])
        self.assertEqual(calls[1:], [("score", ["i1"], ["p1"], 0.0), ("score", ["i2"], ["p2"], 0.0)])
        self.assertEqual(worker.run({**request, "operation": "caption", "max_new_tokens": 64}), ["ok"])
        self.assertEqual(calls[-1], ("caption", ["i1", "i2"], 64))
        for name in ("evaluation", "evaluation.hpsv3pp_quantized", "hpsv3_test_worker"):
            sys.modules.pop(name, None)

    def test_main_serializes_deduplicated_warnings_with_result(self):
        evaluation = types.ModuleType("evaluation")
        quantized = types.ModuleType("evaluation.hpsv3pp_quantized")
        quantized.HPSv3PPQuantizedInferencer = object
        sys.modules.update({"evaluation": evaluation, "evaluation.hpsv3pp_quantized": quantized})
        spec = importlib.util.spec_from_file_location("hpsv3_test_worker_main", ROOT / "worker.py")
        worker = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = worker
        spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as directory:
            request_path = Path(directory) / "request.json"
            result_path = Path(directory) / "result.json"
            request_path.write_text("{}", encoding="utf-8")
            with mock.patch.object(worker, "run") as run, mock.patch.object(sys, "argv", ["worker.py", str(request_path), str(result_path)]):
                def fake_run(request):
                    import warnings
                    warnings.warn("precision warning")
                    warnings.warn("precision warning")
                    return [1.25]
                run.side_effect = fake_run
                worker.main()
            self.assertEqual(json.loads(result_path.read_text(encoding="utf-8")), {"result": [1.25], "warnings": ["precision warning"]})
        for name in ("evaluation", "evaluation.hpsv3pp_quantized", "hpsv3_test_worker_main"):
            sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
