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
        self.assertEqual(self.backend.list_models(), [self.backend.DEFAULT_MODEL_NAME, "z-model"])

    def test_default_missing_model_can_be_downloaded_on_explicit_resolution(self):
        downloaded = self.models / self.backend.DEFAULT_MODEL_NAME
        downloaded.mkdir()
        with mock.patch.object(self.backend, "_download_default_model", return_value=downloaded) as download:
            with mock.patch.object(self.backend, "_validate_download", return_value=True):
                self.assertEqual(self.backend.resolve_model(self.backend.DEFAULT_MODEL_NAME, allow_download=True), downloaded)
        download.assert_called_once_with()

    def test_inference_resolution_never_downloads_missing_default(self):
        with mock.patch.object(self.backend, "_download_default_model") as download:
            with self.assertRaises(FileNotFoundError):
                self.backend.resolve_model(self.backend.DEFAULT_MODEL_NAME)
        download.assert_not_called()

    def test_unknown_and_traversal_names_never_download(self):
        with mock.patch.object(self.backend, "_download_default_model") as download:
            with self.assertRaises(FileNotFoundError):
                self.backend.resolve_model("unknown", allow_download=True)
            with self.assertRaises(ValueError):
                self.backend.resolve_model("../outside", allow_download=True)
            with self.assertRaises(ValueError):
                self.backend.resolve_model(self.backend.DOWNLOAD_STAGING_NAME, allow_download=True)
        download.assert_not_called()

    def make_complete_download(self, path):
        path.mkdir(parents=True, exist_ok=True)
        (path / "config.json").write_text(json.dumps({"quantization_config": {
            "quant_method": "bitsandbytes", "bnb_4bit_quant_type": "nf4", "load_in_4bit": True,
        }}), encoding="utf-8")
        for name in ("reward_config.json", "tokenizer_config.json", "preprocessor_config.json", "tokenizer.json"):
            (path / name).write_text("{}", encoding="utf-8")
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "model-00001-of-00001.safetensors"}}), encoding="utf-8")
        (path / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    def test_download_validation_rejects_missing_index_shard_and_escape(self):
        path = self.models / ".staging"
        self.make_complete_download(path)
        self.assertTrue(self.backend._validate_download(path))
        (path / "model-00001-of-00001.safetensors").write_bytes(b"")
        self.assertFalse(self.backend._validate_download(path))
        (path / "model-00001-of-00001.safetensors").unlink()
        self.assertFalse(self.backend._validate_download(path))
        self.make_complete_download(path)
        (self.models / "outside.safetensors").write_bytes(b"outside weights")
        (path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "../outside.safetensors"}}), encoding="utf-8")
        self.assertFalse(self.backend._validate_download(path))

    def test_download_success_publishes_valid_staging_and_reuses_it(self):
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        staging = self.models / self.backend.DOWNLOAD_STAGING_NAME
        process = mock.Mock(returncode=0)
        process.poll.return_value = 0

        def wait(timeout=None):
            if timeout is not None:
                self.make_complete_download(staging)
            return 0

        process.wait.side_effect = wait
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), mock.patch.object(self.backend.subprocess, "Popen", return_value=process) as popen:
            result = self.backend._download_default_model()
            self.backend.HPSv3PPModel(self.backend.DEFAULT_MODEL_NAME)
        self.assertEqual(result, self.models / self.backend.DEFAULT_MODEL_NAME)
        self.assertTrue((result / "config.json").is_file())
        self.assertFalse(staging.exists())
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(command, [str(runtime), "-I", "-X", "utf8", str(ROOT / "download_model.py"), self.backend.DEFAULT_MODEL_REPO, str(staging)])
        self.assertNotIn("stdout", popen.call_args.kwargs)
        self.assertIs(popen.call_args.kwargs["shell"], False)
        self.assertTrue(process.wait.called)

    def test_download_failure_keeps_staging_for_retry(self):
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        staging = self.models / self.backend.DOWNLOAD_STAGING_NAME
        self.make_complete_download(staging)
        process = mock.Mock(returncode=2)
        process.poll.return_value = 2
        process.wait.return_value = 2
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), mock.patch.object(self.backend.subprocess, "Popen", return_value=process):
            with self.assertRaisesRegex(RuntimeError, "download failed"):
                self.backend._download_default_model()
        self.assertTrue(staging.is_dir())
        self.assertFalse((self.models / self.backend.DEFAULT_MODEL_NAME).exists())
        process.returncode = 0
        process.poll.return_value = 0
        process.wait.return_value = 0
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), mock.patch.object(self.backend.subprocess, "Popen", return_value=process):
            result = self.backend._download_default_model()
        self.assertTrue((result / "model-00001-of-00001.safetensors").is_file())
        self.assertFalse(staging.exists())

    def test_download_cancellation_kills_and_reaps_child(self):
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        process = mock.Mock(returncode=None)
        process.poll.return_value = None
        process.wait.side_effect = [__import__("subprocess").TimeoutExpired("download", 1), None]
        self.model_management.throw_exception_if_processing_interrupted.side_effect = [None, None, KeyboardInterrupt]
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), mock.patch.object(self.backend.subprocess, "Popen", return_value=process):
            with self.assertRaises(KeyboardInterrupt):
                self.backend._download_default_model()
        process.kill.assert_called_once_with()
        self.assertGreaterEqual(process.wait.call_count, 2)

    def test_download_cancellation_before_spawn_does_not_start_child(self):
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        self.model_management.throw_exception_if_processing_interrupted.side_effect = KeyboardInterrupt
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), mock.patch.object(self.backend.subprocess, "Popen") as popen:
            with self.assertRaises(KeyboardInterrupt):
                self.backend._download_default_model()
        popen.assert_not_called()

    def test_download_refuses_existing_invalid_target_without_spawning(self):
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        target = self.models / self.backend.DEFAULT_MODEL_NAME
        target.mkdir()
        (target / "user-file").write_text("keep", encoding="utf-8")
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), mock.patch.object(self.backend.subprocess, "Popen") as popen:
            with self.assertRaisesRegex(ValueError, "exists but is incomplete"):
                self.backend._download_default_model()
        popen.assert_not_called()
        self.assertEqual((target / "user-file").read_text(encoding="utf-8"), "keep")

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
            captured["request"] = json.loads(Path(command[5]).read_text(encoding="utf-8"))
            Path(command[6]).write_text('{"result": [1], "warnings": []}', encoding="utf-8")
            return process

        self.backend.RUNTIME_PYTHON = runtime
        image_a = Image.new("RGB", (2, 2), "red")
        image_b = Image.new("RGB", (2, 2), "blue")
        with mock.patch.object(self.backend.subprocess, "Popen", side_effect=popen):
            result = self.backend.HPSv3PPModel("model")._run("score", [image_a, image_b], prompts=["a & $(echo injected); | < >", "b"])
        self.assertEqual(result, [1])
        request = captured["request"]
        self.assertEqual(request["operation"], "score")
        self.assertEqual(request["prompts"], ["a & $(echo injected); | < >", "b"])
        self.assertIs(captured["kwargs"]["shell"], False)
        self.assertEqual(len(captured["command"]), 7)
        self.assertEqual(captured["command"][:5], [str(runtime), "-I", "-X", "utf8", str(ROOT / "worker.py")])
        self.assertNotIn(request["prompts"][0], captured["command"])
        self.assertEqual([Path(path).name for path in request["images"]], ["0.png", "1.png"])
        env = captured["kwargs"]["env"]
        self.assertEqual({key: env[key] for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_HUB_DISABLE_TELEMETRY", "DO_NOT_TRACK")},
                         {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1", "DO_NOT_TRACK": "1"})
        self.assertEqual(process.communicate.call_args_list[0].kwargs, {"timeout": 1})
        self.assertEqual(process.communicate.call_args_list[-1].args, ())
        self.assertFalse(Path(captured["command"][5]).parent.exists())

    def test_real_child_ignores_inherited_python_paths(self):
        self.make_model()
        scripts = Path(self.workspace.name) / "scripts"
        scripts.mkdir()
        injected = Path(self.workspace.name) / "injected"
        injected.mkdir()
        marker = injected / "loaded.txt"
        (injected / "sitecustomize.py").write_text(
            "from pathlib import Path\nPath(__file__).with_name('loaded.txt').write_text('loaded')\n",
            encoding="utf-8",
        )
        (scripts / "worker.py").write_text(
            "import json, os, sys\nfrom pathlib import Path\n"
            "request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))\n"
            "assert request['prompts'] == ['日本語 & $(echo example)']\n"
            "assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
            "assert os.environ['TRANSFORMERS_OFFLINE'] == '1'\n"
            "Path(sys.argv[2]).write_text(json.dumps({'result': [1.0]}), encoding='utf-8')\n",
            encoding="utf-8",
        )
        with mock.patch.object(self.backend, "ROOT", scripts), \
             mock.patch.object(self.backend, "RUNTIME_PYTHON", Path(sys.executable)), \
             mock.patch.dict(os.environ, {
                 "PYTHONPATH": str(injected), "PYTHONHOME": str(injected / "missing"),
             }):
            result = self.backend.HPSv3PPModel("model").score(
                [Image.new("RGB", (1, 1))], ["日本語 & $(echo example)"]
            )
        self.assertEqual(result, [1.0])
        self.assertFalse(marker.exists())

    def test_run_relays_worker_warnings_once_and_keeps_result_shape(self):
        self.make_model()
        runtime = Path(self.workspace.name) / "runtime.exe"
        runtime.touch()
        process = self.make_process()
        captured = {}
        def popen(command, **kwargs):
            captured["command"] = command
            Path(command[6]).write_text('{"result": [0.75], "warnings": ["dtype mismatch"]}', encoding="utf-8")
            return process
        with mock.patch.object(self.backend, "RUNTIME_PYTHON", runtime), \
             mock.patch.object(self.backend.subprocess, "Popen", side_effect=popen), \
             mock.patch.object(self.backend.logger, "warning") as warning:
            result = self.backend.HPSv3PPModel("model").score([Image.new("RGB", (1, 1))], ["p"])
        self.assertEqual(result, [0.75])
        warning.assert_called_once_with("HPSv3++: %s", "dtype mismatch")
        self.assertFalse(Path(captured["command"][5]).parent.exists())

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
        self.assertFalse(Path(captured["command"][5]).parent.exists())

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
        self.assertFalse(Path(captured["command"][5]).parent.exists())


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
