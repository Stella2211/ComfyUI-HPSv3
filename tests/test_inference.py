import importlib
from pathlib import Path
import sys
import types
import unittest
from unittest import mock
import weakref

from PIL import Image

ROOT = Path(__file__).parents[1]
PACKAGE = "hps_native_tests"


class Handle:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class Layer:
    def __init__(self):
        self.handle = Handle()

    def register_forward_pre_hook(self, callback):
        self.callback = callback
        return self.handle


class Model:
    def __init__(self):
        self.layer = Layer()

    def named_modules(self):
        return [("", self), ("transformer.layers.0", self.layer)]


class Session:
    def __init__(self):
        self.model = Model()
        self.scores = []
        self.captions = []

    def score(self, image, prompt):
        self.scores.append((image, prompt))
        if hasattr(self.model.layer, "callback"):
            self.model.layer.callback(self.model.layer, ())
        return len(self.scores) + 0.25

    def caption(self, image, max_new_tokens, stopping_criteria):
        self.captions.append((image, max_new_tokens, stopping_criteria))
        stopping_criteria[0](None, None)
        return " generated caption "


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(ROOT)]
        vendor = types.ModuleType(f"{PACKAGE}._vendor")
        runtime = types.ModuleType(f"{PACKAGE}._vendor.hpsv3_4bit")
        runtime.load_model = mock.Mock()
        sys.modules[PACKAGE] = package
        sys.modules[f"{PACKAGE}._vendor"] = vendor
        sys.modules[f"{PACKAGE}._vendor.hpsv3_4bit"] = runtime
        cls.inference = importlib.import_module(f"{PACKAGE}.inference")
        cls.runtime = runtime

    @classmethod
    def tearDownClass(cls):
        for name in tuple(sys.modules):
            if name == PACKAGE or name.startswith(PACKAGE + "."):
                sys.modules.pop(name)

    def setUp(self):
        self.session = Session()
        self.runtime.load_model.reset_mock(return_value=True)
        self.runtime.load_model.return_value = self.session
        self.comfy = types.ModuleType("comfy")
        self.comfy.model_management = types.SimpleNamespace(
            get_torch_device=lambda: "cpu",
            free_memory=mock.Mock(),
            soft_empty_cache=mock.Mock(),
        )
        self.transformers = types.ModuleType("transformers")
        self.transformers.StoppingCriteria = object
        self.transformers.StoppingCriteriaList = list
        self.modules = mock.patch.dict(sys.modules, {"comfy": self.comfy, "transformers": self.transformers})
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_score_uses_runtime_one_pair_at_a_time_and_cleans_hooks(self):
        images = [Image.new("RGB", (8, 8)), Image.new("RGBA", (8, 8))]
        result = self.inference.run_inference(
            "hpsv3pp", Path("model"), "score", images,
            prompts=["first", "second"], check_cancel=lambda: None,
        )
        self.assertEqual(result, [1.25, 2.25])
        self.assertEqual([prompt for _, prompt in self.session.scores], ["first", "second"])
        self.runtime.load_model.assert_called_once_with("hpsv3pp", Path("model"), "cpu", check_cancel=mock.ANY)
        self.assertTrue(self.session.model.layer.handle.removed)

    def test_caption_passes_stopping_criteria_and_max_tokens(self):
        result = self.inference.run_inference(
            "hpsv3", "model", "caption", [Image.new("RGB", (8, 8))],
            max_new_tokens=17, check_cancel=lambda: None,
        )
        self.assertEqual(result, ["generated caption"])
        self.assertEqual(self.session.captions[0][1], 17)
        self.assertIsInstance(self.session.captions[0][2], self.transformers.StoppingCriteriaList)
        self.assertEqual(len(self.session.captions[0][2]), 1)

    def test_validation_happens_before_model_load(self):
        with self.assertRaisesRegex(ValueError, "Provide one prompt"):
            self.inference.run_inference("hpsv3", "model", "score", [Image.new("RGB", (8, 8))], prompts=[], check_cancel=lambda: None)
        with self.assertRaises(TypeError):
            self.inference.run_inference("hpsv3", "model", "caption", ["path"], check_cancel=lambda: None)
        with self.assertRaises(ValueError):
            self.inference.run_inference("hpsv3", "model", "caption", [Image.new("RGB", (1, 1000))], check_cancel=lambda: None)
        self.runtime.load_model.assert_not_called()

    def test_forward_hook_and_generation_criterion_propagate_cancellation(self):
        checks = mock.Mock(side_effect=[None, None, None, KeyboardInterrupt])
        with self.assertRaises(KeyboardInterrupt):
            self.inference.run_inference("hpsv3", "model", "score", [Image.new("RGB", (8, 8))], prompts=["p"], check_cancel=checks)
        self.assertTrue(self.session.model.layer.handle.removed)
        self.session = Session()
        self.runtime.load_model.return_value = self.session
        criterion_check = mock.Mock(side_effect=[None, None, None, KeyboardInterrupt])
        with self.assertRaises(KeyboardInterrupt):
            self.inference.run_inference("hpsv3", "model", "caption", [Image.new("RGB", (8, 8))], check_cancel=criterion_check)
        self.assertTrue(self.session.model.layer.handle.removed)

    def test_session_is_released_after_runtime_error(self):
        refs = []
        def create_session(*args, **kwargs):
            session = Session()
            refs.append(weakref.ref(session))
            session.score = mock.Mock(side_effect=RuntimeError("boom"))
            return session
        self.runtime.load_model.side_effect = create_session
        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.inference.run_inference("hpsv3", "model", "score", [Image.new("RGB", (8, 8))], prompts=["p"], check_cancel=lambda: None)
        self.assertIsNone(refs[0]())
    def test_invalid_or_nonfinite_runtime_outputs_are_rejected(self):
        self.session.score = lambda image, prompt: float("nan")
        with self.assertRaisesRegex(ValueError, "non-finite"):
            self.inference.run_inference("hpsv3", "model", "score", [Image.new("RGB", (8, 8))], prompts=["p"], check_cancel=lambda: None)
        self.session.caption = lambda image, **kwargs: ""
        with self.assertRaisesRegex(ValueError, "empty caption"):
            self.inference.run_inference("hpsv3", "model", "caption", [Image.new("RGB", (8, 8))], check_cancel=lambda: None)
        self.assertFalse(self.inference._INFERENCE_LOCK.locked())

    def test_cancellation_while_waiting_does_not_load_model(self):
        acquired = self.inference._INFERENCE_LOCK.acquire()
        self.assertTrue(acquired)
        try:
            checks = mock.Mock(side_effect=KeyboardInterrupt)
            with self.assertRaises(KeyboardInterrupt):
                self.inference.run_inference("hpsv3", "model", "score", [], prompts=[], check_cancel=checks)
            self.runtime.load_model.assert_not_called()
        finally:
            self.inference._INFERENCE_LOCK.release()

    def test_failure_removes_hooks_and_releases_lock(self):
        self.session.score = mock.Mock(side_effect=RuntimeError("boom"))
        with self.assertRaisesRegex(RuntimeError, "boom"):
            self.inference.run_inference("hpsv3", "model", "score", [Image.new("RGB", (8, 8))], prompts=["p"], check_cancel=lambda: None)
        self.assertTrue(self.session.model.layer.handle.removed)
        self.assertFalse(self.inference._INFERENCE_LOCK.locked())


if __name__ == "__main__":
    unittest.main()
