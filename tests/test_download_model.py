import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).parents[1]


class DownloadHelperTests(unittest.TestCase):
    def load_helper(self):
        spec = importlib.util.spec_from_file_location("hpsv3_test_download_model", ROOT / "download_model.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module, spec.name

    def test_invokes_snapshot_download_without_pinning_revision(self):
        helper, name = self.load_helper()
        snapshot = mock.Mock()
        sys.modules["huggingface_hub"] = types.SimpleNamespace(snapshot_download=snapshot)
        try:
            with mock.patch.object(sys, "argv", ["download_model.py", "owner/repo", "C:/staging"]):
                helper.main()
            snapshot.assert_called_once_with("owner/repo", local_dir="C:/staging")
        finally:
            sys.modules.pop("huggingface_hub", None)
            sys.modules.pop(name, None)

    def test_rejects_wrong_argument_count(self):
        helper, name = self.load_helper()
        try:
            with mock.patch.object(sys, "argv", ["download_model.py"]):
                with self.assertRaises(SystemExit):
                    helper.main()
        finally:
            sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
