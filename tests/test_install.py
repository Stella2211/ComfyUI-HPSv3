import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).parents[1]


class InstallTests(unittest.TestCase):
    def test_git_and_registry_install_use_separate_frozen_environments(self):
        spec = importlib.util.spec_from_file_location("hps_install", ROOT / "install.py")
        installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(installer)
        for git_checkout in (True, False):
            with self.subTest(git_checkout=git_checkout), \
                 mock.patch.object(Path, "exists", return_value=git_checkout), \
                 mock.patch.object(installer.shutil, "which", return_value="uv"), \
                 mock.patch.object(installer.subprocess, "run") as run:
                installer.install()
                syncs = [call for call in run.call_args_list if call.args[0][:2] == ["uv", "sync"]]
                self.assertEqual(len(syncs), 2)
                for call, project, environment in zip(syncs, ("hpsv3pp", "hpsv3"), (".venv", ".venv-hpsv3")):
                    self.assertEqual(call.args[0], ["uv", "sync", "--project", str(installer.UPSTREAM / project), "--frozen", "--python", "3.12"])
                    self.assertEqual(call.kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(ROOT / environment))
                    self.assertTrue(call.kwargs["check"])
                if not git_checkout:
                    self.assertIn(mock.call(["git", "checkout", "--detach", installer.UPSTREAM_COMMIT], cwd=installer.UPSTREAM, check=True), run.call_args_list)


if __name__ == "__main__":
    unittest.main()
