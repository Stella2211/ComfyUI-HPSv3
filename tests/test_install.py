import importlib.util
from pathlib import Path
import unittest
from unittest import mock


ROOT = Path(__file__).parents[1]


def load_installer():
    spec = importlib.util.spec_from_file_location("hps_install", ROOT / "install.py")
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    return installer


class InstallTests(unittest.TestCase):
    def test_git_and_registry_install_use_separate_frozen_environments(self):
        installer = load_installer()
        for git_checkout in (True, False):
            with self.subTest(git_checkout=git_checkout):
                revision = mock.Mock(stdout=installer.UPSTREAM_COMMIT + "\n")
                with mock.patch.object(Path, "exists", return_value=git_checkout), \
                     mock.patch.object(installer.shutil, "which", return_value="uv"), \
                     mock.patch.object(installer.subprocess, "run", return_value=revision) as run:
                    installer.install()
                syncs = [call for call in run.call_args_list if call.args[0][:2] == ["uv", "sync"]]
                self.assertEqual(len(syncs), 2)
                for call, project, environment in zip(syncs, ("hpsv3pp", "hpsv3"), (".venv", ".venv-hpsv3")):
                    self.assertEqual(call.args[0], ["uv", "sync", "--project", str(installer.UPSTREAM / project), "--frozen", "--python", "3.12"])
                    self.assertEqual(call.kwargs["env"]["UV_PROJECT_ENVIRONMENT"], str(ROOT / environment))
                    self.assertTrue(call.kwargs["check"])
                    self.assertFalse(call.kwargs["shell"])
                self.assertTrue(any(call.args[0] == ["git", "rev-parse", "HEAD"] for call in run.call_args_list))
                if not git_checkout:
                    self.assertTrue(any(
                        call.args[0] == ["git", "checkout", "--detach", installer.UPSTREAM_COMMIT]
                        and call.kwargs["cwd"] == installer.UPSTREAM
                        for call in run.call_args_list
                    ))

    def test_revision_mismatch_prevents_runtime_setup(self):
        installer = load_installer()
        revision = mock.Mock(stdout="unexpected\n")
        for git_checkout in (True, False):
            with self.subTest(git_checkout=git_checkout), \
                 mock.patch.object(Path, "exists", return_value=git_checkout), \
                 mock.patch.object(installer.subprocess, "run", return_value=revision) as run:
                with self.assertRaisesRegex(RuntimeError, "revision mismatch"):
                    installer.install()
            self.assertTrue(all(call.args[0][0] == "git" for call in run.call_args_list))
            self.assertFalse(any("--recursive" in call.args[0] for call in run.call_args_list))


if __name__ == "__main__":
    unittest.main()
