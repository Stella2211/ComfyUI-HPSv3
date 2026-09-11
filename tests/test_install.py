"""The Manager hook must provision source and propagate installation failures."""

import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location("hps_install", Path(__file__).parents[1] / "install.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def test_hook_provisions_source_without_importing_comfy_nodes(self):
        provider = types.ModuleType("_vendor.hpsv3_4bit.hpsv3pp.upstream")
        provider.ensure_source = Mock(return_value=Path("verified-source"))
        with patch.dict(sys.modules, {provider.__name__: provider}), patch("builtins.print"):
            installer.main()
        provider.ensure_source.assert_called_once_with()

    def test_source_failure_is_reported_to_manager(self):
        provider = types.ModuleType("_vendor.hpsv3_4bit.hpsv3pp.upstream")
        provider.ensure_source = Mock(side_effect=RuntimeError("Source verification failed"))
        with patch.dict(sys.modules, {provider.__name__: provider}):
            with self.assertRaisesRegex(RuntimeError, "verification failed"):
                installer.main()


if __name__ == "__main__":
    unittest.main()
