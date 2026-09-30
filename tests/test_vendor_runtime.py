import importlib.util
from pathlib import Path
import tempfile
import unittest


spec = importlib.util.spec_from_file_location("vendor_runtime", Path(__file__).parents[1] / "scripts" / "vendor_runtime.py")
vendor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vendor)


class VendorRuntimeTests(unittest.TestCase):
    def test_snapshot_requires_revision_and_notices_and_detects_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "source"
            source.mkdir()
            (source / "__init__.py").write_text("value = 1\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "commit"):
                vendor.snapshot(source, "main")
            with self.assertRaisesRegex(ValueError, "notices"):
                vendor.snapshot(source, "a" * 40)
            (source / "THIRD_PARTY_NOTICES.md").write_text("Provenance\n", encoding="utf-8")
            (source / "weights.bin").write_bytes(b"excluded")
            (source / "cli.py").write_text("cli_only = True\n", encoding="utf-8")
            (source / "model_source.py").write_text("hub_only = True\n", encoding="utf-8")
            files = vendor.snapshot(source, "a" * 40)
            self.assertNotIn("weights.bin", files)
            self.assertNotIn("cli.py", files)
            self.assertNotIn("model_source.py", files)
            destination = Path(tmp) / "snapshot"
            vendor.materialize(files, destination)
            vendor.materialize(files, destination, check=True)
            (destination / "__init__.py").write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs"):
                vendor.materialize(files, destination, check=True)
            vendor.materialize(files, destination)
            (destination / "obsolete.py").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "obsolete"):
                vendor.materialize(files, destination, check=True)
            vendor.materialize(files, destination)
            self.assertFalse((destination / "obsolete.py").exists())


if __name__ == "__main__":
    unittest.main()
