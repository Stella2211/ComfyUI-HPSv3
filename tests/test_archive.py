import importlib.util
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile


spec = importlib.util.spec_from_file_location("archive_check", Path(__file__).parents[1] / "scripts" / "check_archive.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)


class ArchiveTests(unittest.TestCase):
    def test_install_hook_is_required(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "node.zip"
            with ZipFile(path, "w"):
                pass
            with self.assertRaisesRegex(ValueError, "installation hook"):
                checker.check_archive(path)

    def test_external_source_cannot_leak_into_archive(self):
        for filename in ("_external/original.py", "qwen3vl_rm.py", "data_collator_qwen.py"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "node.zip"
                with ZipFile(path, "w") as archive:
                    archive.writestr("install.py", "")
                    archive.writestr(filename, "")
                with self.assertRaises(ValueError):
                    checker.check_archive(path)


if __name__ == "__main__":
    unittest.main()
