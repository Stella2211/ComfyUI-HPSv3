"""Check that the Registry ZIP contains the reviewed runtime and no development code."""

import ast
import hashlib
import json
from pathlib import PurePosixPath
import sys
from zipfile import ZipFile


def check_archive(path):
    with ZipFile(path) as archive:
        names = set(archive.namelist())
        if "install.py" not in names:
            raise ValueError("Archive is missing the Manager source installation hook.")
        forbidden = {"third_party", "scripts", "tests", ".github", ".comfy", ".venv", ".venv-hpsv3", "__pycache__", ".git", "artifacts", "_external"}
        for name in names:
            parts = PurePosixPath(name).parts
            if PurePosixPath(name).name in {"qwen3vl_rm.py", "data_collator_qwen.py"}:
                raise ValueError(f"External HPSv3++ source must not be bundled: {name}")
            if forbidden.intersection(parts) or PurePosixPath(name).suffix in (".safetensors", ".pth", ".pyc"):
                raise ValueError(f"Unexpected development/model file in archive: {name}")
        prefix = "_vendor/hpsv3_4bit/"
        manifest = json.loads(archive.read(prefix + "SOURCE.json"))
        for name, digest in manifest["files"].items():
            data = archive.read(prefix + name)
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Runtime source mismatch: {name}")
        expected = {prefix + name for name in manifest["files"]} | {prefix + "SOURCE.json"}
        actual = {name for name in names if name.startswith(prefix) and not name.endswith("/")}
        if actual != expected:
            raise ValueError("Archive contains files outside the runtime manifest.")
        for name in names:
            if not name.endswith(".py"):
                continue
            tree = ast.parse(archive.read(name), filename=name)
            if name.startswith(prefix + "hpsv3pp/"):
                for node in ast.walk(tree):
                    if isinstance(node, ast.ClassDef) and node.name.startswith("Qwen3VLRewardModel"):
                        raise ValueError(f"HPSv3++ reward classes must be inherited from external source: {name}")
                    if isinstance(node, ast.Assign) and any(
                        isinstance(target, ast.Name) and target.id in {"INSTRUCTION", "prompt_with_special_token", "prompt_without_special_token"}
                        for target in node.targets
                    ):
                        raise ValueError(f"HPSv3++ prompt constants must be read from external source: {name}")
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(alias.name in ("subprocess", "multiprocessing") for alias in node.names):
                    raise ValueError(f"Process module in runtime archive: {name}")
                if isinstance(node, ast.ImportFrom) and node.module in ("subprocess", "multiprocessing"):
                    raise ValueError(f"Process module in runtime archive: {name}")
                if isinstance(node, ast.Call):
                    func = node.func
                    if isinstance(func, ast.Name) and func.id in ("eval", "exec", "__import__"):
                        raise ValueError(f"Dynamic code execution in runtime archive: {name}")
                    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                        if (func.value.id, func.attr) in (("os", "system"), ("os", "popen"), ("torch", "load")):
                            raise ValueError(f"Unexpected command/pickle loader in runtime archive: {name}")
        print(f"Archive verified: {len(names)} files; runtime source {manifest['revision']}.")


if __name__ == "__main__":
    check_archive(sys.argv[1] if len(sys.argv) > 1 else "node.zip")
