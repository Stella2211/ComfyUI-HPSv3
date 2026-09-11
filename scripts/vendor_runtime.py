"""Build/check the runtime snapshot; this development tool is excluded from node.zip."""

import argparse
import hashlib
import json
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "third_party" / "hpsv3-4bit" / "src" / "hpsv3_4bit"
DESTINATION = ROOT / "_vendor" / "hpsv3_4bit"
REPOSITORY = "https://github.com/Stella2211/hpsv3-4bit"
# Host integrations own model acquisition and do not expose standalone CLIs.
CLI_FILES = {"cli.py", "model_source.py"}


def snapshot(source, revision):
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Provide the full upstream Git commit as --revision.")
    source = Path(source).resolve()
    if not (source / "__init__.py").is_file():
        raise ValueError("Initialize the pinned hpsv3-4bit submodule before building.")
    files = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Runtime source must not contain symlinks: {path}")
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.relative_to(source).as_posix() in CLI_FILES:
            continue
        if path.suffix not in (".py", ".md", ".json", ".txt") and not path.name.startswith(("LICENSE", "NOTICE")):
            continue
        files[path.relative_to(source).as_posix()] = path.read_bytes().replace(b"\r\n", b"\n")
    if "THIRD_PARTY_NOTICES.md" not in files:
        raise ValueError("The runtime must include its third-party notices.")
    manifest = {
        "repository": REPOSITORY,
        "revision": revision,
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in files.items()},
    }
    files["SOURCE.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    return files


def materialize(files, destination, *, check=False):
    destination = Path(destination)
    if destination.is_symlink():
        raise ValueError("The runtime destination must not be a symlink.")
    destination = destination.resolve()
    existing = {}
    if destination.exists():
        for path in destination.rglob("*"):
            if path.is_symlink():
                raise ValueError(f"The runtime snapshot contains a symlink: {path}")
            if path.is_file() and "__pycache__" not in path.parts:
                existing[path.relative_to(destination).as_posix()] = path.read_bytes()
    differences = sorted(name for name in files.keys() | existing.keys() if files.get(name) != existing.get(name))
    if check:
        if differences:
            raise ValueError("Runtime snapshot differs from its pinned source: " + ", ".join(differences))
        return
    for name in existing.keys() - files.keys():
        (destination / name).unlink()
    for name, data in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, help="git -C third_party/hpsv3-4bit rev-parse HEAD")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    files = snapshot(SOURCE, args.revision)
    materialize(files, DESTINATION, check=args.check)
    print("Runtime snapshot verified." if args.check else "Runtime snapshot generated.")


if __name__ == "__main__":
    main()
