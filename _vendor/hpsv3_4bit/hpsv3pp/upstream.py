"""Bounded access to the two reviewed HPSv3++ upstream source files.

Importing this module does not import or download upstream code. ``ensure_source``
is the explicit provisioning entry point used by installers and the CLI; the
load helpers only inspect already validated local bytes.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
import os
import shutil
import sys
import uuid
from pathlib import Path
from types import ModuleType
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

COMMIT = "6a095f68ee98330bf22365f872ed609bd44a216f"
_FILES = {
    "hpsv3/model/qwen3vl_rm.py": ("1919a0ce0d36d9a66d79ccb2ab467ae6d0c059a38040f96b42a38392bbd1b675", 256 * 1024),
    "hpsv3/dataset/data_collator_qwen.py": ("d7babc902fcbbfecdbf47687866bf6d737bad56b7ba283685c03220836c0c9f1", 256 * 1024),
}
_REPOSITORY = "https://api.github.com/repos/PlantPotatoOnMoon/HPSv3-PlusPlus/contents"
_TIMEOUT = 20.0
_REPAIR = "Reinstall the extension in ComfyUI-Manager and restart, or rerun the standalone CLI with network access."


class SourceProvisionError(RuntimeError):
    """The reviewed source is unavailable or failed validation."""


def _cache_base(source_directory: str | Path | None = None) -> Path:
    if source_directory is not None:
        return Path(source_directory).expanduser()
    return Path.home() / ".cache" / "hpsv3-4bit" / "upstream"


def _source_root(base: Path) -> Path:
    return base / COMMIT


def _has_link(path: Path) -> bool:
    return any(part.is_symlink() or part.is_junction() for part in (path, *path.parents))


def _safe_file(root: Path, relative: str) -> Path | None:
    root = root.resolve()
    path = root / relative
    try:
        resolved = path.resolve(strict=True)
    except OSError:
        return None
    if not resolved.is_relative_to(root) or not resolved.is_file() or path.is_symlink():
        return None
    current = path
    while current != root:
        if current.is_symlink() or current.is_junction():
            return None
        current = current.parent
    return resolved


def _valid_root(root: Path) -> bool:
    if not root.exists() or _has_link(root) or not root.is_dir():
        return False
    try:
        return all((path := _safe_file(root, relative)) is not None and _sha256(path, limit) == digest
                   for relative, (digest, limit) in _FILES.items())
    except OSError:
        return False


def _bounded_bytes(path: Path, limit: int) -> bytes | None:
    data = bytearray()
    with path.open("rb") as handle:
        while len(data) <= limit:
            block = handle.read(min(64 * 1024, limit + 1 - len(data)))
            if not block:
                return bytes(data)
            data.extend(block)
    return None


def _sha256(path: Path, limit: int) -> str:
    digest = hashlib.sha256()
    raw = _bounded_bytes(path, limit)
    if raw is None:
        return ""
    # Normalize after reading the bounded file so CRLF pairs split across
    # chunks cannot evade the fixed Git blob hash.
    digest.update(raw.replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def _local_source(source_directory: str | Path | None = None) -> Path | None:
    for candidate in (_source_root(_cache_base(source_directory)),):
        if _valid_root(candidate):
            return candidate
    return None


def _download(relative: str, destination: Path) -> None:
    digest, limit = _FILES[relative]
    url = f"{_REPOSITORY}/{relative}?ref={COMMIT}"
    try:
        request = Request(url, headers={
            "Accept": "application/vnd.github.raw+json",
            "User-Agent": "hpsv3-4bit-source-provisioner",
        })
        with urlopen(request, timeout=_TIMEOUT) as response, destination.open("wb") as handle:
            total = 0
            while True:
                block = response.read(64 * 1024)
                if not block:
                    break
                total += len(block)
                if total > limit:
                    raise SourceProvisionError(f"reviewed source file is too large: {relative}")
                handle.write(block)
    except SourceProvisionError:
        raise
    except HTTPError as exc:
        headers = exc.headers or {}
        retry_after = headers.get("Retry-After")
        reset = headers.get("X-RateLimit-Reset")
        remaining = headers.get("X-RateLimit-Remaining")
        is_rate_limit = exc.code == 429 or (exc.code == 403 and (retry_after or remaining == "0"))
        if is_rate_limit:
            hint = []
            if retry_after:
                hint.append(f"Retry-After={retry_after}")
            if reset:
                hint.append(f"X-RateLimit-Reset={reset}")
            suffix = f" ({', '.join(hint)})" if hint else ""
            raise SourceProvisionError(
                f"GitHub Contents API rate limit response for reviewed source ({relative}): "
                f"HTTP {exc.code}{suffix}. Wait for the limit to reset before retrying. {_REPAIR}"
            ) from exc
        if exc.code == 403:
            raise SourceProvisionError(
                f"GitHub Contents API access denied for reviewed source ({relative}): HTTP 403. {_REPAIR}"
            ) from exc
        raise SourceProvisionError(
            f"GitHub Contents API request failed for reviewed source ({relative}): HTTP {exc.code}"
        ) from exc
    except (OSError, URLError) as exc:
        raise SourceProvisionError(f"could not fetch reviewed HPSv3++ source ({relative}): {exc}") from exc
    if _sha256(destination, limit) != digest:
        raise SourceProvisionError(f"downloaded reviewed source failed hash validation: {relative}")


def ensure_source(local_files_only: bool = False, *, source_directory: str | Path | None = None) -> Path:
    """Return a validated source root, downloading only the pinned files if needed."""
    local = _local_source(source_directory)
    if local is not None:
        return local
    if local_files_only:
        raise SourceProvisionError(
            f"Validated HPSv3++ source is unavailable in offline mode. {_REPAIR}"
        )
    base = _cache_base(source_directory).expanduser()
    if _has_link(base):
        raise SourceProvisionError(f"source cache must not be a symlink: {base}")
    base = base.resolve()
    base.parent.mkdir(parents=True, exist_ok=True)
    base.mkdir(parents=True, exist_ok=True)
    target = _source_root(base)
    if _has_link(target):
        raise SourceProvisionError(f"source cache target must not be a symlink: {target}")
    staging = base / f".{COMMIT}.staging-{uuid.uuid4().hex}"
    try:
        staging.mkdir()
        for relative in _FILES:
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            _download(relative, path)
        if not _valid_root(staging):
            raise SourceProvisionError("downloaded reviewed source failed validation")
        if _valid_root(target):
            shutil.rmtree(staging)
            return target
        backup = base / f".{COMMIT}.invalid-{uuid.uuid4().hex}"
        moved_target = False
        try:
            if target.exists():
                os.replace(target, backup)
                moved_target = True
            os.replace(staging, target)
            if moved_target:
                if backup.is_dir():
                    shutil.rmtree(backup)
                else:
                    backup.unlink()
        except BaseException:
            if moved_target and not target.exists() and backup.exists():
                os.replace(backup, target)
            raise
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return target


def _require_local_source(source_directory: str | Path | None = None) -> Path:
    source = _local_source(source_directory)
    if source is None:
        raise SourceProvisionError(
            f"Validated HPSv3++ source is unavailable. {_REPAIR}"
        )
    return source


def _validated_file(source: Path, relative: str) -> tuple[Path, bytes]:
    path = _safe_file(source, relative)
    digest, limit = _FILES[relative]
    raw = _bounded_bytes(path, limit) if path is not None else None
    if raw is None or hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != digest:
        raise SourceProvisionError(f"reviewed HPSv3++ source changed or escaped its cache: {relative}")
    return path, raw


def load_model_module(source_directory: str | Path | None = None) -> ModuleType:
    """Load the reviewed reward model module from validated local bytes only."""
    source = _require_local_source(source_directory)
    path, _ = _validated_file(source, "hpsv3/model/qwen3vl_rm.py")
    path_key = hashlib.sha256(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    name = f"_hpsv3pp_reviewed_{COMMIT[:12]}_{path_key}"
    class _ReviewedSourceLoader(SourceFileLoader):
        def get_code(self, fullname):
            digest, limit = _FILES["hpsv3/model/qwen3vl_rm.py"]
            source = _bounded_bytes(Path(self.path), limit)
            if source is None or hashlib.sha256(source.replace(b"\r\n", b"\n")).hexdigest() != digest:
                raise SourceProvisionError("reviewed HPSv3++ model source changed while loading")
            return self.source_to_code(source, self.path)

    loader = _ReviewedSourceLoader(name, str(path))
    spec = importlib.util.spec_from_file_location(name, path, loader=loader)
    if spec is None or spec.loader is None:
        raise SourceProvisionError("could not construct a loader for reviewed HPSv3++ source")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module


def load_prompts(source_directory: str | Path | None = None) -> dict[str, str]:
    """Read prompt constants from the reviewed collator without importing it."""
    source = _require_local_source(source_directory)
    path, raw = _validated_file(source, "hpsv3/dataset/data_collator_qwen.py")
    tree = ast.parse(raw.decode("utf-8"), filename=str(path))
    values: dict[str, str] = {}
    expected = {"INSTRUCTION", "prompt_with_special_token", "prompt_without_special_token"}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in expected:
                    if not isinstance(node.value, ast.Constant) or not isinstance(node.value.value, str):
                        raise SourceProvisionError(f"reviewed prompt constant is not text: {target.id}")
                    value = node.value.value
                    values[target.id] = value
    if not {"INSTRUCTION", "prompt_with_special_token"}.issubset(values):
        raise SourceProvisionError("reviewed HPSv3++ prompt constants are missing")
    return values
