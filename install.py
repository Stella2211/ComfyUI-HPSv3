import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "third_party" / "hpsv3-4bit"
UPSTREAM_COMMIT = "650f86cc1830463b5bd8af073da27c04becb1fce"


def _run(command, **kwargs):
    """Use non-shell subprocesses and keep installer windows unobtrusive."""
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    return subprocess.run(command, check=True, shell=False, creationflags=flags, **kwargs)


def _verify_upstream_revision():
    result = _run(
        ["git", "rev-parse", "HEAD"],
        cwd=UPSTREAM,
        capture_output=True,
        text=True,
    )
    revision = result.stdout.strip()
    if revision != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"HPSv3 upstream revision mismatch: expected {UPSTREAM_COMMIT}, got {revision or '<empty>'}"
        )


def install():
    if (ROOT / ".git").exists():
        _run(["git", "submodule", "update", "--init", "--", "third_party/hpsv3-4bit"], cwd=ROOT)
    else:
        if not (UPSTREAM / ".git").exists():
            _run(["git", "clone", "--no-checkout", "https://github.com/Stella2211/hpsv3-4bit.git", str(UPSTREAM)])
        _run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=UPSTREAM)
    _verify_upstream_revision()
    _run(["git", "submodule", "update", "--init", "--recursive"], cwd=UPSTREAM)
    uv = shutil.which("uv")
    command = [uv] if uv else [sys.executable, "-m", "uv"]
    # The wrappers require incompatible Transformers versions.
    for project, environment in (("hpsv3pp", ".venv"), ("hpsv3", ".venv-hpsv3")):
        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / environment)
        _run(command + ["sync", "--project", str(UPSTREAM / project), "--frozen", "--python", "3.12"], env=env)
    print("HPSv3 and HPSv3++ runtimes installed. Restart ComfyUI; each Model Loader downloads its NF4 model when missing.")


if __name__ == "__main__":
    install()
