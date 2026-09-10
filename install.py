import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "third_party" / "hpsv3-4bit"
UPSTREAM_COMMIT = "650f86cc1830463b5bd8af073da27c04becb1fce"


def install():
    if (ROOT / ".git").exists():
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=ROOT, check=True)
    else:
        if not (UPSTREAM / ".git").exists():
            subprocess.run(["git", "clone", "--no-checkout", "https://github.com/Stella2211/hpsv3-4bit.git", str(UPSTREAM)], check=True)
        subprocess.run(["git", "checkout", "--detach", UPSTREAM_COMMIT], cwd=UPSTREAM, check=True)
        subprocess.run(["git", "submodule", "update", "--init", "--recursive"], cwd=UPSTREAM, check=True)
    uv = shutil.which("uv")
    command = [uv] if uv else [sys.executable, "-m", "uv"]
    # The wrappers require incompatible Transformers versions.
    for project, environment in (("hpsv3pp", ".venv"), ("hpsv3", ".venv-hpsv3")):
        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / environment)
        subprocess.run(command + ["sync", "--project", str(UPSTREAM / project), "--frozen", "--python", "3.12"], env=env, check=True)
    print("HPSv3 and HPSv3++ runtimes installed. Restart ComfyUI; each Model Loader downloads its NF4 model when missing.")


if __name__ == "__main__":
    install()
