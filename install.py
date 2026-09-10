import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT / "third_party" / "hpsv3-4bit"
UPSTREAM_COMMIT = "f9878d0535205da70fe701e6a4df2abd268b88d7"


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
    env = os.environ.copy()
    env["UV_PROJECT_ENVIRONMENT"] = str(ROOT / ".venv")
    subprocess.run(command + ["sync", "--project", str(UPSTREAM / "hpsv3pp"), "--frozen", "--python", "3.12"], env=env, check=True)
    print("HPSv3++ runtime installed. Place the NF4 model in ComfyUI/models/hpsv3pp, then restart ComfyUI.")


if __name__ == "__main__":
    install()
