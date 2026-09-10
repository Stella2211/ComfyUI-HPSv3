import json
import logging
import os
from pathlib import Path
import subprocess
import tempfile

import torch

import folder_paths
from comfy import model_management


logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parent
RUNTIME_PYTHON = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
folder_paths.add_model_folder_path("hpsv3pp", os.path.join(folder_paths.models_dir, "hpsv3pp"))


def list_models():
    return sorted({
        str(config.parent.relative_to(root))
        for directory in folder_paths.get_folder_paths("hpsv3pp")
        for root in [Path(directory)]
        for config in root.glob("*/config.json")
    })


def resolve_model(name):
    if Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Select a model folder inside models/hpsv3pp.")
    for directory in folder_paths.get_folder_paths("hpsv3pp"):
        root = Path(directory).resolve()
        candidate = (root / name).resolve()
        if candidate == root or not candidate.is_relative_to(root):
            raise ValueError("Select a model folder inside models/hpsv3pp.")
        if (candidate / "config.json").is_file():
            config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
            quant = config.get("quantization_config", {})
            if quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit", quant.get("_load_in_4bit", False)):
                raise ValueError("Select a merged HPSv3++ bitsandbytes NF4 model.")
            if not (candidate / "reward_config.json").is_file():
                raise ValueError("The HPSv3++ model is incomplete: reward_config.json is missing.")
            return candidate
    raise FileNotFoundError("No HPSv3++ model found. Place the complete NF4 model in models/hpsv3pp and refresh the model list.")


class HPSv3PPModel:
    def __init__(self, model_name):
        resolve_model(model_name)
        self.model_name = model_name

    def score(self, images, prompts):
        return self._run("score", images, prompts=prompts)

    def caption(self, images, max_new_tokens):
        return self._run("caption", images, max_new_tokens=max_new_tokens)

    def _run(self, operation, images, **options):
        model_path = resolve_model(self.model_name)
        if not RUNTIME_PYTHON.is_file():
            raise RuntimeError("HPSv3++ runtime is missing. Run install.py with uv or use Manager's Fix function, then restart ComfyUI.")
        device = model_management.get_torch_device()
        if device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("HPSv3++ NF4 currently requires an NVIDIA CUDA GPU.")
        model_management.throw_exception_if_processing_interrupted()
        model_management.free_memory(float("inf"), device)
        model_management.soft_empty_cache()
        with tempfile.TemporaryDirectory(prefix="comfy-hpsv3pp-") as directory:
            temp = Path(directory)
            image_paths = []
            for i, image in enumerate(images):
                path = temp / f"{i}.png"
                image.save(path)
                image_paths.append(str(path))
            request = dict(operation=operation, model=str(model_path), device=str(device), images=image_paths, **options)
            request_path = temp / "request.json"
            result_path = temp / "result.json"
            request_path.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
            env = os.environ.copy()
            env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", PYTHONUTF8="1")
            command = [str(RUNTIME_PYTHON), str(ROOT / "worker.py"), str(request_path), str(result_path)]
            flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, creationflags=flags)
            try:
                while True:
                    model_management.throw_exception_if_processing_interrupted()
                    try:
                        output, _ = process.communicate(timeout=1)
                        break
                    except subprocess.TimeoutExpired:
                        continue
                if process.returncode:
                    detail = output.decode("utf-8", errors="replace")[-4000:]
                    raise RuntimeError(f"HPSv3++ inference failed:\n{detail}")
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                for warning in payload.get("warnings", []):
                    logger.warning("HPSv3++: %s", warning)
                return payload["result"]
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()
