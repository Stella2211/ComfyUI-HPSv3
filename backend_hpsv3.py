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
RUNTIME_PYTHON = ROOT / ".venv-hpsv3" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
DEFAULT_MODEL_NAME = "HPSv3-bnb-NF4"
DEFAULT_MODEL_REPO = "stella221125/HPSv3-bnb-NF4"
DOWNLOAD_STAGING_NAME = f".{DEFAULT_MODEL_NAME}.download"
folder_paths.add_model_folder_path("hpsv3", os.path.join(folder_paths.models_dir, "hpsv3"))


def list_models():
    models = {
        str(config.parent.relative_to(root))
        for directory in folder_paths.get_folder_paths("hpsv3")
        for root in [Path(directory)]
        for config in root.glob("*/config.json")
        if not config.parent.name.startswith(".")
    }
    models.add(DEFAULT_MODEL_NAME)
    return sorted(models)


def _model_root():
    paths = folder_paths.get_folder_paths("hpsv3")
    if not paths:
        raise RuntimeError("No models/hpsv3 folder is configured for HPSv3.")
    return Path(paths[0]).resolve()


def _validate_download(path):
    """Validate the complete published model, including every indexed shard."""
    try:
        root = path.resolve()

        def has_file(name):
            file = (root / name).resolve()
            return file.is_relative_to(root) and file.is_file() and file.stat().st_size > 0

        required = ("config.json", "reward_config.json", "tokenizer_config.json", "preprocessor_config.json")
        if any(not has_file(name) for name in required):
            return False
        if not (has_file("tokenizer.json") or (has_file("vocab.json") and has_file("merges.txt"))):
            return False
        if any(not isinstance(json.loads((path / name).read_text(encoding="utf-8")), dict) for name in required):
            return False
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if config.get("model_type") != "qwen2_vl":
            return False
        quant = config.get("quantization_config", {})
        if not isinstance(quant, dict):
            return False
        if quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit", quant.get("_load_in_4bit", False)):
            return False
        index = path / "model.safetensors.index.json"
        if index.is_file():
            if not has_file(index.name):
                return False
            index_config = json.loads(index.read_text(encoding="utf-8"))
            if not isinstance(index_config, dict):
                return False
            weights = index_config.get("weight_map", {})
            if not isinstance(weights, dict) or not weights:
                return False
            if any(not isinstance(shard, str) or Path(shard).is_absolute() or ".." in Path(shard).parts or not has_file(shard) for shard in weights.values()):
                return False
        elif not has_file("model.safetensors"):
            return False
    except (OSError, ValueError):
        return False
    return True


def _download_default_model():
    root = _model_root()
    target = root / DEFAULT_MODEL_NAME
    if target.exists() or target.is_symlink():
        raise ValueError("The default HPSv3 model folder exists but is incomplete; repair or remove it before downloading.")
    staging = root / DOWNLOAD_STAGING_NAME
    if staging.is_symlink() or staging.resolve() != staging:
        raise RuntimeError("The HPSv3 download staging path is a link; move it elsewhere before retrying.")
    root.mkdir(parents=True, exist_ok=True)
    command = [str(RUNTIME_PYTHON), str(ROOT / "download_model.py"), DEFAULT_MODEL_REPO, str(staging)]
    if not RUNTIME_PYTHON.is_file():
        raise RuntimeError("HPSv3 runtime is missing. Run install.py with uv or use Manager's Fix function, then restart ComfyUI.")
    env = os.environ.copy()
    env.update(HF_HUB_DISABLE_TELEMETRY="1", DO_NOT_TRACK="1", PYTHONUTF8="1")
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    model_management.throw_exception_if_processing_interrupted()
    logger.info("HPSv3: downloading %s; partial files are kept for retry", DEFAULT_MODEL_REPO)
    # Inherit the console so Hugging Face progress is visible throughout the download.
    process = subprocess.Popen(command, env=env, creationflags=flags)
    try:
        while True:
            model_management.throw_exception_if_processing_interrupted()
            try:
                process.wait(timeout=1)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError("HPSv3 model download failed. Check the ComfyUI console for details, network access and free disk space; retry is safe and partial files were kept.")
        if not _validate_download(staging):
            raise RuntimeError("HPSv3 model download completed but the model is incomplete; retry to resume the download.")
        model_management.throw_exception_if_processing_interrupted()
        if target.exists() or target.is_symlink():
            raise RuntimeError("HPSv3 model appeared while downloading; refusing to overwrite it.")
        os.replace(staging, target)
        logger.info("HPSv3: model download complete: %s", target)
        return target
    finally:
        if process.poll() is None:
            process.kill()
        process.wait()


def resolve_model(name, allow_download=False):
    if Path(name).is_absolute() or ".." in Path(name).parts or DOWNLOAD_STAGING_NAME in Path(name).parts:
        raise ValueError("Select a model folder inside models/hpsv3.")
    for directory in folder_paths.get_folder_paths("hpsv3"):
        root = Path(directory).resolve()
        candidate = (root / name).resolve()
        if candidate == root or not candidate.is_relative_to(root):
            raise ValueError("Select a model folder inside models/hpsv3.")
        if (candidate / "config.json").is_file():
            try:
                config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("The HPSv3 model has an invalid config.json.") from exc
            if not isinstance(config, dict):
                raise ValueError("The HPSv3 model has an invalid config.json.")
            if config.get("model_type") != "qwen2_vl":
                raise ValueError("Select a merged HPSv3 Qwen2-VL bitsandbytes NF4 model.")
            quant = config.get("quantization_config", {})
            if not isinstance(quant, dict) or quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit", quant.get("_load_in_4bit", False)):
                raise ValueError("Select a merged HPSv3 bitsandbytes NF4 model.")
            if not _validate_download(candidate):
                raise ValueError("The HPSv3 model is incomplete or invalid; provide config, reward settings, processor and all weight shards.")
            return candidate
    if name == DEFAULT_MODEL_NAME and allow_download:
        downloaded = _download_default_model()
        if downloaded.is_dir() and _validate_download(downloaded):
            return downloaded
    raise FileNotFoundError("No HPSv3 model found. Place the complete NF4 model in models/hpsv3 and refresh the model list.")


class HPSv3Model:
    def __init__(self, model_name):
        resolve_model(model_name, allow_download=True)
        self.model_name = model_name

    def score(self, images, prompts):
        return self._run("score", images, prompts=prompts)

    def caption(self, images, max_new_tokens):
        return self._run("caption", images, max_new_tokens=max_new_tokens)

    def _run(self, operation, images, **options):
        model_path = resolve_model(self.model_name)
        if not RUNTIME_PYTHON.is_file():
            raise RuntimeError("HPSv3 runtime is missing. Run install.py with uv or use Manager's Fix function, then restart ComfyUI.")
        device = model_management.get_torch_device()
        if device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("HPSv3 NF4 currently requires an NVIDIA CUDA GPU.")
        model_management.throw_exception_if_processing_interrupted()
        model_management.free_memory(float("inf"), device)
        model_management.soft_empty_cache()
        with tempfile.TemporaryDirectory(prefix="comfy-hpsv3-") as directory:
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
            command = [str(RUNTIME_PYTHON), str(ROOT / "worker_hpsv3.py"), str(request_path), str(result_path)]
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
                    raise RuntimeError(f"HPSv3 inference failed:\n{detail}")
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                for warning in payload.get("warnings", []):
                    logger.warning("HPSv3: %s", warning)
                return payload["result"]
            finally:
                if process.poll() is None:
                    process.kill()
                process.communicate()
