import json
import logging
import os
from pathlib import Path

import torch

import folder_paths
from comfy import model_management

logger = logging.getLogger(__name__)
DEFAULT_MODEL_NAME = "HPSv3-PlusPlus-bnb-NF4"
DEFAULT_MODEL_REPO = "stella221125/HPSv3-PlusPlus-bnb-NF4"
DOWNLOAD_STAGING_NAME = f".{DEFAULT_MODEL_NAME}.download"
DOWNLOAD_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE*", "NOTICE*", "README*"]
folder_paths.add_model_folder_path("hpsv3pp", os.path.join(folder_paths.models_dir, "hpsv3pp"))


def list_models():
    models = {str(config.parent.relative_to(root)) for directory in folder_paths.get_folder_paths("hpsv3pp")
              for root in [Path(directory)] for config in root.glob("*/config.json")
              if not config.parent.name.startswith(".")}
    models.add(DEFAULT_MODEL_NAME)
    return sorted(models)


def _model_root():
    paths = folder_paths.get_folder_paths("hpsv3pp")
    if not paths:
        raise RuntimeError("No models/hpsv3pp folder is configured for HPSv3++.")
    return Path(paths[0]).resolve()


def _validate_download(path):
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
        if any(not isinstance(json.loads((root / name).read_text(encoding="utf-8")), dict) for name in required):
            return False
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        if not isinstance(config, dict) or config.get("model_type") != "qwen3_vl":
            return False
        quant = config.get("quantization_config", {})
        if not isinstance(quant, dict) or quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit", quant.get("_load_in_4bit", False)):
            return False
        index = root / "model.safetensors.index.json"
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
    except (OSError, ValueError, TypeError):
        return False
    return True


def _download_default_model():
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import tqdm as hf_tqdm
    root = _model_root()
    target = root / DEFAULT_MODEL_NAME
    if target.exists() or target.is_symlink():
        raise ValueError("The default HPSv3++ model folder exists but is incomplete; repair or remove it before downloading.")
    staging = root / DOWNLOAD_STAGING_NAME
    if staging.is_symlink() or staging.resolve() != staging:
        raise RuntimeError("The HPSv3++ download staging path is a link; move it elsewhere before retrying.")
    root.mkdir(parents=True, exist_ok=True)
    model_management.throw_exception_if_processing_interrupted()
    logger.info("HPSv3++: downloading %s; partial files are kept for retry", DEFAULT_MODEL_REPO)
    cancellation = [None]

    class _CancellableTqdm(hf_tqdm):
        def update(self, n=1):
            if cancellation[0] is not None:
                raise cancellation[0]
            try:
                model_management.throw_exception_if_processing_interrupted()
            except BaseException as exc:
                if cancellation[0] is None:
                    cancellation[0] = exc
                raise
            return super().update(n)

    snapshot_download(repo_id=DEFAULT_MODEL_REPO, local_dir=str(staging), allow_patterns=DOWNLOAD_ALLOW_PATTERNS,
                      max_workers=1, tqdm_class=_CancellableTqdm)
    if cancellation[0] is not None:
        raise cancellation[0]
    model_management.throw_exception_if_processing_interrupted()
    if not _validate_download(staging):
        raise RuntimeError("HPSv3++ model download completed but the model is incomplete; retry to resume the download.")
    if target.exists() or target.is_symlink():
        raise RuntimeError("HPSv3++ model appeared while downloading; refusing to overwrite it.")
    os.replace(staging, target)
    logger.info("HPSv3++: model download complete: %s", target)
    return target


def resolve_model(name, allow_download=False):
    if Path(name).is_absolute() or ".." in Path(name).parts or DOWNLOAD_STAGING_NAME in Path(name).parts:
        raise ValueError("Select a model folder inside models/hpsv3pp.")
    for directory in folder_paths.get_folder_paths("hpsv3pp"):
        root = Path(directory).resolve()
        candidate = (root / name).resolve()
        if candidate == root or not candidate.is_relative_to(root):
            raise ValueError("Select a model folder inside models/hpsv3pp.")
        if (candidate / "config.json").is_file():
            try:
                config = json.loads((candidate / "config.json").read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise ValueError("The HPSv3++ model has an invalid config.json.") from exc
            if not isinstance(config, dict):
                raise ValueError("The HPSv3++ model has an invalid config.json.")
            if config.get("model_type") != "qwen3_vl":
                raise ValueError("Select a merged HPSv3++ Qwen3-VL bitsandbytes NF4 model.")
            quant = config.get("quantization_config", {})
            if not isinstance(quant, dict) or quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit", quant.get("_load_in_4bit", False)):
                raise ValueError("Select a merged HPSv3++ bitsandbytes NF4 model.")
            if not (candidate / "reward_config.json").is_file():
                raise ValueError("The HPSv3++ model is incomplete: reward_config.json is missing.")
            if not _validate_download(candidate):
                raise ValueError("The HPSv3++ model is incomplete or invalid.")
            return candidate
    if name == DEFAULT_MODEL_NAME and allow_download:
        downloaded = _download_default_model()
        if downloaded.is_dir() and _validate_download(downloaded):
            return downloaded
    raise FileNotFoundError("No HPSv3++ model found. Place the complete NF4 model in models/hpsv3pp and refresh the model list.")


def _run_inference(family, model_path, operation, images, **options):
    from .inference import run_inference
    return run_inference(family, model_path, operation, images,
                         check_cancel=model_management.throw_exception_if_processing_interrupted,
                         **options)


class HPSv3PPModel:
    def __init__(self, model_name):
        resolve_model(model_name, allow_download=True)
        self.model_name = model_name

    def score(self, images, prompts):
        return self._run("score", images, prompts=prompts)

    def caption(self, images, max_new_tokens):
        return self._run("caption", images, max_new_tokens=max_new_tokens)

    def _run(self, operation, images, **options):
        model_path = resolve_model(self.model_name)
        device = model_management.get_torch_device()
        if device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError("HPSv3++ NF4 currently requires an NVIDIA CUDA GPU.")
        model_management.throw_exception_if_processing_interrupted()
        return _run_inference("hpsv3pp", model_path, operation, images, **options)
