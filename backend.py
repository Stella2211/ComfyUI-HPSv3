import json
import logging
import os
from pathlib import Path

import torch

import folder_paths
from comfy import model_management

from .inference import run_inference

logger = logging.getLogger(__name__)
DOWNLOAD_ALLOW_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.jinja", "LICENSE*", "NOTICE*", "README*"]


class HPSv3Model:
    family = "hpsv3"
    label = "HPSv3"
    model_type = "qwen2_vl"
    architecture = "Qwen2-VL"
    default_name = "HPSv3-bnb-NF4"
    repo = "stella221125/HPSv3-bnb-NF4"

    def __init__(self, model_name):
        self.resolve(model_name, allow_download=True)
        self.model_name = model_name

    @classmethod
    def list_models(cls):
        models = {str(config.parent.relative_to(root)) for directory in folder_paths.get_folder_paths(cls.family)
                  for root in [Path(directory)] for config in root.glob("*/config.json")
                  if not config.parent.name.startswith(".")}
        models.add(cls.default_name)
        return sorted(models)

    @classmethod
    def staging_name(cls):
        return f".{cls.default_name}.download"

    @classmethod
    def _model_root(cls):
        paths = folder_paths.get_folder_paths(cls.family)
        if not paths:
            raise RuntimeError(f"No models/{cls.family} folder is configured for {cls.label}.")
        return Path(paths[0]).resolve()

    @classmethod
    def _validate(cls, path):
        root = path.resolve()
        try:
            config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"The {cls.label} model has an invalid config.json.") from exc
        if not isinstance(config, dict):
            raise ValueError(f"The {cls.label} model has an invalid config.json.")
        if config.get("model_type") != cls.model_type:
            raise ValueError(f"Select a merged {cls.label} {cls.architecture} bitsandbytes NF4 model.")
        quant = config.get("quantization_config", {})
        if not isinstance(quant, dict) or quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit", quant.get("_load_in_4bit", False)):
            raise ValueError(f"Select a merged {cls.label} bitsandbytes NF4 model.")

        def has_file(name):
            file = (root / name).resolve()
            return file.is_relative_to(root) and file.is_file() and file.stat().st_size > 0

        def is_complete():
            required = ("reward_config.json", "tokenizer_config.json", "preprocessor_config.json")
            if any(not has_file(name) for name in required):
                return False
            if not (has_file("tokenizer.json") or (has_file("vocab.json") and has_file("merges.txt"))):
                return False
            if any(not isinstance(json.loads((root / name).read_text(encoding="utf-8")), dict) for name in required):
                return False
            index = root / "model.safetensors.index.json"
            if not index.is_file():
                return has_file("model.safetensors")
            index_config = json.loads(index.read_text(encoding="utf-8"))
            weights = index_config.get("weight_map") if isinstance(index_config, dict) else None
            if not isinstance(weights, dict) or not weights:
                return False
            return all(isinstance(shard, str) and not Path(shard).is_absolute() and ".." not in Path(shard).parts and has_file(shard) for shard in weights.values())

        try:
            complete = is_complete()
        except (OSError, ValueError):
            complete = False
        if not complete:
            raise ValueError(f"The {cls.label} model is incomplete or invalid.")

    @classmethod
    def _download_default_model(cls):
        from huggingface_hub import snapshot_download
        from huggingface_hub.utils import tqdm as hf_tqdm
        root = cls._model_root()
        target = root / cls.default_name
        if target.exists() or target.is_symlink():
            raise ValueError(f"The default {cls.label} model folder exists but is incomplete; repair or remove it before downloading.")
        staging = root / cls.staging_name()
        if staging.is_symlink() or staging.resolve() != staging:
            raise RuntimeError(f"The {cls.label} download staging path is a link; move it elsewhere before retrying.")
        root.mkdir(parents=True, exist_ok=True)
        model_management.throw_exception_if_processing_interrupted()
        logger.info("%s: downloading %s; partial files are kept for retry", cls.label, cls.repo)
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

        snapshot_download(repo_id=cls.repo, local_dir=str(staging), allow_patterns=DOWNLOAD_ALLOW_PATTERNS,
                          max_workers=1, tqdm_class=_CancellableTqdm)
        if cancellation[0] is not None:
            raise cancellation[0]
        model_management.throw_exception_if_processing_interrupted()
        try:
            cls._validate(staging)
        except ValueError as exc:
            raise RuntimeError(f"{cls.label} model download completed but the model is incomplete; retry to resume the download.") from exc
        if target.exists() or target.is_symlink():
            raise RuntimeError(f"{cls.label} model appeared while downloading; refusing to overwrite it.")
        os.replace(staging, target)
        logger.info("%s: model download complete: %s", cls.label, target)
        return target

    @classmethod
    def resolve(cls, name, allow_download=False):
        if Path(name).is_absolute() or ".." in Path(name).parts or cls.staging_name() in Path(name).parts:
            raise ValueError(f"Select a model folder inside models/{cls.family}.")
        for directory in folder_paths.get_folder_paths(cls.family):
            root = Path(directory).resolve()
            candidate = (root / name).resolve()
            if candidate == root or not candidate.is_relative_to(root):
                raise ValueError(f"Select a model folder inside models/{cls.family}.")
            if (candidate / "config.json").is_file():
                cls._validate(candidate)
                return candidate
        if name == cls.default_name and allow_download:
            return cls._download_default_model()
        raise FileNotFoundError(f"No {cls.label} model found. Place the complete NF4 model in models/{cls.family} and refresh the model list.")

    def score(self, images, prompts):
        return self._run("score", images, prompts=prompts)

    def caption(self, images, max_new_tokens):
        return self._run("caption", images, max_new_tokens=max_new_tokens)

    def _run(self, operation, images, prompts=(), max_new_tokens=96):
        model_path = self.resolve(self.model_name)
        device = model_management.get_torch_device()
        if device.type != "cuda" or torch.version.hip is not None:
            raise RuntimeError(f"{self.label} NF4 currently requires an NVIDIA CUDA GPU.")
        return run_inference(self.family, model_path, operation, images,
                             check_cancel=model_management.throw_exception_if_processing_interrupted,
                             prompts=prompts, max_new_tokens=max_new_tokens)


class HPSv3PPModel(HPSv3Model):
    family = "hpsv3pp"
    label = "HPSv3++"
    model_type = "qwen3_vl"
    architecture = "Qwen3-VL"
    default_name = "HPSv3-PlusPlus-bnb-NF4"
    repo = "stella221125/HPSv3-PlusPlus-bnb-NF4"


for model_class in (HPSv3Model, HPSv3PPModel):
    folder_paths.add_model_folder_path(model_class.family, os.path.join(folder_paths.models_dir, model_class.family))
