"""Load the architecture saved with an HPSv3++ checkpoint."""
import json
from pathlib import Path

from transformers import Qwen3VLConfig


def load_merged_config(merged_dir, tokenizer):
    directory = Path(merged_dir)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if (raw.get("model_type") != "qwen3_vl"
            or not isinstance(raw.get("text_config"), dict)
            or not raw["text_config"]
            or not isinstance(raw.get("vision_config"), dict)
            or not raw["vision_config"]):
        raise ValueError(
            "HPSv3++ requires a Qwen3-VL config.json with text_config and vision_config. "
            "Legacy training-only configs are not supported. Use the published NF4 "
            "model or create a complete checkpoint with hpsv3pp/scripts/merge_bf16.py."
        )
    return Qwen3VLConfig.from_dict(raw)
