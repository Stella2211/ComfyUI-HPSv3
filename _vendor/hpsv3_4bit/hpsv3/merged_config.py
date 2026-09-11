"""Validate the published Qwen2-VL export and retain its text configuration."""
import json
from pathlib import Path

from safetensors import safe_open
from transformers import Qwen2VLConfig


def load_merged_config(directory, tokenizer):
    directory = Path(directory)
    raw = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    if raw.get("model_type") != "qwen2_vl":
        raise ValueError("Expected a Qwen2-VL architecture config")
    nested = raw.get("text_config")
    if nested is not None:
        for key in ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size"):
            if nested.get(key) != raw.get(key):
                raise ValueError(f"Conflicting Qwen2-VL architecture field: {key}")
    shapes = {}
    keys = {"model.embed_tokens.weight", "lm_head.weight"}
    for shard in directory.glob("*.safetensors"):
        with safe_open(str(shard), framework="pt", device="cpu") as weights:
            for key in keys.intersection(weights.keys()):
                if key in shapes:
                    raise ValueError(f"Duplicate weight: {key}")
                shapes[key] = weights.get_slice(key).get_shape()
    embedding = shapes.get("model.embed_tokens.weight")
    if (embedding is None or len(embedding) != 2 or embedding[1] != raw["hidden_size"]
            or shapes.get("lm_head.weight") != embedding):
        raise ValueError(f"Unsupported embedding/head shapes: {shapes}")
    if max(tokenizer.get_vocab().values()) >= embedding[0]:
        raise ValueError("Tokenizer exceeds checkpoint vocabulary size")
    if isinstance(raw.get("text_config"), dict):
        raw["text_config"]["vocab_size"] = embedding[0]
    else:
        raw["vocab_size"] = embedding[0]
    return Qwen2VLConfig.from_dict(raw)
