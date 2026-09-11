"""Strict local checkpoint helpers for the reusable runtime."""
import json
from pathlib import Path

def load_reward_settings(directory, defaults, tokenizer):
    path = Path(directory) / "reward_config.json"
    if not path.is_file():
        raise ValueError("reward_config.json is missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != 1:
        raise ValueError("Unsupported reward_config format version")
    settings = data.get("model_kwargs")
    if settings is None or set(settings) != set(defaults) or settings != defaults:
        raise ValueError("Reward model settings do not match this loader")
    token_id = tokenizer.convert_tokens_to_ids(data.get("reward_token"))
    if token_id != data.get("reward_token_id") or settings["special_token_ids"] != [token_id]:
        raise ValueError("Reward token ID does not match the saved tokenizer")
    return settings
