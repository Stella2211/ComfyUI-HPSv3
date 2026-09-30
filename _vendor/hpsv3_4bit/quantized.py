"""NF4 checkpoint loading and inference shared by both model families."""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Sequence
import json
from pathlib import Path

import torch
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info

SPECIAL_REWARD_TOKEN = "<|Reward|>"
MAX_PIXELS = MIN_PIXELS = 256 * 28 * 28
CAPTION_INSTRUCTION = "Describe this image as a concise text-to-image prompt in one sentence."


def _local_model_dir(directory):
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {directory}")
    return path


def _load_processor(directory):
    return AutoProcessor.from_pretrained(str(directory), padding_side="right", local_files_only=True, trust_remote_code=False)


def load_processor(merged_dir, processor_directory, check_cancel):
    directory = _local_model_dir(merged_dir)
    check_cancel()
    processor_path = directory if processor_directory is None else _local_model_dir(processor_directory)
    processor = _load_processor(processor_path)
    # Base Qwen processors do not contain the reward token. Recreate the
    # original CLI override behavior, then validate its ID against the
    # checkpoint before loading any model weights.
    if processor_path != directory and SPECIAL_REWARD_TOKEN not in processor.tokenizer.get_vocab():
        processor.tokenizer.add_special_tokens({"additional_special_tokens": [SPECIAL_REWARD_TOKEN]})
    check_cancel()
    return directory, processor


def validate_checkpoint(directory, processor, model_type, settings):
    """Return saved reward settings that exactly match ``settings`` plus the reward token."""
    path = directory / "reward_config.json"
    if not path.is_file():
        raise ValueError("reward_config.json is missing")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("format_version") != 1:
        raise ValueError("Unsupported reward_config format version")
    token_id = processor.tokenizer.convert_tokens_to_ids(SPECIAL_REWARD_TOKEN)
    expected = {**settings, "special_token_ids": [token_id]}
    if data.get("model_kwargs") != expected:
        raise ValueError("Reward model settings do not match this loader")
    if data.get("reward_token") != SPECIAL_REWARD_TOKEN or data.get("reward_token_id") != token_id:
        raise ValueError("Reward token ID does not match the saved tokenizer")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or {}
    if config.get("model_type") != model_type or quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit", quant.get("_load_in_4bit", False)):
        raise ValueError("Model is not a serialized bitsandbytes NF4 checkpoint")
    return expected


def load_reward_model(model_class, directory, config, processor, device, check_cancel, **options):
    # Transformers 5 nests text settings; the reward pooling code also
    # needs the tokenizer's padding ID on the outer config for batches.
    config.pad_token_id = processor.tokenizer.pad_token_id
    model, info = model_class.from_pretrained(
        str(directory), config=config, **options,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=None,
        device_map={"": str(device)}, use_safetensors=True, output_loading_info=True,
        local_files_only=True, trust_remote_code=False)
    check_cancel()
    if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs") or info.get("unexpected_keys"):
        raise ValueError(f"Backbone checkpoint mismatch: {info}")
    model.eval()
    return model


def _batch(processor, image, text, device):
    images, texts = (image, text) if isinstance(image, (list, tuple)) else ([image], [text])
    messages = [[{"role": "user", "content": [{"type": "image", "image": item,
        "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS}, {"type": "text", "text": prompt}]}]
        for item, prompt in zip(images, texts)]
    image_inputs, _ = process_vision_info(messages)
    batch = processor(text=[processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
                            for message in messages],
                      images=image_inputs, padding=True, return_tensors="pt")
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


@dataclass
class QuantizedInferencer:
    """Scoring and captioning shared by the family inferencers.

    Subclasses set ``generation_class`` and implement ``from_merged_dir`` and
    ``reward``.
    """

    model: object
    processor: object
    device: str
    instruction: str
    reward_prompt: str

    def prepare_batch(self, image_paths: Sequence, prompts: Sequence[str]):
        if not image_paths or len(image_paths) != len(prompts):
            raise ValueError("Provide a nonempty batch with one prompt per image.")
        batch = _batch(self.processor, list(image_paths),
                       [self.instruction.format(text_prompt=prompt) + self.reward_prompt for prompt in prompts], self.device)
        token_id = self.processor.tokenizer.convert_tokens_to_ids(SPECIAL_REWARD_TOKEN)
        if not torch.all((batch["input_ids"] == token_id).sum(dim=1) == 1).item():
            raise ValueError("Each scoring request must contain exactly one reward token.")
        return batch

    def score(self, image_paths, prompts, **options):
        rewards = self.reward(image_paths, prompts, **options)
        if rewards.shape[0] != len(image_paths):
            raise ValueError("Reward result count does not match the image count")
        return rewards[:, 0].tolist()

    @torch.inference_mode()
    def caption(self, image_paths, max_new_tokens=96, instruction=CAPTION_INSTRUCTION, stopping_criteria=None):
        had = "forward" in vars(self.model)
        previous = vars(self.model).get("forward")
        self.model.forward = MethodType(self.generation_class.forward, self.model)
        try:
            results = []
            for image in image_paths:
                batch = _batch(self.processor, image, instruction, self.device)
                options = dict(**batch, max_new_tokens=max_new_tokens, do_sample=False, use_cache=True,
                               pad_token_id=self.processor.tokenizer.pad_token_id)
                if stopping_criteria is not None:
                    options["stopping_criteria"] = stopping_criteria
                output = self.model.generate(**options)
                results.append(self.processor.tokenizer.decode(output[0, batch["input_ids"].shape[1]:], skip_special_tokens=True).strip())
            return results
        finally:
            if had:
                self.model.forward = previous
            else:
                del self.model.forward
