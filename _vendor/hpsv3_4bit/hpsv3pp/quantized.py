"""Inference-only HPSv3++ NF4 runtime."""
from __future__ import annotations

import torch
from transformers import Qwen3VLForConditionalGeneration
from safetensors import safe_open

from ..quantized import QuantizedInferencer, load_processor, load_reward_model, validate_checkpoint
from .merged_config import load_merged_config
from .model import get_reward_model_class, install_vision_interpolation_hook
from .upstream import load_prompts

REWARD_SETTINGS = dict(output_dim=2, reward_token="special", rm_head_type="ranknet", rm_head_kwargs=None, cond_dim=256)
CAPABILITY_DTYPES = {"BF16": torch.bfloat16, "F32": torch.float32}


def _restore_capability_dtype(model, directory):
    # Capability prediction retains the checkpoint precision. Transformers can
    # otherwise preserve the upstream class's FP32 construction dtype.
    dtypes = set()
    for shard in directory.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as weights:
            for key in weights.keys():
                if key.startswith("cap_encoder."):
                    dtypes.add(weights.get_slice(key).get_dtype())
    if len(dtypes) != 1 or not dtypes <= CAPABILITY_DTYPES.keys():
        raise ValueError("Capability weights must have one supported floating dtype.")
    model.cap_encoder.to(dtype=CAPABILITY_DTYPES[dtypes.pop()])


class HPSv3PPQuantizedInferencer(QuantizedInferencer):
    generation_class = Qwen3VLForConditionalGeneration

    @classmethod
    def from_merged_dir(cls, merged_dir, device="cuda", check_cancel=None, processor_directory=None,
                        source_directory=None):
        check_cancel = check_cancel or (lambda: None)
        directory, processor = load_processor(merged_dir, processor_directory, check_cancel)
        settings = validate_checkpoint(directory, processor, "qwen3_vl", REWARD_SETTINGS)
        prompts = load_prompts(source_directory)
        config = load_merged_config(directory, processor.tokenizer)
        model = load_reward_model(get_reward_model_class(source_directory), directory, config, processor, device,
                                  check_cancel, **settings)
        install_vision_interpolation_hook(model)
        _restore_capability_dtype(model, directory)
        return cls(model, processor, str(device), prompts["INSTRUCTION"], prompts["prompt_with_special_token"])

    @torch.inference_mode()
    def reward(self, image_paths, prompts, iter_step=0.0):
        batch = self.prepare_batch(image_paths, prompts)
        iteration = torch.full((batch["input_ids"].shape[0],), float(iter_step), dtype=torch.float32, device=self.device)
        return self.model(return_dict=True, iter_values=iteration, **batch)["logits"]
