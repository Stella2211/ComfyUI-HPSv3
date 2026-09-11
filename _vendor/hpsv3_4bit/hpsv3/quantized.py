"""Inference-only HPSv3 NF4 runtime."""
from __future__ import annotations

from dataclasses import dataclass
from types import MethodType
from typing import Sequence
import json
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from .hub import load_reward_settings
from .merged_config import load_merged_config
from .model import Qwen2VLRewardModelBT

SPECIAL_REWARD_TOKEN = "<|Reward|>"
OUTPUT_DIM = 2
RM_HEAD_TYPE = "ranknet"
RM_HEAD_KWARGS = None
REWARD_TOKEN_MODE = "special"
MAX_PIXELS = MIN_PIXELS = 256 * 28 * 28
CAPTION_INSTRUCTION = "Describe this image as a concise text-to-image prompt in one sentence."
KEY_MAPPING = {r"^visual": "model.visual", r"^model(?!\.(language_model|visual))": "model.language_model"}
INSTRUCTION = '''
You are tasked with evaluating a generated image based on Visual Quality and Text Alignment and give a overall score to estimate the human preference. Please provide a rating from 0 to 10, with 0 being the worst and 10 being the best.

**Visual Quality:**
Evaluate the overall visual quality of the image. The following sub-dimensions should be considered:
- **Reasonableness:** The image should not contain any significant biological or logical errors, such as abnormal body structures or nonsensical environmental setups.
- **Clarity:** Evaluate the sharpness and visibility of the image. The image should be clear and easy to interpret, with no blurring or indistinct areas.
- **Detail Richness:** Consider the level of detail in textures, materials, lighting, and other visual elements (e.g., hair, clothing, shadows).
- **Aesthetic and Creativity:** Assess the artistic aspects of the image, including the color scheme, composition, atmosphere, depth of field, and the overall creative appeal. The scene should convey a sense of harmony and balance.
- **Safety:** The image should not contain harmful or inappropriate content, such as political, violent, or adult material. If such content is present, the image quality and satisfaction score should be the lowest possible.

**Text Alignment:**
Assess how well the image matches the textual prompt across the following sub-dimensions:
- **Subject Relevance** Evaluate how accurately the subject(s) in the image (e.g., person, animal, object) align with the textual description. The subject should match the description in terms of number, appearance, and behavior.
- **Style Relevance:** If the prompt specifies a particular artistic or stylistic style, evaluate how well the image adheres to this style.
- **Contextual Consistency**: Assess whether the background, setting, and surrounding elements in the image logically fit the scenario described in the prompt. The environment should support and enhance the subject without contradictions.
- **Attribute Fidelity**: Check if specific attributes mentioned in the prompt (e.g., colors, clothing, accessories, expressions, actions) are faithfully represented in the image. Minor deviations may be acceptable, but critical attributes should be preserved.
- **Semantic Coherence**: Evaluate whether the overall meaning and intent of the prompt are captured in the image. The generated content should not introduce elements that conflict with or distort the original description.
Textual prompt - {text_prompt}


'''
PROMPT_WITH_SPECIAL_TOKEN = '''
Please provide the overall ratings of this image: <|Reward|>

END
'''


def _local_model_dir(directory):
    path = Path(directory).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {directory}")
    return path


def _load_processor(directory):
    return AutoProcessor.from_pretrained(str(directory), padding_side="right", local_files_only=True, trust_remote_code=False)


def _validate_checkpoint(directory, processor):
    raw = json.loads((directory / "reward_config.json").read_text(encoding="utf-8"))
    token_id = processor.tokenizer.convert_tokens_to_ids(SPECIAL_REWARD_TOKEN)
    defaults = dict(output_dim=OUTPUT_DIM, reward_token=REWARD_TOKEN_MODE,
                    special_token_ids=[token_id], rm_head_type=RM_HEAD_TYPE,
                    rm_head_kwargs=RM_HEAD_KWARGS)
    settings = load_reward_settings(directory, defaults, processor.tokenizer)
    if raw.get("reward_token") != SPECIAL_REWARD_TOKEN or raw.get("reward_token_id") != token_id:
        raise ValueError("Reward token ID does not match the saved tokenizer")
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    quant = config.get("quantization_config") or {}
    if config.get("model_type") != "qwen2_vl" or quant.get("quant_method") != "bitsandbytes" or quant.get("bnb_4bit_quant_type") != "nf4" or not quant.get("load_in_4bit", quant.get("_load_in_4bit", False)):
        raise ValueError("Model is not a serialized bitsandbytes NF4 checkpoint")
    return settings


def _patch_quantized_visual_dtype(model):
    visual = getattr(model, "visual", getattr(model.model, "visual", None))
    if visual is None:
        raise RuntimeError("Qwen2-VL visual tower is unavailable")
    visual.get_dtype = MethodType(lambda module: module.patch_embed.proj.weight.dtype, visual)


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
class HPSv3QuantizedInferencer:
    model: Qwen2VLRewardModelBT
    processor: object
    device: str = "cuda"

    @classmethod
    def from_merged_dir(cls, merged_dir, device="cuda", check_cancel=None, processor_directory=None):
        check_cancel = check_cancel or (lambda: None)
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
        settings = _validate_checkpoint(directory, processor)
        config = load_merged_config(directory, processor.tokenizer)
        # Transformers 5 nests text settings; the reward pooling code also
        # needs the tokenizer's padding ID on the outer config for batches.
        config.pad_token_id = processor.tokenizer.pad_token_id
        config.use_cache = False
        model, info = Qwen2VLRewardModelBT.from_pretrained(
            str(directory), config=config, **settings,
            torch_dtype=torch.bfloat16, attn_implementation="sdpa", quantization_config=None,
            device_map={"": str(device)}, use_safetensors=True, output_loading_info=True,
            local_files_only=True, trust_remote_code=False, key_mapping=KEY_MAPPING)
        check_cancel()
        if info.get("missing_keys") or info.get("mismatched_keys") or info.get("error_msgs") or info.get("unexpected_keys"):
            raise ValueError(f"Backbone checkpoint mismatch: {info}")
        model.rm_head.float()
        _patch_quantized_visual_dtype(model)
        model.eval()
        return cls(model, processor, str(device))

    def prepare_batch(self, image_paths: Sequence, prompts: Sequence[str]):
        if not image_paths or len(image_paths) != len(prompts):
            raise ValueError("Provide a nonempty batch with one prompt per image.")
        batch = _batch(self.processor, list(image_paths),
                       [INSTRUCTION.format(text_prompt=prompt) + PROMPT_WITH_SPECIAL_TOKEN for prompt in prompts], self.device)
        token_id = self.processor.tokenizer.convert_tokens_to_ids(SPECIAL_REWARD_TOKEN)
        if not torch.all((batch["input_ids"] == token_id).sum(dim=1) == 1).item():
            raise ValueError("Each scoring request must contain exactly one reward token.")
        return batch

    @torch.inference_mode()
    def reward(self, image_paths, prompts):
        batch = self.prepare_batch(image_paths, prompts)
        return self.model(return_dict=True, **batch)["logits"]

    def score(self, image_paths, prompts):
        rewards = self.reward(image_paths, prompts)
        if rewards.shape[0] != len(image_paths):
            raise ValueError("Reward result count does not match the image count")
        return rewards[:, 0].tolist()

    @torch.inference_mode()
    def caption(self, image_paths, max_new_tokens=96, instruction=CAPTION_INSTRUCTION, stopping_criteria=None):
        had = "forward" in vars(self.model)
        previous = vars(self.model).get("forward")
        self.model.forward = MethodType(Qwen2VLForConditionalGeneration.forward, self.model)
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
