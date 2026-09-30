"""Inference-only HPSv3 NF4 runtime."""
from __future__ import annotations

from types import MethodType

import torch
from transformers import Qwen2VLForConditionalGeneration

from ..quantized import QuantizedInferencer, load_processor, load_reward_model, validate_checkpoint
from .merged_config import load_merged_config
from .model import Qwen2VLRewardModelBT

REWARD_SETTINGS = dict(output_dim=2, reward_token="special", rm_head_type="ranknet", rm_head_kwargs=None)
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


def _patch_quantized_visual_dtype(model):
    visual = getattr(model, "visual", getattr(model.model, "visual", None))
    if visual is None:
        raise RuntimeError("Qwen2-VL visual tower is unavailable")
    visual.get_dtype = MethodType(lambda module: module.patch_embed.proj.weight.dtype, visual)


class HPSv3QuantizedInferencer(QuantizedInferencer):
    generation_class = Qwen2VLForConditionalGeneration

    @classmethod
    def from_merged_dir(cls, merged_dir, device="cuda", check_cancel=None, processor_directory=None):
        check_cancel = check_cancel or (lambda: None)
        directory, processor = load_processor(merged_dir, processor_directory, check_cancel)
        settings = validate_checkpoint(directory, processor, "qwen2_vl", REWARD_SETTINGS)
        config = load_merged_config(directory, processor.tokenizer)
        model = load_reward_model(Qwen2VLRewardModelBT, directory, config, processor, device, check_cancel,
                                  output_dim=settings["output_dim"], reward_token_id=settings["special_token_ids"][0],
                                  key_mapping=KEY_MAPPING)
        _patch_quantized_visual_dtype(model)
        return cls(model, processor, str(device), INSTRUCTION, PROMPT_WITH_SPECIAL_TOKEN)

    @torch.inference_mode()
    def reward(self, image_paths, prompts):
        return self.model(**self.prepare_batch(image_paths, prompts))["logits"]
