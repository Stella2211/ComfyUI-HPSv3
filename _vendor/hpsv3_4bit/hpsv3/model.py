"""HPSv3 reward model for inference.

Adapted from MizzenAI/HPSv3 (hpsv3/model/qwen2vl_trainer.py, class
Qwen2VLRewardModelBT) at upstream commit
bd0c5fcb5f587617b0169c07222ab78d01e2f3c2. Only the released configuration
is kept: the "ranknet" head without extra kwargs, pooled at the special
reward token. The training trainer/callback code and its dependencies are
not included.

Source: https://github.com/MizzenAI/HPSv3 (MIT license)
Copyright (c) 2024 HPSv3 Team. See THIRD_PARTY_NOTICES.md at the repository
root for the full upstream license text.
"""

import torch
import torch.nn as nn
from transformers import Qwen2VLForConditionalGeneration


class Qwen2VLRewardModelBT(Qwen2VLForConditionalGeneration):
    """Qwen2-VL backbone + a small reward head (rm_head).

    output_dim=2 in HPSv3's released config: the head predicts
    (mu, log_sigma) of a Gaussian over human preference score; HPSv3's own
    inference code selects index 0 (mu) as the final scalar score.
    """

    # Preserve checkpoint FP32 head weights during BF16 backbone loading.
    _keep_in_fp32_modules_strict = ["rm_head"]

    def __init__(self, config, output_dim, reward_token_id):
        super().__init__(config)
        # Identity replaces training-only dropout and keeps checkpoint indices.
        self.rm_head = nn.Sequential(
            nn.Linear(config.text_config.hidden_size, 1024),
            nn.ReLU(),
            nn.Identity(),
            nn.Linear(1024, 16),
            nn.ReLU(),
            nn.Linear(16, output_dim),
        )
        self.reward_token_id = reward_token_id

    def forward(self, input_ids, attention_mask=None, pixel_values=None, image_grid_thw=None, mm_token_type_ids=None):
        # The reward checkpoint uses sequential text positions. The processor's
        # mm_token_type_ids are used by base-model generation, not this path.
        inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image_embeds = self.model.get_image_features(pixel_values, image_grid_thw).pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds)
            image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
            if image_mask.sum().item() != image_embeds.numel():
                raise ValueError("Image features do not match image token positions")
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
        hidden_states = self.model.language_model(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask, use_cache=False,
        ).last_hidden_state
        # Callers guarantee exactly one reward token per row.
        return {"logits": self.rm_head(hidden_states[input_ids == self.reward_token_id].float())}
