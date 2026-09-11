"""Standalone vendor copy of HPSv3's reward model class.

Vendored from MizzenAI/HPSv3 (hpsv3/model/qwen2vl_trainer.py, class
Qwen2VLRewardModelBT) at upstream commit
bd0c5fcb5f587617b0169c07222ab78d01e2f3c2. We copy only the model class
here — NOT the training
trainer/callback code from that file — because the original module also
imports heavy/training-only dependencies (matplotlib, tensorboard, trl's
RewardTrainer, several transformers.trainer internals) that are pinned to
transformers==4.45.2 in upstream requirements.txt and are prone to break
against the newer transformers version this project uses. Importing only
this class avoids that whole dependency chain.

Source: https://github.com/MizzenAI/HPSv3 (MIT license)
Copyright (c) 2024 HPSv3 Team. See THIRD_PARTY_NOTICES.md at the repository
root for the full upstream license text.
"""

from typing import List, Optional

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

    def __init__(
        self,
        config,
        output_dim=4,
        reward_token="last",
        special_token_ids=None,
        rm_head_type="default",
        rm_head_kwargs=None,
    ):
        super().__init__(config)
        # Transformers 5 stores Qwen2-VL text settings under text_config and
        # exposes the embedding module through the public model interface.
        # Keep the upstream reward class while accepting both layouts.
        if not hasattr(config, "hidden_size") and hasattr(config, "text_config"):
            config.hidden_size = config.text_config.hidden_size
        self.output_dim = output_dim
        if rm_head_type == "default":
            self.rm_head = nn.Linear(config.hidden_size, output_dim, bias=False)
        elif rm_head_type == "ranknet":
            if rm_head_kwargs is not None:
                for layer in range(rm_head_kwargs.get("num_layers", 3)):
                    if layer == 0:
                        self.rm_head = nn.Sequential(
                            nn.Linear(config.hidden_size, rm_head_kwargs["hidden_size"]),
                            nn.ReLU(),
                            nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                        )
                    elif layer < rm_head_kwargs.get("num_layers", 3) - 1:
                        self.rm_head.add_module(
                            f"layer_{layer}",
                            nn.Sequential(
                                nn.Linear(rm_head_kwargs["hidden_size"], rm_head_kwargs["hidden_size"]),
                                nn.ReLU(),
                                nn.Dropout(rm_head_kwargs.get("dropout", 0.1)),
                            ),
                        )
                    else:
                        self.rm_head.add_module(
                            "output_layer",
                            nn.Linear(rm_head_kwargs["hidden_size"], output_dim, bias=rm_head_kwargs.get("bias", False)),
                        )
            else:
                self.rm_head = nn.Sequential(
                    nn.Linear(config.hidden_size, 1024),
                    nn.ReLU(),
                    nn.Dropout(0.05),
                    nn.Linear(1024, 16),
                    nn.ReLU(),
                    nn.Linear(16, output_dim),
                )

        self.rm_head.to(torch.float32)
        self.reward_token = reward_token

        self.special_token_ids = special_token_ids
        if self.special_token_ids is not None:
            self.reward_token = "special"

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.Tensor] = None,
    ):
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # The reward checkpoint uses sequential text positions. The processor's
        # mm_token_type_ids are used by base-model generation, not this path.
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            core = self.model
            if pixel_values is not None:
                image_embeds = core.get_image_features(pixel_values, image_grid_thw).pooler_output
                image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds)
                image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1).expand_as(inputs_embeds)
                if image_mask.sum().item() != image_embeds.numel():
                    raise ValueError("Image features do not match image token positions")
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)
            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]  # [B, L, D]
        logits = self.rm_head(hidden_states.float())  # [B, L, N]

        if input_ids is not None:
            batch_size = input_ids.shape[0]
        else:
            batch_size = inputs_embeds.shape[0]

        pad_token_id = getattr(self.config, "pad_token_id", None)
        if pad_token_id is None and batch_size != 1:
            raise ValueError("Cannot handle batch sizes > 1 if no padding token is defined.")
        if pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = torch.eq(input_ids, pad_token_id).int().argmax(-1) - 1
                sequence_lengths = sequence_lengths % input_ids.shape[-1]
                sequence_lengths = sequence_lengths.to(logits.device)
            else:
                sequence_lengths = -1

        if self.reward_token == "last":
            pooled_logits = logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        elif self.reward_token == "mean":
            valid_lengths = torch.clamp(sequence_lengths, min=0, max=logits.size(1) - 1)
            pooled_logits = torch.stack([logits[i, : valid_lengths[i]].mean(dim=0) for i in range(batch_size)])
        elif self.reward_token == "special":
            special_token_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for special_token_id in self.special_token_ids:
                special_token_mask = special_token_mask | (input_ids == special_token_id)
            pooled_logits = logits[special_token_mask, ...]
            pooled_logits = pooled_logits.view(batch_size, 1, -1)
            pooled_logits = pooled_logits.view(batch_size, -1)
        else:
            raise ValueError("Invalid reward_token")

        return {"logits": pooled_logits}
