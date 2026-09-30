"""Adapter for the unchanged upstream HPSv3++ reward class."""
from transformers.vision_utils import get_vision_interpolation_indices_and_weights

from .upstream import load_model_module


def get_reward_model_class(source_directory=None):
    base = load_model_module(source_directory).Qwen3VLRewardModelFiLMHybrid

    class Compatible(base):
        _keep_in_fp32_modules_strict = ["rm_head", "cond_encoder", "film_gen"]

    Compatible.__name__ = base.__name__
    Compatible.__qualname__ = base.__name__
    Compatible.__module__ = __name__
    return Compatible


def install_vision_interpolation_hook(model):
    """Install score-mode cache and BF16 visual compatibility hooks."""
    def pre_hook(module, args, kwargs):
        grid = kwargs.get("grid_thw")
        if grid is None and len(args) > 1:
            grid = args[1]
        if grid is None:
            return args, kwargs
        indices, weights = get_vision_interpolation_indices_and_weights(
            grid, num_grid_per_side=module.num_grid_per_side,
            mode=module.interpolation_mode,
            align_corners=module.interpolation_align_corners,
            spatial_merge_size=module.spatial_merge_size,
        )
        kwargs["interp_indices"] = indices
        kwargs["interp_weights"] = weights.to(module.pos_embed.weight.dtype)
        return args, kwargs

    def score_cache_hook(module, args, kwargs):
        kwargs.setdefault("use_cache", False)
        return args, kwargs

    model.model.visual.register_forward_pre_hook(pre_hook, with_kwargs=True)
    model.model.register_forward_pre_hook(score_cache_hook, with_kwargs=True)
