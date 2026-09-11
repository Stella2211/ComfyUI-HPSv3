"""Lazy adapters for the unchanged upstream HPSv3++ reward classes."""

def _compat_class(name, source_directory=None):
    from .upstream import load_model_module

    base = getattr(load_model_module(source_directory), name)

    class Compatible(base):
        _keep_in_fp32_modules_strict = ["rm_head", "cond_encoder", "film_gen"]

    Compatible.__name__ = name
    Compatible.__qualname__ = name
    Compatible.__module__ = __name__
    return Compatible


def get_reward_model_class(name="Qwen3VLRewardModelFiLMHybrid", source_directory=None):
    return _compat_class(name, source_directory)


def install_vision_interpolation_hook(model):
    """Install score-mode cache and BF16 visual compatibility hooks."""
    from transformers.vision_utils import get_vision_interpolation_indices_and_weights

    visual = getattr(getattr(model, "model", None), "visual", None)
    if visual is None:
        raise AttributeError("Qwen3-VL reward model has no visual module")

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

    handles = [visual.register_forward_pre_hook(pre_hook, with_kwargs=True)]
    backbone = getattr(model, "model", None)
    if backbone is not None:
        def score_cache_hook(module, args, kwargs):
            kwargs.setdefault("use_cache", False)
            return args, kwargs

        handles.append(backbone.register_forward_pre_hook(score_cache_hook, with_kwargs=True))
    rm_head = getattr(model, "rm_head", None)
    if rm_head is not None:
        def cast_head_input(module, args):
            if args and hasattr(args[0], "to"):
                return (args[0].to(dtype=next(module.parameters()).dtype), *args[1:])
            return args

        handles.append(rm_head.register_forward_pre_hook(cast_head_input))

    class _Handles:
        def remove(self):
            for handle in handles:
                handle.remove()

    return _Handles()


def __getattr__(name):
    if name in {"Qwen3VLRewardModelBT", "Qwen3VLRewardModelFiLMContinuous", "Qwen3VLRewardModelFiLMHybrid"}:
        return _compat_class(name)
    raise AttributeError(name)


__all__ = ["Qwen3VLRewardModelBT", "Qwen3VLRewardModelFiLMContinuous", "Qwen3VLRewardModelFiLMHybrid", "get_reward_model_class", "install_vision_interpolation_hook"]
