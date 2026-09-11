"""Lazy access to exact prompt constants from the external source."""


def load_prompts(source_directory=None):
    from .upstream import load_prompts as _load_prompts
    return _load_prompts(source_directory)


def __getattr__(name):
    if name in {"INSTRUCTION", "prompt_with_special_token", "prompt_without_special_token"}:
        return load_prompts()[name]
    raise AttributeError(name)


__all__ = ["INSTRUCTION", "prompt_with_special_token", "prompt_without_special_token", "load_prompts"]
