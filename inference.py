"""Thin ComfyUI adapter for the vendored HPSv3 runtime."""

import gc
import math
from pathlib import Path
import threading
import traceback

import torch
from PIL import Image

from ._vendor.hpsv3_4bit import load_model


_INFERENCE_LOCK = threading.Lock()


def _prepare_image(image):
    """Validate and normalize an input image for the runtime API."""
    if not isinstance(image, Image.Image):
        raise TypeError("Inference accepts decoded PIL images only.")
    width, height = image.size
    if min(width, height) <= 0 or max(width, height) / min(width, height) > 200:
        raise ValueError("Unsupported image aspect ratio.")
    if image.mode == "RGBA":
        return Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
    return image.convert("RGB")


def _install_cancellation_hooks(model, check_cancel):
    """Install cancellation checks around transformer blocks."""
    handles = []
    try:
        for name, layer in model.named_modules():
            parent = name.rpartition(".")[0]
            if parent.endswith((".layers", ".blocks")):
                handles.append(layer.register_forward_pre_hook(lambda module, args: check_cancel()))
    except BaseException:
        for handle in handles:
            handle.remove()
        raise
    return handles


def _execute(family, directory, operation, images, check_cancel, **options):
    from comfy import model_management
    from transformers import StoppingCriteria, StoppingCriteriaList

    prompts = options.get("prompts", [])
    if operation == "score" and len(prompts) != len(images):
        raise ValueError("Provide one prompt for each image.")
    prepared_images = [_prepare_image(image) for image in images]

    device = model_management.get_torch_device()
    model_management.free_memory(float("inf"), device)
    model_management.soft_empty_cache()
    session = load_model(family, Path(directory), device, check_cancel=check_cancel)
    model = session.model

    class CancelGeneration(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            check_cancel()
            return False

    handles = _install_cancellation_hooks(model, check_cancel)
    try:
        check_cancel()
        results = []
        for index, image in enumerate(prepared_images):
            check_cancel()
            if operation == "score":
                value = session.score(image, prompts[index])
                if not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError("Model returned a non-finite score.")
                results.append(float(value))
            else:
                value = session.caption(
                    image,
                    max_new_tokens=options.get("max_new_tokens", 96),
                    stopping_criteria=StoppingCriteriaList([CancelGeneration()]),
                )
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Model returned an empty caption.")
                results.append(value.strip())
        return results
    finally:
        for handle in handles:
            handle.remove()


def run_inference(family, model_path, operation, images, *, check_cancel, **options):
    if family not in ("hpsv3", "hpsv3pp") or operation not in ("score", "caption"):
        raise ValueError("Unknown HPS model or operation.")
    while not _INFERENCE_LOCK.acquire(timeout=0.1):
        check_cancel()
    try:
        check_cancel()
        return _execute(family, Path(model_path), operation, images, check_cancel, **options)
    except BaseException as error:
        traceback.clear_frames(error.__traceback__)
        raise
    finally:
        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        finally:
            _INFERENCE_LOCK.release()
