"""Thin ComfyUI adapter for the vendored HPSv3 runtime."""

import gc
from pathlib import Path
import threading
import traceback

import torch
from PIL import Image
from transformers import StoppingCriteria, StoppingCriteriaList

from comfy import model_management

from ._vendor.hpsv3_4bit import load_model


_INFERENCE_LOCK = threading.Lock()


def _prepare_image(image):
    """Validate and normalize an input image for the runtime API."""
    if not isinstance(image, Image.Image):
        raise TypeError("Inference accepts decoded PIL images only.")
    if image.mode == "RGBA":
        return Image.alpha_composite(Image.new("RGBA", image.size, "white"), image).convert("RGB")
    return image.convert("RGB")


class CancelGeneration(StoppingCriteria):
    def __init__(self, check_cancel):
        self.check_cancel = check_cancel

    def __call__(self, input_ids, scores, **kwargs):
        self.check_cancel()
        return False


def _install_cancellation_hooks(model, check_cancel):
    """Install cancellation checks around transformer blocks."""
    handles = []
    for name, layer in model.named_modules():
        parent = name.rpartition(".")[0]
        if parent.endswith((".layers", ".blocks")):
            handles.append(layer.register_forward_pre_hook(lambda module, args: check_cancel()))
    return handles


def _execute(family, directory, operation, images, check_cancel, prompts, max_new_tokens):
    if operation == "score" and len(prompts) != len(images):
        raise ValueError("Provide one prompt for each image.")
    prepared_images = [_prepare_image(image) for image in images]

    device = model_management.get_torch_device()
    model_management.free_memory(float("inf"), device)
    model_management.soft_empty_cache()
    session = load_model(family, Path(directory), device, check_cancel=check_cancel)
    model = session.model

    handles = _install_cancellation_hooks(model, check_cancel)
    try:
        check_cancel()
        results = []
        for index, image in enumerate(prepared_images):
            check_cancel()
            if operation == "score":
                results.append(session.score(image, prompts[index]))
            else:
                results.append(session.caption(
                    image,
                    max_new_tokens=max_new_tokens,
                    stopping_criteria=StoppingCriteriaList([CancelGeneration(check_cancel)]),
                ))
        return results
    finally:
        for handle in handles:
            handle.remove()


def run_inference(family, model_path, operation, images, *, check_cancel, prompts=(), max_new_tokens=96):
    while not _INFERENCE_LOCK.acquire(timeout=0.1):
        check_cancel()
    try:
        check_cancel()
        return _execute(family, Path(model_path), operation, images, check_cancel, prompts, max_new_tokens)
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
