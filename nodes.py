import json
import math
import os
import textwrap

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, PngImagePlugin

import folder_paths
from comfy.cli_args import args

from .backend import HPSv3PPModel, list_models


def _single(value, name):
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ValueError(f"{name} accepts one model or settings value")
        return value[0]
    return value


def _images(value):
    if isinstance(value, torch.Tensor):
        value = [value]
    if not isinstance(value, (list, tuple)):
        raise ValueError("images must be a ComfyUI IMAGE batch")
    output = []
    for image in value:
        if not isinstance(image, torch.Tensor):
            raise ValueError("images must contain ComfyUI IMAGE tensors")
        if image.ndim == 4:
            output.extend(image[i] for i in range(image.shape[0]))
        elif image.ndim == 3:
            output.append(image)
        else:
            raise ValueError("each image must have shape [H, W, C] or [B, H, W, C]")
    return output


def _to_pil(image):
    array = image.detach().cpu().numpy()
    array = np.clip(array * 255.0, 0, 255).astype(np.uint8)
    if array.shape[-1] == 1:
        array = array[..., 0]
    elif array.shape[-1] not in (3, 4):
        raise ValueError("IMAGE tensors must have 1, 3, or 4 channels")
    return Image.fromarray(array)


def _to_tensor(image):
    array = np.array(image, dtype=np.float32) / 255.0
    if array.ndim == 2:
        array = array[..., None]
    return torch.from_numpy(array).unsqueeze(0)


def _banner(image, score):
    width, height = image.size
    font = ImageFont.load_default(size=max(10, min(28, width // 32)))
    label = f"HPSv3++ score: {score:.4f}"
    label = "\n".join(textwrap.wrap(label, width=max(1, int((width - 8) / font.getlength("M")))))
    left, top, right, bottom = ImageDraw.Draw(image).multiline_textbbox((0, 0), label, font=font)
    bar_height = bottom - top + 12
    banner = Image.new(image.mode, (width, height + bar_height), "white")
    banner.paste(image, (0, bar_height))
    ImageDraw.Draw(banner).multiline_text((4, 6 - top), label, fill="black", font=font)
    return banner


def _save(image, filename_prefix, score, prompt, model_name, mode, batch_number, workflow_prompt, extra_pnginfo):
    output_dir = folder_paths.get_output_directory()
    full_dir, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
        filename_prefix, output_dir, image.width, image.height
    )
    os.makedirs(full_dir, exist_ok=True)
    metadata = None
    if not args.disable_metadata:
        metadata = PngImagePlugin.PngInfo()
        if workflow_prompt is not None:
            metadata.add_text("prompt", json.dumps(workflow_prompt))
        for key, value in (extra_pnginfo or {}).items():
            metadata.add_text(key, json.dumps(value))
        if mode in ("metadata", "both"):
            metadata.add_text("hpsv3pp", json.dumps({
                "model": model_name, "score": score, "prompt": prompt,
            }, ensure_ascii=False))
    filename = filename.replace("%batch_num%", str(batch_number))
    file = f"{filename}_{counter:05}_.png"
    image.save(os.path.join(full_dir, file), pnginfo=metadata, compress_level=4)
    return {"filename": file, "subfolder": subfolder, "type": "output"}


class HPSv3PPModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": (list_models(), {
            "tooltip": "HPSv3-PlusPlus-bnb-NF4 downloads automatically from Hugging Face when missing (about 6.5 GB). Download progress appears in the ComfyUI console.",
        })}}

    RETURN_TYPES = ("HPSV3PP_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "HPSv3++"

    def load(self, model):
        return (HPSv3PPModel(_single(model, "model")),)


class HPSv3PPScore:
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    OUTPUT_NODE = True
    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("images", "scores")
    FUNCTION = "score"
    CATEGORY = "HPSv3++"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("HPSV3PP_MODEL",),
            "images": ("IMAGE",),
            "prompt": ("STRING", {"multiline": True, "default": ""}),
            "score_mode": (["banner", "metadata", "both"], {"default": "banner"}),
            "filename_prefix": ("STRING", {
                "default": "HPSv3pp",
                "tooltip": "Output folder and filename prefix. Supports ComfyUI substitutions, e.g. %date:yyyy%/%date:MM%/%date:dd%/HPSv3pp.",
            }),
        }, "hidden": {"workflow_prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    def score(self, model, images, prompt, score_mode, filename_prefix, workflow_prompt=None, extra_pnginfo=None):
        model = _single(model, "model")
        score_mode = _single(score_mode, "score_mode")
        filename_prefix = _single(filename_prefix, "filename_prefix")
        if score_mode not in ("banner", "metadata", "both"):
            raise ValueError("score_mode must be banner, metadata, or both")
        if score_mode in ("metadata", "both") and args.disable_metadata:
            raise ValueError("Metadata is disabled. Select banner mode or remove --disable-metadata.")
        image_tensors = _images(images)
        pil_images = [_to_pil(image) for image in image_tensors]
        if not pil_images:
            raise ValueError("Provide at least one image.")
        prompts = list(prompt) if isinstance(prompt, (list, tuple)) else [prompt]
        if len(prompts) == 1:
            prompts *= len(pil_images)
        elif len(prompts) != len(pil_images):
            raise ValueError(f"got {len(pil_images)} images but {len(prompts)} prompts; provide one prompt or one per image")
        scores = model.score(pil_images, prompts)
        if len(scores) != len(pil_images):
            raise ValueError(f"model returned {len(scores)} scores for {len(pil_images)} images")
        scores = [float(score) for score in scores]
        if not all(math.isfinite(score) for score in scores):
            raise ValueError("HPSv3++ returned a non-finite score")
        workflow_prompt = _single(workflow_prompt, "workflow_prompt")
        extra_pnginfo = _single(extra_pnginfo, "extra_pnginfo")
        rendered = [_banner(image, score) if score_mode in ("banner", "both") else image for image, score in zip(pil_images, scores)]
        ui_images = []
        for i, (image, score, text) in enumerate(zip(rendered, scores, prompts)):
            ui_images.append(_save(image, filename_prefix, score, text, model.model_name, score_mode, i, workflow_prompt, extra_pnginfo))
        return {"ui": {"images": ui_images}, "result": ([_to_tensor(image) for image in rendered], scores)}


class HPSv3PPCaption:
    OUTPUT_IS_LIST = (True,)
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("captions",)
    FUNCTION = "caption"
    CATEGORY = "HPSv3++"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("HPSV3PP_MODEL",),
            "images": ("IMAGE",),
            "max_new_tokens": ("INT", {"default": 96, "min": 16, "max": 512}),
        }}

    def caption(self, model, images, max_new_tokens=96):
        model = _single(model, "model")
        max_new_tokens = int(_single(max_new_tokens, "max_new_tokens"))
        if not 16 <= max_new_tokens <= 512:
            raise ValueError("max_new_tokens must be between 16 and 512.")
        pil_images = [_to_pil(image) for image in _images(images)]
        if not pil_images:
            raise ValueError("Provide at least one image.")
        captions = model.caption(pil_images, max_new_tokens)
        if len(captions) != len(pil_images) or any(not isinstance(text, str) or not text.strip() for text in captions):
            raise ValueError("HPSv3++ did not return a non-empty caption for each image.")
        return (captions,)


NODE_CLASS_MAPPINGS = {
    "HPSv3PPModelLoader": HPSv3PPModelLoader,
    "HPSv3PPScore": HPSv3PPScore,
    "HPSv3PPCaption": HPSv3PPCaption,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HPSv3PPModelLoader": "HPSv3++ Model Loader",
    "HPSv3PPScore": "HPSv3++ Score",
    "HPSv3PPCaption": "HPSv3++ Caption",
}
