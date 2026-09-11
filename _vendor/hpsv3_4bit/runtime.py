"""Canonical local-only runtime for HPSv3 and HPSv3++ NF4 checkpoints.

Callers own batching and lifecycle policy; this module owns model construction
and single-pair or batch inference. Source preparation is explicit and happens
outside this API. No Hub download, subprocess, remote code, or runtime
installation is performed while loading or scoring.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence
import math

from PIL import Image


class HPSv3Session:
    def __init__(self, family: str, inferencer, check_cancel: Callable[[], None] | None = None):
        self.family = family
        self.inferencer = inferencer
        self.model = inferencer.model
        self._check_cancel = check_cancel or (lambda: None)

    def score(self, image: Image.Image, prompt: str) -> float:
        return self.score_batch([image], [prompt])[0]

    def score_batch(self, images: Sequence[Image.Image], prompts: Sequence[str],
                    iter_step: float = 0.0) -> list[float]:
        """Evaluate one actual batch; HPSv3++ scores depend on its composition.

        Integrations requiring independent scores should use ``score`` instead.
        """
        if not images or len(images) != len(prompts):
            raise ValueError("Provide a nonempty batch with one prompt per image.")
        if not math.isfinite(iter_step) or not 0.0 <= iter_step <= 1.0:
            raise ValueError("iter_step must be finite and in [0, 1].")
        if self.family == "hpsv3" and iter_step != 0.0:
            raise ValueError("HPSv3 does not support iteration conditioning.")
        self._check_cancel()
        options = {"iter_step": iter_step} if self.family == "hpsv3pp" else {}
        values = self.inferencer.score(images, prompts, **options)
        self._check_cancel()
        if len(values) != len(images):
            raise RuntimeError(f"Expected {len(images)} scores, got {len(values)}")
        scores = [float(value) for value in values]
        if not all(math.isfinite(value) for value in scores):
            raise ValueError("Model returned a non-finite score.")
        return scores

    def caption(self, image: Image.Image, max_new_tokens: int = 96, stopping_criteria=None) -> str:
        self._check_cancel()
        values = self.inferencer.caption([image], max_new_tokens=max_new_tokens, stopping_criteria=stopping_criteria)
        self._check_cancel()
        if len(values) != 1 or not values[0].strip():
            raise ValueError("Model returned an empty caption.")
        return values[0].strip()


def load_model(
    family: str,
    directory: str | Path,
    device: str = "cuda",
    check_cancel: Callable[[], None] | None = None,
    processor_directory: str | Path | None = None,
    source_directory: str | Path | None = None,
) -> HPSv3Session:
    """Load a local merged NF4 model and return an inference session.

    An optional processor override must also be a local directory. Hub
    resolution belongs to the caller, such as the standalone CLI.
    """
    if family == "hpsv3":
        from .hpsv3.quantized import HPSv3QuantizedInferencer

        inferencer_class = HPSv3QuantizedInferencer
    elif family == "hpsv3pp":
        from .hpsv3pp.quantized import HPSv3PPQuantizedInferencer

        inferencer_class = HPSv3PPQuantizedInferencer
    else:
        raise ValueError(f"Unknown HPS model family: {family}")
    options = {}
    if processor_directory is not None:
        options["processor_directory"] = str(Path(processor_directory))
    if family == "hpsv3pp" and source_directory is not None:
        options["source_directory"] = str(Path(source_directory))
    inferencer = inferencer_class.from_merged_dir(
        merged_dir=str(Path(directory)), device=str(device), check_cancel=check_cancel, **options
    )
    return HPSv3Session(family, inferencer, check_cancel)
