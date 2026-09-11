__all__ = ["HPSv3PPQuantizedInferencer"]


def __getattr__(name):
    if name == "HPSv3PPQuantizedInferencer":
        from .quantized import HPSv3PPQuantizedInferencer

        return HPSv3PPQuantizedInferencer
    raise AttributeError(name)
