__all__ = ["HPSv3QuantizedInferencer"]


def __getattr__(name):
    if name == "HPSv3QuantizedInferencer":
        from .quantized import HPSv3QuantizedInferencer

        return HPSv3QuantizedInferencer
    raise AttributeError(name)
