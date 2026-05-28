from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


@dataclass(frozen=True)
class ShapeOnlyTensor:
    shape: tuple[int, ...]


def _torch_module() -> Any | None:
    try:
        import torch
    except ImportError:
        return None
    return torch


def _numpy_module() -> Any | None:
    try:
        import numpy
    except ImportError:
        return None
    return numpy


def _image_tensor(image: Image.Image) -> Any:
    torch = _torch_module()
    width, height = image.size
    if torch is None:
        return ShapeOnlyTensor((1, height, width, 3))

    rgb_image = image.convert("RGB")
    numpy = _numpy_module()
    if numpy is not None and hasattr(torch, "from_numpy"):
        array = numpy.asarray(rgb_image, dtype=numpy.float32)
        return torch.from_numpy(array.reshape(1, height, width, 3)) / 255.0

    data = list(rgb_image.tobytes())
    return torch.tensor(data, dtype=torch.float32).reshape(1, height, width, 3) / 255.0


def _mask_tensor(image: Image.Image) -> Any:
    torch = _torch_module()
    width, height = image.size
    if torch is None:
        return ShapeOnlyTensor((1, height, width))

    numpy = _numpy_module()
    if "A" in image.getbands():
        alpha = image.getchannel("A")
        if numpy is not None and hasattr(torch, "from_numpy"):
            alpha_array = numpy.asarray(alpha, dtype=numpy.float32)
            mask_array = (1.0 - (alpha_array / 255.0)).reshape(1, height, width)
            return torch.from_numpy(mask_array)
        mask_data = [1.0 - (value / 255.0) for value in alpha.tobytes()]
    else:
        if numpy is not None and hasattr(torch, "from_numpy"):
            return torch.from_numpy(numpy.zeros((1, height, width), dtype=numpy.float32))
        mask_data = [0.0] * (width * height)
    return torch.tensor(mask_data, dtype=torch.float32).reshape(1, height, width)


def load_image_and_mask(image_path: str | Path) -> tuple[Any, Any, int, int]:
    path = Path(image_path)
    with Image.open(path) as loaded:
        image = ImageOps.exif_transpose(loaded)
        width, height = image.size
        return _image_tensor(image), _mask_tensor(image), width, height
