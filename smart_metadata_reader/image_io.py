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


def _image_tensor(image: Image.Image) -> Any:
    torch = _torch_module()
    width, height = image.size
    if torch is None:
        return ShapeOnlyTensor((1, height, width, 3))

    rgb_image = image.convert("RGB")
    data = list(rgb_image.getdata())
    return torch.tensor(data, dtype=torch.float32).reshape(1, height, width, 3) / 255.0


def _mask_tensor(image: Image.Image) -> Any:
    torch = _torch_module()
    width, height = image.size
    if torch is None:
        return ShapeOnlyTensor((1, height, width))

    if "A" in image.getbands():
        alpha = image.getchannel("A")
        mask_data = [1.0 - (value / 255.0) for value in alpha.getdata()]
    else:
        mask_data = [0.0] * (width * height)
    return torch.tensor(mask_data, dtype=torch.float32).reshape(1, height, width)


def load_image_and_mask(image_path: str | Path) -> tuple[Any, Any, int, int]:
    path = Path(image_path)
    with Image.open(path) as loaded:
        image = ImageOps.exif_transpose(loaded)
        width, height = image.size
        return _image_tensor(image), _mask_tensor(image), width, height
