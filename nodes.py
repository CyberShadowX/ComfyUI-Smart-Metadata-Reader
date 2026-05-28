from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

try:
    from .smart_metadata_reader.image_io import load_image_and_mask
    from .smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata
except ImportError:
    from smart_metadata_reader.image_io import load_image_and_mask
    from smart_metadata_reader.metadata_reader import parse_metadata_bundle, read_metadata

try:
    import folder_paths  # type: ignore
except ImportError:
    folder_paths = None  # type: ignore


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class SmartMetadataReader:
    RETURN_TYPES = (
        "IMAGE",
        "MASK",
        "STRING",
        "STRING",
        "INT",
        "INT",
        "FLOAT",
        "INT",
        "INT",
        "STRING",
        "STRING",
        "STRING",
    )
    RETURN_NAMES = (
        "image",
        "mask",
        "positive",
        "negative",
        "seed",
        "steps",
        "cfg",
        "width",
        "height",
        "model_name",
        "filename",
        "setting",
    )
    FUNCTION = "read_metadata"
    CATEGORY = "metadata/image"

    @classmethod
    def INPUT_TYPES(cls) -> dict[str, dict[str, Any]]:
        return {
            "required": {
                "image": (_input_images(), {"image_upload": True}),
                "parameter_index": ("INT", {"default": 0, "min": 0}),
                "prefer_cached_text": ("BOOLEAN", {"default": True}),
                "include_raw_json": ("BOOLEAN", {"default": True}),
                "max_depth": ("INT", {"default": 40, "min": 1, "max": 200}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, image: str, **kwargs: Any) -> str:
        del kwargs
        image_path = _resolve_image_path_safely(image)
        return _file_digest(image_path)

    @classmethod
    def VALIDATE_INPUTS(cls, image: str, **kwargs: Any) -> bool | str:
        del kwargs
        if folder_paths is not None and hasattr(folder_paths, "exists_annotated_filepath"):
            try:
                if folder_paths.exists_annotated_filepath(image):
                    return True
                return f"Invalid image file: {image}"
            except Exception:
                pass

        image_path = _resolve_image_path_safely(image)
        if Path(image_path).exists():
            return True
        return f"Invalid image file: {image_path}"

    def read_metadata(
        self,
        image: str,
        parameter_index: int = 0,
        prefer_cached_text: bool = True,
        include_raw_json: bool = True,
        max_depth: int = 40,
    ) -> tuple[Any, Any, str, str, int, int, float, int, int, str, str, str]:
        image_path = _resolve_image_path(image)
        image_tensor, mask_tensor, image_width, image_height = _load_image_or_raise(image_path)
        filename = Path(image_path).name

        try:
            bundle = read_metadata(image_path)
            result = parse_metadata_bundle(
                bundle,
                parameter_index=parameter_index,
                prefer_cached_text=prefer_cached_text,
                include_raw_json=include_raw_json,
                max_depth=max_depth,
            )
        except Exception as exc:
            setting = _failed_setting(filename, image_width, image_height, exc)
            return (
                image_tensor,
                mask_tensor,
                "",
                "",
                -1,
                0,
                0.0,
                image_width,
                image_height,
                "",
                filename,
                setting,
            )

        return (
            image_tensor,
            mask_tensor,
            result.positive,
            result.negative,
            result.seed,
            result.steps,
            result.cfg,
            result.width or image_width,
            result.height or image_height,
            result.model_name,
            result.filename or filename,
            result.setting,
        )


def _input_images() -> list[str]:
    if folder_paths is None or not hasattr(folder_paths, "get_input_directory"):
        return []

    try:
        input_dir_value = folder_paths.get_input_directory()
    except Exception:
        return []
    if not input_dir_value:
        return []

    input_dir = Path(input_dir_value)
    if not input_dir.is_dir():
        return []

    images: list[str] = []
    for root, _dirs, files in os.walk(input_dir):
        for filename in files:
            if Path(filename).suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            full_path = Path(root) / filename
            relative_path = os.path.relpath(full_path, input_dir)
            images.append(relative_path.replace(os.sep, "/"))
    return sorted(images)


def _resolve_image_path(image: str) -> str:
    if folder_paths is not None and hasattr(folder_paths, "get_annotated_filepath"):
        return str(folder_paths.get_annotated_filepath(image))
    if folder_paths is not None and hasattr(folder_paths, "get_input_directory"):
        return str(Path(folder_paths.get_input_directory()) / image)
    return os.fspath(image)


def _resolve_image_path_safely(image: str) -> str:
    try:
        return _resolve_image_path(image)
    except Exception:
        pass

    if folder_paths is not None and hasattr(folder_paths, "get_input_directory"):
        try:
            input_dir = folder_paths.get_input_directory()
        except Exception:
            input_dir = None
        if input_dir:
            return str(Path(input_dir) / image)
    return os.fspath(image)


def _load_image_or_raise(image_path: str) -> tuple[Any, Any, int, int]:
    try:
        return load_image_and_mask(image_path)
    except Exception as exc:
        raise RuntimeError(f"Failed to load image '{image_path}': {exc}") from exc


def _file_digest(image_path: str) -> str:
    path = Path(image_path)
    try:
        hasher = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except OSError:
        return f"missing:{image_path}"


def _failed_setting(filename: str, width: int, height: int, exc: Exception) -> str:
    size = f"{width} x {height}" if width > 0 and height > 0 else "unknown"
    return "\n".join(
        [
            f"Filename: {filename or 'unknown'}",
            "Source: unknown",
            "Model: unknown",
            "VAE: unknown",
            "LoRA: none",
            "Seed: -1",
            "Steps: 0",
            "CFG: 0.0",
            "Sampler: unknown",
            "Scheduler: unknown",
            f"Size: {size}",
            "Status: FAILED",
            "Confidence: 0.00",
            f"Error: {exc}",
        ]
    )


NODE_CLASS_MAPPINGS = {"SmartMetadataReader": SmartMetadataReader}
NODE_DISPLAY_NAME_MAPPINGS = {"SmartMetadataReader": "Smart Metadata Reader"}
