from __future__ import annotations

from typing import Any

from ..models import NodeRecord


CHECKPOINT_NAME_FIELDS = ("ckpt_name", "checkpoint", "model_name", "name")
VAE_NAME_FIELDS = ("vae_name", "name", "model_name")
LORA_NAME_FIELDS = ("lora_name", "name")


def is_checkpoint_loader(node: NodeRecord) -> bool:
    lowered = node.class_type.lower()
    if node.class_type in {"CheckpointLoaderSimple", "CheckpointLoader"}:
        return True
    return "checkpoint" in lowered and "loader" in lowered


def is_vae_loader(node: NodeRecord) -> bool:
    lowered = node.class_type.lower()
    return node.class_type == "VAELoader" or ("vae" in lowered and "loader" in lowered)


def is_lora_loader(node: NodeRecord) -> bool:
    lowered = node.class_type.lower()
    if node.class_type in {"LoraLoader", "LoraLoaderModelOnly"}:
        return True
    return "lora" in lowered and any(field in node.inputs for field in LORA_NAME_FIELDS)


def checkpoint_name(node: NodeRecord) -> str:
    return _first_string(node, CHECKPOINT_NAME_FIELDS)


def vae_name(node: NodeRecord) -> str:
    return _first_string(node, VAE_NAME_FIELDS)


def lora_name(node: NodeRecord) -> str:
    return _first_string(node, LORA_NAME_FIELDS)


def strength_model(node: NodeRecord) -> float | None:
    return _number_or_none(node.inputs.get("strength_model"))


def strength_clip(node: NodeRecord) -> float | None:
    return _number_or_none(node.inputs.get("strength_clip"))


def _first_string(node: NodeRecord, field_names: tuple[str, ...]) -> str:
    for field_name in field_names:
        value = node.inputs.get(field_name)
        if isinstance(value, str) and value:
            return value
    return ""


def _number_or_none(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None
