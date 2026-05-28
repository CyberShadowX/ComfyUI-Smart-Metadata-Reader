from __future__ import annotations

from typing import Any

from ..models import NodeRecord


SAMPLER_CLASS_TYPES = {
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
}

SAMPLER_LIKE_INPUT_FIELDS = {
    "positive",
    "negative",
    "model",
    "vae",
    "seed",
    "noise_seed",
    "steps",
    "cfg",
    "cfg_scale",
    "sampler_name",
    "sampler",
    "scheduler",
    "scheduler_name",
    "denoise",
}


def is_sampler(node: NodeRecord) -> bool:
    return node.class_type in SAMPLER_CLASS_TYPES or _is_usdu_sampler_like(node)


def extract_sampler_settings(sampler: NodeRecord) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    field_map = {
        "seed": ("seed", "noise_seed"),
        "steps": ("steps",),
        "cfg": ("cfg", "cfg_scale"),
        "sampler_name": ("sampler_name", "sampler"),
        "scheduler": ("scheduler", "scheduler_name", "schedule"),
        "denoise": ("denoise",),
    }

    for output_name, field_names in field_map.items():
        value = _first_existing_input(sampler, field_names)
        if value is not None:
            settings[output_name] = value
    return settings


def _first_existing_input(node: NodeRecord, field_names: tuple[str, ...]) -> Any | None:
    for field_name in field_names:
        if field_name in node.inputs:
            return node.inputs[field_name]
    return None


def _is_usdu_sampler_like(node: NodeRecord) -> bool:
    lowered = node.class_type.lower().replace("_", " ")
    looks_like_usdu = ("ultimate" in lowered and "upscale" in lowered) or "usdu" in lowered
    if not looks_like_usdu:
        return False
    return any(field in node.inputs for field in SAMPLER_LIKE_INPUT_FIELDS)
