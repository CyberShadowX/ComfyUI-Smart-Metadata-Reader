from __future__ import annotations

from typing import Any

from smart_metadata_reader.models import NodeRecord


SAMPLER_CLASS_TYPES = {
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "SamplerCustomAdvanced",
}


def is_sampler(node: NodeRecord) -> bool:
    return node.class_type in SAMPLER_CLASS_TYPES


def extract_sampler_settings(sampler: NodeRecord) -> dict[str, Any]:
    settings: dict[str, Any] = {}
    field_map = {
        "seed": ("seed", "noise_seed"),
        "steps": ("steps",),
        "cfg": ("cfg", "cfg_scale"),
        "sampler_name": ("sampler_name", "sampler"),
        "scheduler": ("scheduler", "schedule"),
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
