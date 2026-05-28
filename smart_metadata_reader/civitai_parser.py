from __future__ import annotations

import json
import re
from typing import Any

from .models import LoraInfo, ParseResult


SETTING_KEYS = (
    "Steps",
    "Sampler",
    "CFG scale",
    "Seed",
    "Size",
    "Model type",
    "Model",
    "Created Date",
    "Civitai resources",
    "Civitai metadata",
)


def parse_exif_user_comment(comment: str, source_label: str = "EXIF UserComment") -> ParseResult:
    civitai_metadata = _extract_labeled_json(comment, "Civitai metadata")
    civitai_resources = _extract_labeled_json(comment, "Civitai resources")
    source_format = _source_format(source_label, civitai_metadata, civitai_resources)

    positive, negative = _extract_prompts(comment)
    if isinstance(civitai_metadata, dict):
        positive = _string_value(civitai_metadata.get("prompt")) or positive
        negative = _string_value(civitai_metadata.get("negativePrompt")) or negative

    model_name = _extract_checkpoint_name(civitai_resources)
    if not model_name:
        model_name = _extract_setting_value(comment, "Model type") or _extract_setting_value(
            comment, "Model"
        )

    partial_result: dict[str, Any] = {"source_format": source_format}
    if civitai_metadata is not None:
        partial_result["civitai_metadata"] = civitai_metadata
    if civitai_resources is not None:
        partial_result["civitai_resources"] = civitai_resources

    return ParseResult(
        positive=positive,
        negative=negative,
        seed=_int_or_default(_extract_setting_value(comment, "Seed"), -1),
        steps=_int_or_default(_extract_setting_value(comment, "Steps"), 0),
        cfg=_float_or_default(_extract_setting_value(comment, "CFG scale"), 0.0),
        width=_width_from_size(_extract_setting_value(comment, "Size")),
        height=_height_from_size(_extract_setting_value(comment, "Size")),
        model_name=model_name,
        sampler_name=_extract_setting_value(comment, "Sampler"),
        loras=_extract_loras(civitai_resources),
        partial_result=partial_result,
        debug_trace=(
            "EXIF_USER_COMMENT_FALLBACK:\n"
            f"Parsed metadata from {source_label}."
        ),
        confidence=0.75,
        status_message=f"{source_format} fallback",
    )


def _source_format(
    source_label: str,
    civitai_metadata: Any,
    civitai_resources: Any,
) -> str:
    if civitai_metadata is not None or civitai_resources is not None:
        return f"{source_label} / Civitai metadata"
    return source_label


def _extract_prompts(comment: str) -> tuple[str, str]:
    settings_start = _settings_start(comment)
    prompt_block = comment[:settings_start].strip(" \t\r\n,") if settings_start else comment.strip()
    negative_match = re.search(r"(?im)^Negative prompt:\s*", prompt_block)
    if negative_match is None:
        return prompt_block, ""
    positive = prompt_block[: negative_match.start()].strip()
    negative = prompt_block[negative_match.end() :].strip()
    return positive, negative


def _settings_start(comment: str) -> int:
    match = re.search(r"(?im)(?:^|\n|,\s*)Steps:\s*", comment)
    if match is None:
        return 0
    return match.start()


def _extract_labeled_json(text: str, label: str) -> Any | None:
    marker = f"{label}:"
    index = text.find(marker)
    if index < 0:
        return None

    start = index + len(marker)
    while start < len(text) and text[start].isspace():
        start += 1

    try:
        value, _end = json.JSONDecoder().raw_decode(text[start:])
    except json.JSONDecodeError:
        return None
    return value


def _extract_setting_value(text: str, key: str) -> str:
    known_keys_pattern = "|".join(re.escape(known_key) for known_key in SETTING_KEYS)
    pattern = (
        rf"(?:^|[\n,]\s*){re.escape(key)}:\s*"
        rf"(.*?)"
        rf"(?=,\s*(?:{known_keys_pattern}):|$)"
    )
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        return ""
    return match.group(1).strip()


def _extract_checkpoint_name(resources: Any) -> str:
    if not isinstance(resources, list):
        return ""
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        resource_type = str(resource.get("type", "")).lower()
        if resource_type != "checkpoint":
            continue
        return _string_value(resource.get("modelName")) or _string_value(
            resource.get("modelVersionName")
        )
    return ""


def _extract_loras(resources: Any) -> list[LoraInfo]:
    if not isinstance(resources, list):
        return []

    loras: list[LoraInfo] = []
    for index, resource in enumerate(resources):
        if not isinstance(resource, dict):
            continue
        resource_type = str(resource.get("type", "")).lower()
        if "lora" not in resource_type:
            continue
        name = _string_value(resource.get("modelName")) or _string_value(
            resource.get("modelVersionName")
        )
        if not name:
            continue
        weight = _number_or_none(resource.get("weight"))
        loras.append(
            LoraInfo(
                node_id=f"civitai:{index}",
                class_type="CivitaiResource",
                lora_name=name,
                strength_model=weight,
                strength_clip=weight,
                path=["EXIF UserComment", f"Civitai resources[{index}]"],
            )
        )
    return loras


def _width_from_size(size: str) -> int:
    width, _height = _parse_size(size)
    return width


def _height_from_size(size: str) -> int:
    _width, height = _parse_size(size)
    return height


def _parse_size(size: str) -> tuple[int, int]:
    match = re.match(r"\s*(\d+)\s*x\s*(\d+)\s*$", size, flags=re.IGNORECASE)
    if match is None:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def _string_value(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _int_or_default(value: str | None, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _float_or_default(value: str | None, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _number_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return None
