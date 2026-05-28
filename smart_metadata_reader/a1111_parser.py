from __future__ import annotations

import re

from .models import ParseResult


KNOWN_SETTING_KEYS = (
    "Steps",
    "Sampler",
    "CFG scale",
    "Seed",
    "Size",
    "Model",
    "Model hash",
    "Scheduler",
    "Denoising strength",
)


def parse_a1111_parameters(parameters: str, parameter_index: int = 0) -> ParseResult:
    del parameter_index

    prompt_block, settings_line = _split_prompt_and_settings(parameters)
    positive, negative = _split_positive_negative(prompt_block)
    settings = _parse_settings(settings_line)

    partial_result = {"source_format": "A1111 parameters"}
    if model_hash := settings.get("model_hash"):
        partial_result["model_hash"] = model_hash
    if denoising_strength := settings.get("denoising_strength"):
        partial_result["denoising_strength"] = denoising_strength

    return ParseResult(
        positive=positive,
        negative=negative,
        seed=_int_or_default(settings.get("seed"), -1),
        steps=_int_or_default(settings.get("steps"), 0),
        cfg=_float_or_default(settings.get("cfg"), 0.0),
        width=_int_or_default(settings.get("width"), 0),
        height=_int_or_default(settings.get("height"), 0),
        model_name=settings.get("model_name", ""),
        sampler_name=settings.get("sampler_name", ""),
        scheduler=settings.get("scheduler", ""),
        partial_result=partial_result,
        confidence=0.75,
        status_message="A1111 parameters fallback",
    )


def _split_prompt_and_settings(parameters: str) -> tuple[str, str]:
    normalized = parameters.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return "", ""

    lines = normalized.split("\n")
    for index in range(len(lines) - 1, -1, -1):
        line = lines[index].strip()
        if _looks_like_settings_line(line):
            return "\n".join(lines[:index]).strip(), line
    return normalized, ""


def _looks_like_settings_line(line: str) -> bool:
    if not line:
        return False
    key_count = sum(1 for key in KNOWN_SETTING_KEYS if f"{key}:" in line)
    return key_count >= 1 and ("," in line or line.startswith("Steps:"))


def _split_positive_negative(prompt_block: str) -> tuple[str, str]:
    match = re.search(r"(?im)^Negative prompt:\s*", prompt_block)
    if match is None:
        return prompt_block.strip(), ""
    positive = prompt_block[: match.start()].strip()
    negative = prompt_block[match.end() :].strip()
    return positive, negative


def _parse_settings(settings_line: str) -> dict[str, str]:
    settings: dict[str, str] = {}
    if not settings_line:
        return settings

    settings["steps"] = _extract_value(settings_line, "Steps")
    settings["sampler_name"] = _extract_value(settings_line, "Sampler")
    settings["scheduler"] = _extract_value(settings_line, "Scheduler")
    settings["cfg"] = _extract_value(settings_line, "CFG scale")
    settings["seed"] = _extract_value(settings_line, "Seed")
    settings["model_name"] = _extract_value(settings_line, "Model")
    settings["model_hash"] = _extract_value(settings_line, "Model hash")
    settings["denoising_strength"] = _extract_value(settings_line, "Denoising strength")

    size = _extract_value(settings_line, "Size")
    size_match = re.match(r"\s*(\d+)\s*x\s*(\d+)\s*$", size, flags=re.IGNORECASE)
    if size_match:
        settings["width"] = size_match.group(1)
        settings["height"] = size_match.group(2)

    return {key: value for key, value in settings.items() if value != ""}


def _extract_value(settings_line: str, key: str) -> str:
    known_keys_pattern = "|".join(re.escape(known_key) for known_key in KNOWN_SETTING_KEYS)
    pattern = (
        rf"(?:^|,\s*){re.escape(key)}:\s*"
        rf"(.*?)"
        rf"(?=,\s*(?:{known_keys_pattern}):|$)"
    )
    match = re.search(pattern, settings_line)
    if match is None:
        return ""
    return match.group(1).strip()


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
