from __future__ import annotations

from typing import Any

from .models import LoraInfo, ParseResult


def format_setting(result: ParseResult) -> str:
    lines = [
        f"Filename: {_text_or_unknown(result.filename)}",
        f"Source: {_source(result)}",
        f"Model: {_text_or_unknown(result.model_name)}",
        f"VAE: {_text_or_unknown(result.vae_name)}",
    ]
    lines.extend(_format_loras(result.loras))
    lines.extend(
        [
            f"Seed: {result.seed}",
            f"Steps: {result.steps}",
            f"CFG: {result.cfg}",
            f"Sampler: {_text_or_unknown(result.sampler_name)}",
            f"Scheduler: {_text_or_unknown(result.scheduler)}",
            f"Size: {_format_size(result.width, result.height)}",
            f"Status: {result.status_message or 'OK'}",
            f"Confidence: {result.confidence:.2f}",
        ]
    )
    unresolved_lines = _format_unresolved(result.partial_result)
    if unresolved_lines:
        lines.append("Unresolved:")
        lines.extend(unresolved_lines)
    reason = result.partial_result.get("reason")
    if isinstance(reason, str) and reason:
        lines.append(f"Failure Reason: {reason}")
    return "\n".join(lines)


def _source(result: ParseResult) -> str:
    source = result.partial_result.get("source_format")
    if isinstance(source, str) and source:
        return source
    if result.raw_prompt_json or result.raw_workflow_json:
        return "ComfyUI prompt/workflow"
    return "unknown"


def _format_loras(loras: list[LoraInfo]) -> list[str]:
    if not loras:
        return ["LoRA: none"]

    lines = ["LoRA:"]
    for lora in loras:
        lines.append(
            "* "
            f"{lora.lora_name} "
            f"(model {_format_strength(lora.strength_model)}, "
            f"clip {_format_strength(lora.strength_clip)})"
        )
    return lines


def _format_strength(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value:.2f}"


def _format_size(width: int, height: int) -> str:
    if width <= 0 or height <= 0:
        return "unknown"
    return f"{width} x {height}"


def _format_unresolved(partial_result: dict[str, Any]) -> list[str]:
    unresolved = partial_result.get("unresolved") or partial_result.get("unresolved_nodes")
    if not isinstance(unresolved, list):
        return []

    lines: list[str] = []
    for entry in unresolved:
        if not isinstance(entry, dict):
            continue
        node_id = _text_or_unknown(str(entry.get("node_id", "")))
        class_type = _text_or_unknown(str(entry.get("class_type", "")))
        field = _text_or_unknown(str(entry.get("field", "")))
        role = _text_or_unknown(str(entry.get("role", "")))
        reason = _text_or_unknown(str(entry.get("reason", "")))
        lines.append(
            f"* Node {node_id} {class_type} input {field} "
            f"role={role} reason={reason}"
        )
    return lines


def _text_or_unknown(value: str) -> str:
    return value if value else "unknown"
