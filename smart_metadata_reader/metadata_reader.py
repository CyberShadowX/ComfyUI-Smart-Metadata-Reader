from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .a1111_parser import parse_a1111_parameters
from .adapters.samplers import extract_sampler_settings
from .graph import GraphIndex
from .lora_extractor import extract_loras, extract_model_name, extract_vae_name
from .models import MetadataBundle, ParseResult, PromptSegment
from .prompt_merge import merge_prompt_segments
from .resolver import ConditioningResolver
from .sampler_selector import select_final_sampler
from .settings_formatter import format_setting


def _metadata_value(info: dict[str, Any], key: str) -> str | None:
    value = info.get(key)
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _source_format(prompt_raw: str | None, workflow_raw: str | None, parameters_raw: str | None) -> str:
    if prompt_raw or workflow_raw:
        return "ComfyUI prompt/workflow"
    if parameters_raw:
        return "A1111 parameters"
    return "none"


def read_metadata(image_path: str | Path) -> MetadataBundle:
    path = Path(image_path)
    with Image.open(path) as image:
        prompt_raw = _metadata_value(image.info, "prompt")
        workflow_raw = _metadata_value(image.info, "workflow")
        parameters_raw = _metadata_value(image.info, "parameters")
        width, height = image.size

    return MetadataBundle(
        filename=path.name,
        width=width,
        height=height,
        prompt_raw=prompt_raw,
        workflow_raw=workflow_raw,
        parameters_raw=parameters_raw,
        source_format=_source_format(prompt_raw, workflow_raw, parameters_raw),
    )


def parse_metadata_bundle(
    bundle: MetadataBundle,
    parameter_index: int = 0,
    prefer_cached_text: bool = True,
    include_raw_json: bool = True,
    max_depth: int = 40,
) -> ParseResult:
    if bundle.prompt_raw:
        try:
            return _parse_comfyui_bundle(
                bundle=bundle,
                prefer_cached_text=prefer_cached_text,
                include_raw_json=include_raw_json,
                max_depth=max_depth,
            )
        except _ComfyGraphStartError as exc:
            if bundle.parameters_raw:
                return _parse_a1111_bundle(
                    bundle=bundle,
                    parameter_index=parameter_index,
                    fallback_reason=str(exc),
                )
            return _failed_result(bundle, "FAILED", str(exc), include_raw_json)

    if bundle.parameters_raw:
        return _parse_a1111_bundle(
            bundle=bundle,
            parameter_index=parameter_index,
            fallback_reason="missing ComfyUI prompt JSON",
        )

    return _failed_result(
        bundle=bundle,
        status_message="NO_METADATA",
        reason="no ComfyUI prompt/workflow or A1111 parameters metadata found",
        include_raw_json=include_raw_json,
    )


class _ComfyGraphStartError(Exception):
    pass


def _parse_comfyui_bundle(
    bundle: MetadataBundle,
    prefer_cached_text: bool,
    include_raw_json: bool,
    max_depth: int,
) -> ParseResult:
    prompt = _load_required_json(bundle.prompt_raw, "prompt")
    workflow, workflow_error = _load_optional_json(bundle.workflow_raw, "workflow")
    graph = GraphIndex(prompt=prompt, workflow=workflow)

    try:
        sampler_selection = select_final_sampler(graph)
    except ValueError as exc:
        raise _ComfyGraphStartError(str(exc)) from exc

    sampler = sampler_selection.sampler
    resolver = ConditioningResolver(
        graph=graph,
        max_depth=max_depth,
        prefer_cached_text=prefer_cached_text,
    )
    positive_segments = resolver.resolve_positive(sampler)
    negative_segments = resolver.resolve_negative(sampler)

    settings = extract_sampler_settings(sampler)
    loras = extract_loras(graph, sampler)
    partial_result = _partial_result(
        sampler_selection=sampler_selection,
        resolver=resolver,
        workflow_error=workflow_error,
    )
    debug_trace = _debug_trace(
        sampler_trace=sampler_selection.debug_trace,
        resolver_trace=resolver.debug_trace,
        loras=loras,
        partial_result=partial_result,
        workflow_error=workflow_error,
    )
    confidence = _confidence(
        selection_confidence=sampler_selection.confidence,
        segments=positive_segments + negative_segments,
        has_unresolved=bool(resolver.unresolved),
        used_fallback_sampler=sampler_selection.selected_by == "fallback",
        has_workflow_error=workflow_error is not None,
        has_llm_template_skip=_has_llm_template_skip(debug_trace),
    )
    status_message = _status_message(
        has_unresolved=bool(resolver.unresolved),
        used_fallback_sampler=sampler_selection.selected_by == "fallback",
        has_workflow_error=workflow_error is not None,
        has_llm_template_skip=_has_llm_template_skip(debug_trace),
    )

    result = ParseResult(
        positive=merge_prompt_segments(positive_segments),
        negative=merge_prompt_segments(negative_segments),
        seed=_int_or_default(settings.get("seed"), -1),
        steps=_int_or_default(settings.get("steps"), 0),
        cfg=_float_or_default(settings.get("cfg"), 0.0),
        width=bundle.width,
        height=bundle.height,
        model_name=extract_model_name(graph, sampler),
        filename=bundle.filename,
        vae_name=extract_vae_name(graph),
        sampler_name=_str_or_empty(settings.get("sampler_name")),
        scheduler=_str_or_empty(settings.get("scheduler")),
        loras=loras,
        raw_prompt_json=bundle.prompt_raw if include_raw_json and bundle.prompt_raw else "",
        raw_workflow_json=bundle.workflow_raw if include_raw_json and bundle.workflow_raw else "",
        partial_result=partial_result,
        debug_trace=debug_trace,
        confidence=confidence,
        status_message=status_message,
    )
    result.setting = format_setting(result)
    return result


def _load_required_json(raw: str | None, label: str) -> dict[str, Any]:
    if raw is None or not raw.strip():
        raise _ComfyGraphStartError(f"missing {label} JSON")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise _ComfyGraphStartError(f"invalid {label} JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise _ComfyGraphStartError(f"{label} JSON is not an object")
    return value


def _load_optional_json(
    raw: str | None,
    label: str,
) -> tuple[dict[str, Any] | None, str | None]:
    if raw is None or not raw.strip():
        return None, None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, f"invalid {label} JSON: {exc}"
    if not isinstance(value, dict):
        return None, f"{label} JSON is not an object"
    return value, None


def _partial_result(
    sampler_selection: Any,
    resolver: ConditioningResolver,
    workflow_error: str | None,
) -> dict[str, Any]:
    partial: dict[str, Any] = {
        "source_format": "ComfyUI prompt/workflow",
        "sampler_selection": {
            "node_id": sampler_selection.sampler.node_id,
            "class_type": sampler_selection.sampler.class_type,
            "selected_by": sampler_selection.selected_by,
            "confidence": sampler_selection.confidence,
        },
    }
    if resolver.unresolved:
        partial["unresolved"] = list(resolver.unresolved)
    if workflow_error:
        partial["workflow_error"] = workflow_error
    return partial


def _debug_trace(
    sampler_trace: str,
    resolver_trace: str,
    loras: list[Any],
    partial_result: dict[str, Any],
    workflow_error: str | None,
) -> str:
    parts = [sampler_trace]
    if workflow_error:
        parts.append(f"WORKFLOW_FALLBACK:\n{workflow_error}")
    if resolver_trace:
        parts.append(resolver_trace)
    if loras:
        parts.append(_lora_trace(loras))
    if partial_result.get("unresolved"):
        parts.append("PARTIAL:\n" + json.dumps(partial_result["unresolved"], indent=2))
    return "\n\n".join(part for part in parts if part)


def _lora_trace(loras: list[Any]) -> str:
    lines = ["LORA_TRACE:"]
    for lora in loras:
        path = " <- ".join(lora.path)
        lines.append(f"{lora.lora_name}: {path}")
    return "\n".join(lines)


def _confidence(
    selection_confidence: float,
    segments: list[PromptSegment],
    has_unresolved: bool,
    used_fallback_sampler: bool,
    has_workflow_error: bool,
    has_llm_template_skip: bool,
) -> float:
    confidence = min(selection_confidence, 0.95)
    if used_fallback_sampler:
        confidence = min(confidence, 0.7)
    if segments:
        confidence = min(confidence, min(segment.confidence for segment in segments))
    if has_workflow_error:
        confidence = min(confidence, 0.85)
    if has_unresolved or has_llm_template_skip:
        confidence = min(confidence, 0.6)
    return max(0.0, round(confidence, 2))


def _status_message(
    has_unresolved: bool,
    used_fallback_sampler: bool,
    has_workflow_error: bool,
    has_llm_template_skip: bool,
) -> str:
    if has_unresolved or used_fallback_sampler or has_workflow_error or has_llm_template_skip:
        return "PARTIAL"
    return "OK"


def _has_llm_template_skip(debug_trace: str) -> bool:
    return "LLM template input skipped" in debug_trace


def _parse_a1111_bundle(
    bundle: MetadataBundle,
    parameter_index: int,
    fallback_reason: str,
) -> ParseResult:
    result = parse_a1111_parameters(bundle.parameters_raw or "", parameter_index)
    result.filename = bundle.filename
    result.width = bundle.width or result.width
    result.height = bundle.height or result.height
    result.partial_result.setdefault("source_format", "A1111 parameters")
    result.partial_result["fallback_reason"] = fallback_reason
    result.debug_trace = (
        "A1111_FALLBACK:\n"
        f"{fallback_reason}\n"
        "Parsed metadata from parameters field."
    )
    result.setting = format_setting(result)
    return result


def _failed_result(
    bundle: MetadataBundle,
    status_message: str,
    reason: str,
    include_raw_json: bool,
) -> ParseResult:
    result = ParseResult(
        width=bundle.width,
        height=bundle.height,
        filename=bundle.filename,
        raw_prompt_json=bundle.prompt_raw if include_raw_json and bundle.prompt_raw else "",
        raw_workflow_json=bundle.workflow_raw if include_raw_json and bundle.workflow_raw else "",
        partial_result={"source_format": bundle.source_format, "reason": reason},
        debug_trace=f"{status_message}:\n{reason}",
        confidence=0.0,
        status_message=status_message,
    )
    result.setting = format_setting(result)
    return result


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float_or_default(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _str_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)
