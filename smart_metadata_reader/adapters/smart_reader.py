from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from .base import TextResolutionContext
from ..models import NodeRecord, PromptSegment


SMART_READER_OUTPUT_FALLBACK = {
    0: "image",
    1: "mask",
    2: "positive",
    3: "negative",
}
PROMPT_OUTPUT_ROLES = {"positive", "negative"}
ALLOWED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


class SmartMetadataReaderAdapter:
    def matches(self, node: NodeRecord) -> bool:
        return node.class_type == "SmartMetadataReader"

    def resolve(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
        role: str,
        path: list[str],
        depth: int,
        output_index: int | None = None,
    ) -> list[PromptSegment]:
        if output_index is None:
            return self._unresolved(
                node,
                context,
                role,
                output_index=-1,
                output_role="unknown",
                referenced_image="",
                reason="SmartMetadataReader output index missing",
            )

        output_role = self._output_role(node, context, output_index)
        image_name = self._image_name(node)
        if output_role not in PROMPT_OUTPUT_ROLES:
            return self._unresolved(
                node,
                context,
                role,
                output_index=output_index,
                output_role=output_role or "unknown",
                referenced_image=image_name or "",
                reason="SmartMetadataReader output is not a prompt text output",
            )

        if role != output_role:
            context.trace.add(
                f"{role.upper()}_TRACE",
                f"SmartMetadataReader output {output_role} used in {role} chain",
            )

        image_path, failure_reason = self._resolve_referenced_image(context, image_name)
        if image_path is None:
            return self._unresolved(
                node,
                context,
                role,
                output_index=output_index,
                output_role=output_role,
                referenced_image=image_name or "",
                reason=failure_reason,
            )

        stack_key = os.path.normcase(str(image_path))
        if stack_key in context.recursion_stack:
            return self._unresolved(
                node,
                context,
                role,
                output_index=output_index,
                output_role=output_role,
                referenced_image=image_name or "",
                reason="recursive SmartMetadataReader image reference",
            )

        try:
            from ..metadata_reader import parse_metadata_bundle, read_metadata

            nested_bundle = read_metadata(image_path)
            nested_result = parse_metadata_bundle(
                nested_bundle,
                parameter_index=context.parameter_index,
                prefer_cached_text=context.prefer_cached_text,
                include_raw_json=context.include_raw_json,
                max_depth=context.parse_max_depth,
                _recursion_stack=context.recursion_stack + (stack_key,),
            )
        except Exception as exc:
            return self._unresolved(
                node,
                context,
                role,
                output_index=output_index,
                output_role=output_role,
                referenced_image=image_name or "",
                reason=f"referenced image not resolvable: {exc}",
            )

        text = nested_result.positive if output_role == "positive" else nested_result.negative
        if not text and nested_result.status_message in {"FAILED", "NO_METADATA", "PARTIAL"}:
            return self._unresolved(
                node,
                context,
                role,
                output_index=output_index,
                output_role=output_role,
                referenced_image=image_name or "",
                reason=(
                    "SmartMetadataReader output is runtime value; "
                    "referenced image did not resolve prompt text"
                ),
            )

        context.trace.add(
            f"{role.upper()}_TRACE",
            (
                f"Resolved SmartMetadataReader {output_role} output "
                f"from referenced image {image_name}"
            ),
        )
        if not text.strip():
            return []
        return [
            PromptSegment(
                text=text,
                node_id=node.node_id,
                class_type=node.class_type,
                field=output_role,
                path=path
                + [
                    f"{node.class_type} {node.node_id}.output[{output_index}] {output_role}",
                    f"referenced image {image_name}",
                ],
                confidence=min(max(nested_result.confidence, 0.75), 0.95),
            )
        ]

    def _output_role(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
        output_index: int,
    ) -> str | None:
        output_name = context.graph.workflow_output_name(node.node_id, output_index)
        role = _role_from_output_name(output_name)
        if role is not None:
            return role
        return SMART_READER_OUTPUT_FALLBACK.get(output_index)

    def _image_name(self, node: NodeRecord) -> str | None:
        value = node.inputs.get("image")
        if isinstance(value, str) and value:
            return value
        widgets = node.widgets_values
        if isinstance(widgets, dict):
            value = widgets.get("image")
            if isinstance(value, str) and value:
                return value
        elif isinstance(widgets, list):
            for value in widgets:
                if isinstance(value, str) and _allowed_image_extension(value):
                    return value
        return None

    def _resolve_referenced_image(
        self,
        context: TextResolutionContext,
        image_name: str | None,
    ) -> tuple[Path | None, str]:
        if not image_name:
            return None, "SmartMetadataReader referenced image filename missing"
        if not context.base_dir:
            return None, "SmartMetadataReader referenced image base_dir unavailable"
        if not _safe_relative_image_path(image_name):
            return None, "unsafe referenced image path"

        base_dir = Path(context.base_dir).resolve()
        candidate = (base_dir / image_name.replace("\\", "/")).resolve()
        try:
            candidate.relative_to(base_dir)
        except ValueError:
            return None, "unsafe referenced image path"
        if not candidate.exists():
            return None, "referenced image not found"
        return candidate, ""

    def _unresolved(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
        role: str,
        output_index: int,
        output_role: str,
        referenced_image: str,
        reason: str,
    ) -> list[PromptSegment]:
        context.record_unresolved(
            node=node,
            field=f"output[{output_index}]",
            role=role,
            reason=reason,
            output_index=output_index,
            resolved_output_role=output_role,
            referenced_image_filename=referenced_image,
        )
        return []


def _role_from_output_name(output_name: str | None) -> str | None:
    if not output_name:
        return None
    normalized = output_name.strip().lower().replace(" ", "_")
    if normalized in PROMPT_OUTPUT_ROLES:
        return normalized
    return None


def _safe_relative_image_path(image_name: str) -> bool:
    normalized = image_name.replace("\\", "/")
    if not _allowed_image_extension(normalized):
        return False
    if Path(normalized).is_absolute() or re.match(r"^[A-Za-z]:", normalized):
        return False
    parts = [part for part in normalized.split("/") if part]
    if not parts or any(part == ".." for part in parts):
        return False
    return True


def _allowed_image_extension(image_name: str) -> bool:
    return Path(image_name).suffix.lower() in ALLOWED_IMAGE_EXTENSIONS
