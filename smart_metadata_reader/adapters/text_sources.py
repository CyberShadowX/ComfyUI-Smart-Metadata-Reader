from __future__ import annotations

from .base import TextResolutionContext
from ..models import NodeRecord, PromptSegment


COMMON_TEXT_FIELDS = (
    "text",
    "text_0",
    "text_a",
    "text_b",
    "text_c",
    "string",
    "value",
    "prompt",
    "result",
)


class DeepTranslatorTextAdapter:
    def matches(self, node: NodeRecord) -> bool:
        return node.class_type == "DeepTranslatorTextNode"

    def resolve(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
        role: str,
        path: list[str],
        depth: int,
        output_index: int | None = None,
    ) -> list[PromptSegment]:
        del output_index
        return context.resolve_input(node, "text", role, path, depth)


class GenericTextAdapter:
    def matches(self, node: NodeRecord) -> bool:
        lowered = node.class_type.lower()
        if context_marker_is_llm(lowered):
            return False
        return (
            "primitive" in lowered
            or "string" in lowered
            or "text" in lowered
            or lowered in {"string", "text"}
        )

    def resolve(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
        role: str,
        path: list[str],
        depth: int,
        output_index: int | None = None,
    ) -> list[PromptSegment]:
        del output_index
        segments: list[PromptSegment] = []
        for field in COMMON_TEXT_FIELDS:
            if field in node.inputs:
                segments.extend(context.resolve_input(node, field, role, path, depth))
        return segments


def context_marker_is_llm(lowered_class_type: str) -> bool:
    markers = ("gemini", "chatgpt", "claude", "llm", "language_model")
    return any(marker in lowered_class_type for marker in markers)
