from __future__ import annotations

from .base import TextResolutionContext
from ..models import NodeRecord, PromptSegment


class CLIPTextEncodeAdapter:
    def matches(self, node: NodeRecord) -> bool:
        return node.class_type == "CLIPTextEncode"

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


class CLIPTextEncodeSDXLAdapter:
    def matches(self, node: NodeRecord) -> bool:
        return node.class_type == "CLIPTextEncodeSDXL"

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
        for field in ("text", "text_g", "text_l"):
            if field in node.inputs:
                segments.extend(context.resolve_input(node, field, role, path, depth))
        return segments
