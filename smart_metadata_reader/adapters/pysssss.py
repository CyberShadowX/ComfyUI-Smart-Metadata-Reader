from __future__ import annotations

from smart_metadata_reader.adapters.base import TextResolutionContext
from smart_metadata_reader.models import NodeRecord, PromptSegment


class StringFunctionAdapter:
    def matches(self, node: NodeRecord) -> bool:
        return node.class_type == "StringFunction|pysssss"

    def resolve(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
        role: str,
        path: list[str],
        depth: int,
    ) -> list[PromptSegment]:
        action = str(node.inputs.get("action", "append")).lower()
        fields = ("text_a", "text_b", "text_c")
        if action != "append":
            context.trace.add(
                "TEXT_TRACE",
                f"{node.class_type} {node.node_id}: unsupported action {action}",
            )
        segments: list[PromptSegment] = []
        for field in fields:
            if field in node.inputs:
                segments.extend(context.resolve_input(node, field, role, path, depth))
        return segments


class ShowTextAdapter:
    def matches(self, node: NodeRecord) -> bool:
        return node.class_type == "ShowText|pysssss"

    def resolve(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
        role: str,
        path: list[str],
        depth: int,
    ) -> list[PromptSegment]:
        cached_text = self._cached_text(node, context)
        if cached_text:
            text, source = cached_text
            confidence = 0.75 if source == "workflow widget cache fallback" else 1.0
            if source == "workflow widget cache fallback":
                context.trace.add(
                    "TEXT_TRACE",
                    (
                        f"{node.class_type} {node.node_id}: "
                        "workflow widget cache fallback; lowering confidence"
                    ),
                )
            return [
                PromptSegment(
                    text=text,
                    node_id=node.node_id,
                    class_type=node.class_type,
                    field="text_0",
                    path=path + [f"{node.class_type} {node.node_id}.text_0 cache"],
                    confidence=confidence,
                )
            ]

        context.trace.add(
            "TEXT_TRACE",
            (
                f"{node.class_type} {node.node_id}: no cached display text; "
                "following upstream with low confidence"
            ),
        )
        segments = context.resolve_input(node, "text", role, path, depth)
        for segment in segments:
            segment.confidence = min(segment.confidence, 0.5)
        return segments

    def _cached_text(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
    ) -> tuple[str, str] | None:
        if context.prefer_cached_text:
            direct = node.inputs.get("text_0")
            if isinstance(direct, str) and direct.strip():
                return direct, "prompt input text_0"
            workflow = context.workflow_cache_entry(node, ["text_0", "text"])
            if workflow:
                return workflow
        return None
