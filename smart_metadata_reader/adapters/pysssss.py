from __future__ import annotations

from .base import TextResolutionContext
from ..models import NodeRecord, PromptSegment


MISSING_LLM_SHOWTEXT_CACHE_REASON = (
    "ShowText cache missing on final chain; upstream LLM runtime output was not "
    "embedded in metadata. Other ShowText caches outside the final chain are "
    "ignored to avoid prompt pollution."
)


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
        output_index: int | None = None,
    ) -> list[PromptSegment]:
        del output_index
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
        return _is_show_text_node(node)

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
        cached_text = self._cached_text(node, context)
        if cached_text:
            text, source = cached_text
            confidence = 0.75 if source == "workflow widget cache fallback" else 1.0
            context.trace.add(
                "TEXT_TRACE",
                f"{node.class_type} {node.node_id}: using cached ShowText text from {source}",
            )
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

        upstream_llm = self._upstream_llm_runtime_node(node, context)
        if upstream_llm is not None:
            context.trace.add(
                "TEXT_TRACE",
                (
                    f"{upstream_llm.class_type} {upstream_llm.node_id}: low confidence "
                    "LLM template input skipped, no cached generated output found"
                ),
            )
            context.record_unresolved(
                node=node,
                field="text",
                role=role,
                reason=MISSING_LLM_SHOWTEXT_CACHE_REASON,
            )
            return []

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

    def _upstream_llm_runtime_node(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
    ) -> NodeRecord | None:
        target = context.graph.link_target(node.inputs.get("text"))
        if target is None:
            return None
        upstream = context.graph.get_node(target[0])
        if upstream is not None and context.is_llm_template_node(upstream):
            return upstream
        return None


def _is_show_text_node(node: NodeRecord) -> bool:
    lowered = node.class_type.lower().replace("_", "").replace(" ", "")
    return lowered == "showtext" or lowered.startswith("showtext|")
