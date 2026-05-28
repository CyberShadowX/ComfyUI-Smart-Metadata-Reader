from __future__ import annotations

from typing import Protocol

from smart_metadata_reader.graph import GraphIndex
from smart_metadata_reader.models import NodeRecord, PromptSegment
from smart_metadata_reader.trace import TraceCollector


class NodeAdapter(Protocol):
    def matches(self, node: NodeRecord) -> bool:
        ...

    def resolve(
        self,
        node: NodeRecord,
        context: TextResolutionContext,
        role: str,
        path: list[str],
        depth: int,
    ) -> list[PromptSegment]:
        ...


class AdapterRegistry:
    def __init__(self) -> None:
        self._adapters: list[NodeAdapter] = []

    def register(self, adapter: NodeAdapter) -> None:
        self._adapters.append(adapter)

    def adapter_for(self, node: NodeRecord) -> NodeAdapter | None:
        for adapter in self._adapters:
            if adapter.matches(node):
                return adapter
        return None


class TextResolutionContext:
    def __init__(
        self,
        graph: GraphIndex,
        registry: AdapterRegistry | None = None,
        prefer_cached_text: bool = True,
        trace: TraceCollector | None = None,
        max_depth: int = 40,
    ) -> None:
        self.graph = graph
        self.registry = registry or AdapterRegistry()
        self.prefer_cached_text = prefer_cached_text
        self.trace = trace or TraceCollector()
        self.max_depth = max_depth

    def resolve_node(
        self,
        node_id: str | int,
        role: str,
        path: list[str],
        depth: int = 0,
    ) -> list[PromptSegment]:
        if depth > self.max_depth:
            self.trace.add("TEXT_TRACE", f"max depth reached at node {node_id}")
            return []

        node = self.graph.get_node(node_id)
        if node is None:
            self.trace.add("TEXT_TRACE", f"missing node {node_id}")
            return []

        if self.is_llm_template_node(node):
            self.trace.add(
                "TEXT_TRACE",
                (
                    f"{node.class_type} {node.node_id}: low confidence "
                    "LLM template input skipped, no cached generated output found"
                ),
            )
            return []

        adapter = self.registry.adapter_for(node)
        if adapter is None:
            self.trace.add("TEXT_TRACE", f"{node.class_type} {node.node_id}: no text adapter")
            return []
        return adapter.resolve(
            node=node,
            context=self,
            role=role,
            path=path + [f"{node.class_type} {node.node_id}"],
            depth=depth + 1,
        )

    def resolve_input(
        self,
        node: NodeRecord,
        field: str,
        role: str,
        path: list[str],
        depth: int,
    ) -> list[PromptSegment]:
        return self.resolve_value(
            value=node.inputs.get(field),
            role=role,
            path=path + [f"{node.class_type} {node.node_id}.{field}"],
            depth=depth,
            source_node=node,
            field=field,
        )

    def resolve_value(
        self,
        value: object,
        role: str,
        path: list[str],
        depth: int,
        source_node: NodeRecord | None = None,
        field: str | None = None,
    ) -> list[PromptSegment]:
        if depth > self.max_depth:
            self.trace.add("TEXT_TRACE", "max depth reached while resolving value")
            return []

        if isinstance(value, str):
            if not value.strip():
                return []
            return [
                PromptSegment(
                    text=value,
                    node_id=source_node.node_id if source_node else None,
                    class_type=source_node.class_type if source_node else None,
                    field=field,
                    path=path,
                )
            ]

        target = self.graph.link_target(value)
        if target is not None:
            target_id, output_index = target
            return self.resolve_node(
                node_id=target_id,
                role=role,
                path=path + [f"link[{target_id}, {output_index}]"],
                depth=depth + 1,
            )

        if isinstance(value, list):
            segments: list[PromptSegment] = []
            for item in value:
                segments.extend(
                    self.resolve_value(
                        value=item,
                        role=role,
                        path=path,
                        depth=depth + 1,
                        source_node=source_node,
                        field=field,
                    )
                )
            return segments

        return []

    def workflow_cache_for(self, node: NodeRecord, field_names: list[str]) -> str | None:
        return self.graph.workflow_cache_for(node.node_id, field_names)

    def workflow_cache_entry(
        self,
        node: NodeRecord,
        field_names: list[str],
    ) -> tuple[str, str] | None:
        return self.graph.workflow_cache_entry(node.node_id, field_names)

    def is_llm_template_node(self, node: NodeRecord) -> bool:
        lowered = node.class_type.lower()
        markers = ("gemini", "chatgpt", "claude", "llm", "language_model")
        return any(marker in lowered for marker in markers)
