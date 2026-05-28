from __future__ import annotations

from typing import Any

from .adapters.base import AdapterRegistry, TextResolutionContext
from .adapters.clip_text import CLIPTextEncodeAdapter, CLIPTextEncodeSDXLAdapter
from .adapters.conditioning import ConditioningAdapter
from .adapters.pysssss import ShowTextAdapter, StringFunctionAdapter
from .adapters.text_sources import DeepTranslatorTextAdapter, GenericTextAdapter
from .graph import GraphIndex
from .models import NodeRecord, PromptSegment
from .trace import TraceCollector


class ConditioningResolver:
    def __init__(
        self,
        graph: GraphIndex,
        adapters: AdapterRegistry | None = None,
        max_depth: int = 40,
        prefer_cached_text: bool = True,
    ) -> None:
        self.graph = graph
        self.max_depth = max_depth
        self.prefer_cached_text = prefer_cached_text
        self.trace = TraceCollector()
        self.unresolved: list[dict[str, Any]] = []
        self._active: set[tuple[str, str, str]] = set()
        self._text_registry = adapters or self._default_text_registry()
        self._conditioning_adapter = ConditioningAdapter()

    @property
    def debug_trace(self) -> str:
        return self.trace.render()

    @property
    def partial_result(self) -> dict[str, Any]:
        return {"unresolved": list(self.unresolved)}

    def resolve_positive(self, sampler: NodeRecord) -> list[PromptSegment]:
        return self._resolve_sampler_field(sampler, "positive", "positive")

    def resolve_negative(self, sampler: NodeRecord) -> list[PromptSegment]:
        return self._resolve_sampler_field(sampler, "negative", "negative")

    def resolve_value(
        self,
        value: Any,
        role: str,
        path: list[str],
        depth: int,
    ) -> list[PromptSegment]:
        return self._resolve_value(
            value=value,
            role=role,
            path=path,
            depth=depth,
            field=None,
        )

    def _resolve_sampler_field(
        self,
        sampler: NodeRecord,
        field: str,
        role: str,
    ) -> list[PromptSegment]:
        section = f"{role.upper()}_TRACE"
        self.trace.add(section, f"{sampler.class_type} {sampler.node_id}.{field}")
        if field not in sampler.inputs:
            self._record_unresolved(
                node=sampler,
                field=field,
                role=role,
                reason="missing sampler conditioning input",
            )
            return []
        return self._resolve_value(
            value=sampler.inputs[field],
            role=role,
            path=[f"{sampler.class_type} {sampler.node_id}.{field}"],
            depth=0,
            field=field,
        )

    def _resolve_value(
        self,
        value: Any,
        role: str,
        path: list[str],
        depth: int,
        field: str | None,
    ) -> list[PromptSegment]:
        if depth > self.max_depth:
            self._record_unresolved_by_values(
                node_id="",
                class_type="",
                field=field or "",
                role=role,
                reason="max_depth",
            )
            return []

        if isinstance(value, str):
            if not value.strip():
                return []
            return [
                PromptSegment(
                    text=value,
                    node_id=None,
                    class_type=None,
                    field=field,
                    path=path,
                )
            ]

        target = self.graph.link_target(value)
        if target is not None:
            node_id, output_index = target
            return self._resolve_node(
                node_id=node_id,
                role=role,
                path=path + [f"link[{node_id}, {output_index}]"],
                depth=depth + 1,
                field=field or "",
            )

        return []

    def _resolve_node(
        self,
        node_id: str,
        role: str,
        path: list[str],
        depth: int,
        field: str,
    ) -> list[PromptSegment]:
        node = self.graph.get_node(node_id)
        if node is None:
            self._record_unresolved_by_values(
                node_id=node_id,
                class_type="",
                field=field,
                role=role,
                reason="missing node",
            )
            return []

        section = f"{role.upper()}_TRACE"
        self.trace.add(section, f" -> {node.class_type} {node.node_id}")

        active_key = (node.node_id, field, role)
        if active_key in self._active:
            self._record_unresolved(
                node=node,
                field=field,
                role=role,
                reason="cycle",
            )
            return []

        if depth > self.max_depth:
            self._record_unresolved(
                node=node,
                field=field,
                role=role,
                reason="max_depth",
            )
            return []

        self._active.add(active_key)
        try:
            text_adapter = self._text_registry.adapter_for(node)
            if text_adapter is not None:
                text_context = TextResolutionContext(
                    graph=self.graph,
                    registry=self._text_registry,
                    prefer_cached_text=self.prefer_cached_text,
                    trace=self.trace,
                    max_depth=self.max_depth - depth,
                )
                return text_context.resolve_node(
                    node_id=node.node_id,
                    role=role,
                    path=path,
                    depth=0,
                )

            if self._conditioning_adapter.matches(node):
                return self._resolve_conditioning_node(node, role, path, depth)

            problem_field = self._first_linked_field(node) or field
            self._record_unresolved(
                node=node,
                field=problem_field,
                role=role,
                reason="no conditioning adapter matched",
            )
            return []
        finally:
            self._active.remove(active_key)

    def _resolve_conditioning_node(
        self,
        node: NodeRecord,
        role: str,
        path: list[str],
        depth: int,
    ) -> list[PromptSegment]:
        fields = self._conditioning_adapter.fields_for_role(node, role)
        linked_fields = [
            field
            for field in fields
            if self.graph.is_link(node.inputs.get(field))
        ]
        if not linked_fields:
            self._record_unresolved(
                node=node,
                field=fields[0] if fields else self._first_linked_field(node) or "",
                role=role,
                reason="no linked conditioning input",
            )
            return []

        segments: list[PromptSegment] = []
        for field in linked_fields:
            segments.extend(
                self._resolve_value(
                    value=node.inputs[field],
                    role=role,
                    path=path + [f"{node.class_type} {node.node_id}.{field}"],
                    depth=depth + 1,
                    field=field,
                )
            )
        return segments

    def _first_linked_field(self, node: NodeRecord) -> str | None:
        linked_inputs = self.graph.upstream_link_inputs(node.node_id)
        if not linked_inputs:
            return None
        field, _value = linked_inputs[0]
        return field

    def _record_unresolved(
        self,
        node: NodeRecord,
        field: str,
        role: str,
        reason: str,
    ) -> None:
        self._record_unresolved_by_values(
            node_id=node.node_id,
            class_type=node.class_type,
            field=field,
            role=role,
            reason=reason,
        )

    def _record_unresolved_by_values(
        self,
        node_id: str,
        class_type: str,
        field: str,
        role: str,
        reason: str,
    ) -> None:
        self.unresolved.append(
            {
                "node_id": node_id,
                "class_type": class_type,
                "field": field,
                "role": role,
                "reason": reason,
            }
        )
        self.trace.add(
            f"{role.upper()}_TRACE",
            (
                f"UNRESOLVED {class_type} {node_id} "
                f"field={field} role={role} reason={reason}"
            ),
        )

    def _default_text_registry(self) -> AdapterRegistry:
        registry = AdapterRegistry()
        registry.register(CLIPTextEncodeAdapter())
        registry.register(CLIPTextEncodeSDXLAdapter())
        registry.register(StringFunctionAdapter())
        registry.register(ShowTextAdapter())
        registry.register(DeepTranslatorTextAdapter())
        registry.register(GenericTextAdapter())
        return registry
