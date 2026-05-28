from __future__ import annotations

from typing import Any

from .models import NodeRecord


class GraphIndex:
    def __init__(
        self,
        prompt: dict[str, Any] | None,
        workflow: dict[str, Any] | None,
    ) -> None:
        self._workflow_nodes = self._normalize_workflow_nodes(workflow)
        self._nodes = self._normalize_prompt_nodes(prompt)
        if not self._nodes:
            self._nodes = dict(self._workflow_nodes)
        else:
            self._attach_workflow_supplements()

    def get_node(self, node_id: str | int) -> NodeRecord | None:
        return self._nodes.get(str(node_id))

    def nodes(self) -> list[NodeRecord]:
        return list(self._nodes.values())

    def is_link(self, value: Any) -> bool:
        return self.link_target(value) is not None

    def link_target(self, value: Any) -> tuple[str, int] | None:
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            return None
        node_id, output_index = value
        if not isinstance(node_id, (str, int)):
            return None
        if not isinstance(output_index, int):
            return None
        return str(node_id), output_index

    def upstream_link_inputs(self, node_id: str | int) -> list[tuple[str, Any]]:
        node = self.get_node(node_id)
        if node is None:
            return []
        return [
            (field_name, value)
            for field_name, value in node.inputs.items()
            if self.is_link(value)
        ]

    def downstream_nodes(self, node_id: str | int) -> list[NodeRecord]:
        source_id = str(node_id)
        downstream: list[NodeRecord] = []
        for node in self._nodes.values():
            for value in node.inputs.values():
                target = self.link_target(value)
                if target and target[0] == source_id:
                    downstream.append(node)
                    break
        return downstream

    def workflow_cache_for(
        self,
        node_id: str | int,
        field_names: list[str],
    ) -> str | None:
        cache_entry = self.workflow_cache_entry(node_id, field_names)
        if cache_entry is None:
            return None
        value, _source = cache_entry
        return value

    def workflow_cache_entry(
        self,
        node_id: str | int,
        field_names: list[str],
    ) -> tuple[str, str] | None:
        node = self._workflow_nodes.get(str(node_id))
        if node is None:
            return None

        for field_name in field_names:
            value = node.inputs.get(field_name)
            if isinstance(value, str) and value:
                return value, f"workflow input {field_name}"

        widgets = node.widgets_values
        if isinstance(widgets, dict):
            for field_name in field_names:
                value = widgets.get(field_name)
                if isinstance(value, str) and value:
                    return value, f"workflow widget {field_name}"
        elif isinstance(widgets, list):
            for value in widgets:
                if isinstance(value, str) and value:
                    return value, "workflow widget cache fallback"
        return None

    def workflow_output_name(self, node_id: str | int, output_index: int) -> str | None:
        node = self._workflow_nodes.get(str(node_id))
        if node is None or not isinstance(node.outputs, list):
            return None
        if output_index < 0 or output_index >= len(node.outputs):
            return None
        output = node.outputs[output_index]
        if isinstance(output, str):
            return output
        if not isinstance(output, dict):
            return None
        for field in ("name", "label", "display_name", "displayName"):
            value = output.get(field)
            if isinstance(value, str) and value:
                return value
        return None

    def _normalize_prompt_nodes(
        self,
        prompt: dict[str, Any] | None,
    ) -> dict[str, NodeRecord]:
        if not isinstance(prompt, dict):
            return {}

        nodes: dict[str, NodeRecord] = {}
        for raw_id, raw_node in prompt.items():
            if not isinstance(raw_node, dict):
                continue
            node_id = str(raw_id)
            class_type = str(raw_node.get("class_type") or raw_node.get("type") or "")
            inputs = raw_node.get("inputs")
            nodes[node_id] = NodeRecord(
                node_id=node_id,
                class_type=class_type,
                inputs=dict(inputs) if isinstance(inputs, dict) else {},
                widgets_values=None,
                outputs=raw_node.get("outputs"),
            )
        return nodes

    def _normalize_workflow_nodes(
        self,
        workflow: dict[str, Any] | None,
    ) -> dict[str, NodeRecord]:
        if not isinstance(workflow, dict):
            return {}

        raw_nodes = workflow.get("nodes")
        if not isinstance(raw_nodes, list):
            return {}

        nodes: dict[str, NodeRecord] = {}
        for raw_node in raw_nodes:
            if not isinstance(raw_node, dict):
                continue
            raw_id = raw_node.get("id")
            if raw_id is None:
                continue
            node_id = str(raw_id)
            class_type = str(raw_node.get("class_type") or raw_node.get("type") or "")
            inputs = raw_node.get("inputs")
            nodes[node_id] = NodeRecord(
                node_id=node_id,
                class_type=class_type,
                inputs=dict(inputs) if isinstance(inputs, dict) else {},
                widgets_values=raw_node.get("widgets_values"),
                outputs=raw_node.get("outputs"),
            )
        return nodes

    def _attach_workflow_supplements(self) -> None:
        for node_id, workflow_node in self._workflow_nodes.items():
            prompt_node = self._nodes.get(node_id)
            if prompt_node is None:
                continue
            if prompt_node.widgets_values is None:
                prompt_node.widgets_values = workflow_node.widgets_values
            if prompt_node.outputs is None:
                prompt_node.outputs = workflow_node.outputs
