from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from .adapters.samplers import is_sampler
from .graph import GraphIndex
from .models import NodeRecord


OUTPUT_INPUT_FIELDS = ("images", "image")
UPSTREAM_FIELDS = (
    "images",
    "image",
    "samples",
    "sample",
    "latent",
    "latent_image",
    "pixels",
)


@dataclass
class SamplerSelection:
    sampler: NodeRecord
    confidence: float
    debug_trace: str
    selected_by: str


def select_final_sampler(graph: GraphIndex) -> SamplerSelection:
    for output_node in _output_nodes(graph):
        selection = _select_from_output_node(graph, output_node)
        if selection is not None:
            return selection

    fallback = _fallback_sampler(graph)
    if fallback is None:
        raise ValueError("No sampler candidate found")

    return SamplerSelection(
        sampler=fallback,
        confidence=0.5,
        debug_trace=(
            "SAMPLER_SELECTION:\n"
            f"sampler selected by fallback heuristic: "
            f"{fallback.class_type} {fallback.node_id}; "
            "no final SaveImage/PreviewImage output chain was found"
        ),
        selected_by="fallback",
    )


def _output_nodes(graph: GraphIndex) -> list[NodeRecord]:
    return [node for node in graph.nodes() if _is_output_image_node(node)]


def _is_output_image_node(node: NodeRecord) -> bool:
    lowered = node.class_type.lower()
    if lowered in {"saveimage", "previewimage", "saveanimatedwebp"}:
        return True
    has_output_image_field = any(field in node.inputs for field in OUTPUT_INPUT_FIELDS)
    return has_output_image_field and (
        ("save" in lowered and ("image" in lowered or "webp" in lowered))
        or ("preview" in lowered and ("image" in lowered or "webp" in lowered))
    )


def _select_from_output_node(
    graph: GraphIndex,
    output_node: NodeRecord,
) -> SamplerSelection | None:
    queue: deque[tuple[NodeRecord, list[str]]] = deque()
    for field in OUTPUT_INPUT_FIELDS:
        target = graph.link_target(output_node.inputs.get(field))
        if target is None:
            continue
        target_node = graph.get_node(target[0])
        if target_node is None:
            continue
        queue.append(
            (
                target_node,
                [f"{output_node.class_type} {output_node.node_id}.{field}"],
            )
        )

    visited: set[str] = set()
    while queue:
        node, chain = queue.popleft()
        if node.node_id in visited:
            continue
        visited.add(node.node_id)

        if is_sampler(node):
            trace_chain = " <- ".join(chain + [f"{node.class_type} {node.node_id}"])
            return SamplerSelection(
                sampler=node,
                confidence=1.0,
                debug_trace=(
                    "SAMPLER_SELECTION:\n"
                    f"{trace_chain}\n"
                    f"Selected {node.class_type} {node.node_id} because it is "
                    f"upstream of final {output_node.class_type}."
                ),
                selected_by="output_chain",
            )

        for field in UPSTREAM_FIELDS:
            target = graph.link_target(node.inputs.get(field))
            if target is None:
                continue
            target_node = graph.get_node(target[0])
            if target_node is None:
                continue
            queue.append(
                (
                    target_node,
                    chain + [f"{node.class_type} {node.node_id}.{field}"],
                )
            )
    return None


def _fallback_sampler(graph: GraphIndex) -> NodeRecord | None:
    candidates = [
        node
        for node in graph.nodes()
        if is_sampler(node) and _has_conditioning_inputs(node)
    ]
    if candidates:
        return candidates[-1]

    samplers = [node for node in graph.nodes() if is_sampler(node)]
    if samplers:
        return samplers[-1]
    return None


def _has_conditioning_inputs(node: NodeRecord) -> bool:
    return "positive" in node.inputs or "negative" in node.inputs
