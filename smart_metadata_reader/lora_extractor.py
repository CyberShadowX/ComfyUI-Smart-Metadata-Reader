from __future__ import annotations

from collections import deque

from .adapters.loaders import (
    checkpoint_name,
    is_checkpoint_loader,
    is_lora_loader,
    is_vae_loader,
    lora_name,
    strength_clip,
    strength_model,
    vae_name,
)
from .adapters.samplers import is_sampler
from .graph import GraphIndex
from .models import LoraInfo, NodeRecord


def extract_model_name(graph: GraphIndex, sampler: NodeRecord) -> str:
    for node, _path in _walk_upstream(
        graph=graph,
        start_node=sampler,
        start_fields=("model",),
    ):
        if is_checkpoint_loader(node):
            return checkpoint_name(node)
    return ""


def extract_vae_name(graph: GraphIndex) -> str:
    output_chain_name = _extract_vae_name_from_output_chain(graph)
    if output_chain_name:
        return output_chain_name

    for node in graph.nodes():
        if _is_vae_decode(node):
            name = _vae_name_from_decode_node(graph, node)
            if name:
                return name

    for node in graph.nodes():
        if is_vae_loader(node):
            name = vae_name(node)
            if name:
                return name
    return ""


def extract_loras(graph: GraphIndex, sampler: NodeRecord) -> list[LoraInfo]:
    found: dict[str, LoraInfo] = {}

    for field in ("model", "clip"):
        for node, path in _walk_upstream(
            graph=graph,
            start_node=sampler,
            start_fields=(field,),
        ):
            if not is_lora_loader(node):
                continue
            name = lora_name(node)
            if not name:
                continue
            existing = found.get(node.node_id)
            full_path = _extend_path_to_source_loader(graph, node, path)
            lora = LoraInfo(
                node_id=node.node_id,
                class_type=node.class_type,
                lora_name=name,
                strength_model=strength_model(node),
                strength_clip=strength_clip(node),
                path=full_path,
            )
            if existing is None or len(lora.path) < len(existing.path):
                found[node.node_id] = lora

    loras = list(found.values())
    loras.sort(key=_checkpoint_to_sampler_sort_key)
    return loras


def _walk_upstream(
    graph: GraphIndex,
    start_node: NodeRecord,
    start_fields: tuple[str, ...],
) -> list[tuple[NodeRecord, list[str]]]:
    discovered: list[tuple[NodeRecord, list[str]]] = []
    visited: set[tuple[str, str]] = set()

    def visit_from_node(node: NodeRecord, fields: tuple[str, ...], path: list[str]) -> None:
        for field in fields:
            value = node.inputs.get(field)
            target = graph.link_target(value)
            if target is None:
                continue
            linked_node = graph.get_node(target[0])
            if linked_node is None:
                continue

            key = (linked_node.node_id, field)
            if key in visited:
                continue
            visited.add(key)

            linked_path = path + [f"{linked_node.class_type} {linked_node.node_id}"]
            discovered.append((linked_node, linked_path))
            visit_from_node(linked_node, _upstream_fields_for(linked_node), linked_path)

    visit_from_node(
        start_node,
        start_fields,
        [f"{start_node.class_type} {start_node.node_id}.{start_fields[0]}"],
    )
    return discovered


def _upstream_fields_for(node: NodeRecord) -> tuple[str, ...]:
    fields = []
    for field in ("model", "clip", "vae"):
        if field in node.inputs:
            fields.append(field)
    return tuple(fields)


def _extend_path_to_source_loader(
    graph: GraphIndex,
    node: NodeRecord,
    path: list[str],
) -> list[str]:
    extended = list(path)
    visited = {node.node_id}
    current = node

    while True:
        next_node = _first_upstream_node(graph, current)
        if next_node is None or next_node.node_id in visited:
            return extended
        visited.add(next_node.node_id)
        extended.append(f"{next_node.class_type} {next_node.node_id}")
        if is_checkpoint_loader(next_node) or is_vae_loader(next_node):
            return extended
        current = next_node


def _first_upstream_node(graph: GraphIndex, node: NodeRecord) -> NodeRecord | None:
    for field in ("model", "clip", "vae"):
        target = graph.link_target(node.inputs.get(field))
        if target is None:
            continue
        linked_node = graph.get_node(target[0])
        if linked_node is not None:
            return linked_node
    return None


def _checkpoint_to_sampler_sort_key(info: LoraInfo) -> tuple[int, str]:
    marker = f"{info.class_type} {info.node_id}"
    try:
        index = info.path.index(marker)
    except ValueError:
        index = 0
    # Paths are sampler -> source. Larger index means closer to checkpoint.
    return (-index, info.node_id)


def _is_vae_decode(node: NodeRecord) -> bool:
    lowered = node.class_type.lower().replace(" ", "")
    return "vaedecode" in lowered


def _extract_vae_name_from_output_chain(graph: GraphIndex) -> str:
    for output_node in graph.nodes():
        if not _is_image_output_node(output_node):
            continue
        name = _walk_output_chain_for_vae(graph, output_node)
        if name:
            return name
    return ""


def _is_image_output_node(node: NodeRecord) -> bool:
    lowered = node.class_type.lower()
    if lowered in {"saveimage", "previewimage", "saveanimatedwebp"}:
        return True
    has_image_input = "images" in node.inputs or "image" in node.inputs
    return has_image_input and (
        ("save" in lowered and ("image" in lowered or "webp" in lowered))
        or ("preview" in lowered and ("image" in lowered or "webp" in lowered))
    )


def _walk_output_chain_for_vae(graph: GraphIndex, output_node: NodeRecord) -> str:
    queue: deque[NodeRecord] = deque()
    for field in ("images", "image"):
        target = graph.link_target(output_node.inputs.get(field))
        if target is None:
            continue
        node = graph.get_node(target[0])
        if node is not None:
            queue.append(node)

    visited: set[str] = set()
    while queue:
        node = queue.popleft()
        if node.node_id in visited:
            continue
        visited.add(node.node_id)

        if is_sampler(node):
            name = _vae_name_from_node_input(graph, node)
            if name:
                return name

        if _is_vae_decode(node):
            name = _vae_name_from_decode_node(graph, node)
            if name:
                return name

        for field in ("images", "image", "samples", "sample", "latent", "latent_image", "pixels"):
            target = graph.link_target(node.inputs.get(field))
            if target is None:
                continue
            linked_node = graph.get_node(target[0])
            if linked_node is not None:
                queue.append(linked_node)
    return ""


def _vae_name_from_decode_node(graph: GraphIndex, decode_node: NodeRecord) -> str:
    return _vae_name_from_node_input(graph, decode_node)


def _vae_name_from_node_input(graph: GraphIndex, node: NodeRecord) -> str:
    target = graph.link_target(node.inputs.get("vae"))
    if target is None:
        return ""
    vae_node = graph.get_node(target[0])
    if vae_node and is_vae_loader(vae_node):
        return vae_name(vae_node)
    return ""
