from __future__ import annotations

from ..models import NodeRecord


ROLE_FIELD_ORDER = {
    "positive": ("positive", "positive_conditioning", "base_positive"),
    "negative": ("negative", "negative_conditioning", "base_negative"),
}

KNOWN_CONDITIONING_MARKERS = (
    "controlnetapply",
    "inpaintmodelconditioning",
    "conditioningcombine",
    "conditioningconcat",
    "conditioningset",
    "facedetailer",
    "detailerforeach",
    "detailer",
    "impact",
)


class ConditioningAdapter:
    def matches(self, node: NodeRecord) -> bool:
        lowered = node.class_type.lower()
        return any(marker in lowered for marker in KNOWN_CONDITIONING_MARKERS)

    def fields_for_role(self, node: NodeRecord, role: str) -> list[str]:
        role_fields = [
            field
            for field in ROLE_FIELD_ORDER.get(role, ())
            if field in node.inputs
        ]
        if role_fields:
            return role_fields

        conditioning_fields = [
            field
            for field in node.inputs
            if _is_conditioning_field(field)
        ]
        if conditioning_fields:
            return conditioning_fields

        return [
            field
            for field in node.inputs
            if field in {"positive", "negative", "base_positive", "base_negative"}
        ]


def _is_conditioning_field(field: str) -> bool:
    lowered = field.lower()
    return (
        lowered == "conditioning"
        or "conditioning" in lowered
        or lowered in {"positive", "negative", "base_positive", "base_negative"}
    )
