from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class MetadataBundle:
    filename: str
    width: int
    height: int
    prompt_raw: str | None
    workflow_raw: str | None
    parameters_raw: str | None
    source_format: str
    user_comment_raw: str | None = None
    image_description_raw: str | None = None
    software_raw: str | None = None


@dataclass
class NodeRecord:
    node_id: str
    class_type: str
    inputs: dict[str, Any]
    widgets_values: list[Any] | dict[str, Any] | None = None


@dataclass
class PromptSegment:
    text: str
    node_id: str | None
    class_type: str | None
    field: str | None
    path: list[str]
    confidence: float = 1.0


@dataclass
class LoraInfo:
    node_id: str
    class_type: str
    lora_name: str
    strength_model: float | None
    strength_clip: float | None
    path: list[str]


@dataclass
class ParseResult:
    positive: str = ""
    negative: str = ""
    seed: int = -1
    steps: int = 0
    cfg: float = 0.0
    width: int = 0
    height: int = 0
    model_name: str = ""
    filename: str = ""
    setting: str = ""
    vae_name: str = ""
    sampler_name: str = ""
    scheduler: str = ""
    loras: list[LoraInfo] = field(default_factory=list)
    raw_prompt_json: str = ""
    raw_workflow_json: str = ""
    partial_result: dict[str, Any] = field(default_factory=dict)
    debug_trace: str = ""
    confidence: float = 0.0
    status_message: str = ""
