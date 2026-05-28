from __future__ import annotations

import re

from .models import PromptSegment


def _clean_segment(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*,\s*", ", ", cleaned)
    cleaned = re.sub(r"(,\s*)+", ", ", cleaned)
    cleaned = cleaned.strip(" ,\t\r\n")
    return cleaned


def merge_prompt_segments(segments: list[PromptSegment]) -> str:
    cleaned_segments = [
        cleaned
        for segment in segments
        if (cleaned := _clean_segment(segment.text))
    ]
    return _clean_segment(", ".join(cleaned_segments))
