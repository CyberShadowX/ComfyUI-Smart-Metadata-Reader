from __future__ import annotations

from collections import OrderedDict


class TraceCollector:
    def __init__(self) -> None:
        self._sections: OrderedDict[str, list[str]] = OrderedDict()

    def add(self, section: str, line: str) -> None:
        self._sections.setdefault(section, []).append(line)

    def render(self) -> str:
        rendered_sections: list[str] = []
        for section, lines in self._sections.items():
            rendered_sections.append(f"{section}:\n" + "\n".join(lines))
        return "\n\n".join(rendered_sections)
