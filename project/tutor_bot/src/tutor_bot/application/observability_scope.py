from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID


@dataclass
class ObservabilityScope:
    trace_id: str
    observation_id: UUID
    payload: dict[str, Any] = field(default_factory=dict)

    def add_metadata(self, **metadata: Any) -> None:
        self.payload.update(metadata)

    def set_output(self, output: Any) -> None:
        self.payload["output"] = output
