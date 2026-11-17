from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..schemas import EngineName


@dataclass(frozen=True)
class EngineContext:
    media_id: str
    media_hash: str
    media_type: str
    path: Path


class Engine(Protocol):
    name: EngineName
    model_version: str | None

    def analyze(self, context: EngineContext) -> list[dict]:
        """Return findings conforming to the shared findings schema."""


REGISTRY: dict[EngineName, Engine] = {}
