"""Object library helpers for STL and procedural assets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


@dataclass
class ObjectRecord:
    object_id: str
    source: str
    asset_path: Path


class ObjectLibrary:
    """Simple index over normalized object assets."""

    def __init__(self, records: Iterable[ObjectRecord]) -> None:
        self._records: List[ObjectRecord] = list(records)
        self._by_id: Dict[str, ObjectRecord] = {r.object_id: r for r in self._records}

    def all_ids(self) -> List[str]:
        return list(self._by_id.keys())

    def get(self, object_id: str) -> ObjectRecord:
        return self._by_id[object_id]

    @staticmethod
    def from_manifest(manifest_path: Path) -> "ObjectLibrary":
        raise NotImplementedError("Manifest loading not implemented yet.")
