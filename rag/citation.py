from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class EvidenceCard:
    claim: str
    evidence: str
    source_file: str
    page_number: int | None
    chunk_id: str
    score: float = 0.0
    metric: str | None = None
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def citation_label(self) -> str:
        page = f"第{self.page_number}页" if self.page_number else "页码未知"
        return f"{self.source_file}，{page}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvidenceCard":
        return cls(**data)


def citation_text(card: EvidenceCard | dict[str, Any]) -> str:
    if isinstance(card, dict):
        card = EvidenceCard.from_dict(card)
    return f"【来源：{card.citation_label()}】"


def has_valid_citation(card: EvidenceCard | dict[str, Any]) -> bool:
    if isinstance(card, dict):
        card = EvidenceCard.from_dict(card)
    return bool(card.source_file and card.chunk_id and card.page_number is not None)
