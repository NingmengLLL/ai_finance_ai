from __future__ import annotations

from collections import defaultdict


class GraphStore:
    def __init__(self, relations: list[dict] | None = None):
        self.relations = relations or []
        self.adj: dict[str, list[dict]] = defaultdict(list)
        for relation in self.relations:
            self.add_relation(relation)

