from __future__ import annotations

from collections import defaultdict


class GraphStore:
    def __init__(self, relations: list[dict] | None = None):
        self.relations = relations or []
        self.adj: dict[str, list[dict]] = defaultdict(list)
        for relation in self.relations:
            self.add_relation(relation)

    def add_relation(self, relation: dict) -> None:
        self.adj[relation["head"]].append(relation)
        reverse = {
            **relation,
            "head": relation["tail"],
            "tail": relation["head"],
            "relation": f"反向:{relation['relation']}",
        }
        self.adj[relation["tail"]].append(reverse)

    def neighbors(self, entity: str, limit: int = 8) -> list[dict]:
        return self.adj.get(entity, [])[:limit]
