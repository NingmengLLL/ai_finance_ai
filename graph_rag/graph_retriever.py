from __future__ import annotations

from graph_rag.entity_extractor import extract_entities
from graph_rag.graph_store import GraphStore
from graph_rag.relation_extractor import extract_relations
from ingestion.pipeline import read_chunks


class GraphRetriever:
    def __init__(self):
        chunks = read_chunks()
        self.graph = GraphStore(extract_relations(chunks))

    def retrieve(self, query: str) -> list[dict]:
        entities = extract_entities(query)
        seeds = entities.get("companies", []) + entities.get("metrics", [])
        relations: list[dict] = []
        for seed in seeds:
            relations.extend(self.graph.neighbors(seed))
        seen = set()
        unique = []
        for relation in relations:
            key = (relation["head"], relation["relation"], relation["tail"], relation.get("chunk_id"))
            if key not in seen:
                unique.append(relation)
                seen.add(key)
        return unique[:12]
