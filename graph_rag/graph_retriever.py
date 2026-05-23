from __future__ import annotations

from graph_rag.entity_extractor import extract_entities
from graph_rag.graph_store import GraphStore
from graph_rag.relation_extractor import extract_relations
from ingestion.pipeline import read_chunks


class GraphRetriever:
    def __init__(self):
        chunks = read_chunks()
        self.graph = GraphStore(extract_relations(chunks))


