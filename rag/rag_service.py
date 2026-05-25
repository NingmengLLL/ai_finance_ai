from __future__ import annotations

from rag.citation import citation_text
from rag.hybrid_retriever import HybridRetriever


class RagSummaryService:
    def __init__(self):
        self.retriever = HybridRetriever()

    def retrieve_docs(self, query: str):
        return self.retriever.retrieve_evidence(query)

    def rag_summarize(self, query: str) -> str:
        cards = self.retrieve_docs(query)
        if not cards:
            return "当前资料库未检索到足够证据，无法可靠回答该金融问题。"
        lines = ["基于本地金融资料库，检索到以下证据："]
        for idx, card in enumerate(cards, start=1):
            lines.append(f"{idx}. {card.claim} {citation_text(card)}")
        return "\n".join(lines)


if __name__ == "__main__":
    service = RagSummaryService()
    print(service.rag_summarize("示例科技2024年现金流质量如何？"))
