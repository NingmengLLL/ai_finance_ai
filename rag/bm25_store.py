from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import jieba

from ingestion.schema import Chunk
from rag.query_filters import matches_metadata_filter


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_.%+-]+|[\u4e00-\u9fff]+")
CHINESE_PATTERN = re.compile(r"^[\u4e00-\u9fff]+$")


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in TOKEN_PATTERN.findall(text.lower()):
        if CHINESE_PATTERN.fullmatch(match):
            tokens.extend(token.strip() for token in jieba.cut_for_search(match) if token.strip())
        else:
            tokens.append(match)
    return tokens


class BM25Store:
    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.doc_tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_len = [len(tokens) for tokens in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / len(self.doc_len) if self.doc_len else 0.0
        self.term_freqs = [Counter(tokens) for tokens in self.doc_tokens]
        self.doc_freq: dict[str, int] = defaultdict(int)
        for tokens in self.doc_tokens:
            for token in set(tokens):
                self.doc_freq[token] += 1

    def search(self, query: str, top_k: int = 8, metadata_filter: dict[str, str] | None = None) -> list[tuple[Chunk, float]]:
        if not self.chunks:
            return []
        query_terms = tokenize(query)
        scores: list[tuple[int, float]] = []
        total_docs = len(self.chunks)
        for idx, freqs in enumerate(self.term_freqs):
            if not matches_metadata_filter(self.chunks[idx].metadata, metadata_filter):
                continue
            score = 0.0
            for term in query_terms:
                freq = freqs.get(term, 0)
                if not freq:
                    continue
                df = self.doc_freq.get(term, 0)
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_len[idx] / (self.avgdl or 1.0))
                score += idf * (freq * (self.k1 + 1)) / denom
            if score > 0:
                scores.append((idx, score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return [(self.chunks[idx], score) for idx, score in scores[:top_k]]
