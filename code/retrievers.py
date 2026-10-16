"""Seed-chunk retrievers for the KG2RAG pipeline.

KG2RAG first retrieves a small set of *seed chunks*, then expands and filters
them using the knowledge graph. This module only covers that first stage, so
retrieval variants can be swapped without touching the KG expansion/filter code.

Available retrievers (select with ``--retriever``):

    dense         Dense embedding similarity only (original KG2RAG behaviour).
    bm25          Lexical BM25 only.
    hybrid_alpha  Linear fusion of min-max normalised dense and BM25 scores:
                  score = alpha * dense + (1 - alpha) * bm25.
    rrf           Reciprocal Rank Fusion of dense and BM25 rankings
                  (proposed method -- see ``RRFRetriever``).

``dense`` and ``bm25`` are the alpha=1.0 and alpha=0.0 special cases of
``hybrid_alpha``, so they reproduce earlier runs made with ``--alpha 1.0`` and
``--alpha 0.0``.
"""

import re
from typing import List

from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from rank_bm25 import BM25Okapi

RETRIEVER_CHOICES = ('dense', 'bm25', 'hybrid_alpha', 'rrf')


def tokenize(text: str) -> List[str]:
    """Lower-cased word tokens, used for BM25 indexing and querying."""
    return re.findall(r'\w+', text.lower())


def min_max_normalize(scores) -> List[float]:
    scores = list(scores)
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi == lo:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class _DenseAndBM25Base(BaseRetriever):
    """Holds a dense retriever and a BM25 index over the same candidate chunks."""

    def __init__(self, index, doc_chunks: List[TextNode], top_k: int):
        self.doc_chunks = doc_chunks
        self.top_k = top_k
        # Score every candidate so both signals cover the full pool.
        self.dense_retriever = VectorIndexRetriever(index=index, similarity_top_k=len(doc_chunks))
        self.bm25 = BM25Okapi([tokenize(node.text) for node in doc_chunks])
        super().__init__()

    def dense_scores(self, query_str: str) -> List[float]:
        """Dense similarity for each chunk, aligned with ``self.doc_chunks``."""
        dense_nodes = self.dense_retriever.retrieve(query_str)
        by_id = {n.node.id_: n.score for n in dense_nodes}
        return [by_id.get(node.id_, 0.0) for node in self.doc_chunks]

    def bm25_scores(self, query_str: str) -> List[float]:
        """BM25 score for each chunk, aligned with ``self.doc_chunks``."""
        return list(self.bm25.get_scores(tokenize(query_str)))

    def _top_k(self, fused_scores: List[float]) -> List[NodeWithScore]:
        ranked = sorted(range(len(fused_scores)), key=lambda i: fused_scores[i], reverse=True)
        return [NodeWithScore(node=self.doc_chunks[i], score=fused_scores[i]) for i in ranked[:self.top_k]]


class HybridAlphaRetriever(_DenseAndBM25Base):
    """score = alpha * minmax(dense) + (1 - alpha) * minmax(bm25)."""

    def __init__(self, index, doc_chunks, top_k, alpha: float):
        super().__init__(index, doc_chunks, top_k)
        self.alpha = alpha

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        q = query_bundle.query_str
        dense = min_max_normalize(self.dense_scores(q))
        bm25 = min_max_normalize(self.bm25_scores(q))
        fused = [self.alpha * d + (1 - self.alpha) * s for d, s in zip(dense, bm25)]
        return self._top_k(fused)


class RRFRetriever(_DenseAndBM25Base):
    """Static hybrid retrieval via Reciprocal Rank Fusion (proposed method).

    For each chunk d, with rank_r(d) its 1-based rank under retriever r:

        RRF(d) = sum over r in {dense, bm25} of 1 / (k + rank_r(d))

    ``k`` is the smoothing constant (60 in Cormack et al., 2009).

    TODO: implement ``_retrieve``:
      1. get ``self.dense_scores(q)`` and ``self.bm25_scores(q)``
      2. turn each score list into ranks (highest score -> rank 1)
      3. compute the RRF score per chunk with ``self.k``
      4. ``return self._top_k(rrf_scores)``
    """

    def __init__(self, index, doc_chunks, top_k, k: int = 60):
        super().__init__(index, doc_chunks, top_k)
        self.k = k

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
        raise NotImplementedError('RRFRetriever._retrieve is not implemented yet.')


def build_retriever(name: str, index, doc_chunks, top_k: int, alpha: float = 0.5, rrf_k: int = 60) -> BaseRetriever:
    if name == 'dense':
        return HybridAlphaRetriever(index, doc_chunks, top_k, alpha=1.0)
    if name == 'bm25':
        return HybridAlphaRetriever(index, doc_chunks, top_k, alpha=0.0)
    if name == 'hybrid_alpha':
        return HybridAlphaRetriever(index, doc_chunks, top_k, alpha=alpha)
    if name == 'rrf':
        return RRFRetriever(index, doc_chunks, top_k, k=rrf_k)
    raise ValueError(f'Unknown retriever: {name}. Choose from {RETRIEVER_CHOICES}.')
