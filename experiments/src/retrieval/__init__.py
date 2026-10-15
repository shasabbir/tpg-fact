"""Dependency-light retrieval primitives and optional archive clients."""

from .hybrid import bm25_rank, reciprocal_rank_fusion

__all__ = ["bm25_rank", "reciprocal_rank_fusion"]

