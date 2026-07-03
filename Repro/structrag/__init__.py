"""Lightweight StructRAG reproduction package."""

from .graph import EdgeCandidate, GraphData
from .retrieval import retrieve_template
from .evaluation import evaluate_graph

__all__ = ["EdgeCandidate", "GraphData", "retrieve_template", "evaluate_graph"]

