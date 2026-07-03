from __future__ import annotations

from dataclasses import dataclass

from .graph import GraphData


@dataclass
class EvaluationResult:
    accuracy: float
    edge_precision: float
    edge_recall: float
    edge_f1: float
    true_positive_edges: int
    false_positive_edges: int
    false_negative_edges: int


def evaluate_graph(predicted: GraphData, gold: GraphData) -> EvaluationResult:
    predicted_edges = predicted.edge_set()
    gold_edges = gold.edge_set()
    tp = len(predicted_edges.intersection(gold_edges))
    fp = len(predicted_edges.difference(gold_edges))
    fn = len(gold_edges.difference(predicted_edges))
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = 1.0 if predicted.nodes == gold.nodes and predicted_edges == gold_edges else 0.0
    return EvaluationResult(
        accuracy=accuracy,
        edge_precision=precision,
        edge_recall=recall,
        edge_f1=f1,
        true_positive_edges=tp,
        false_positive_edges=fp,
        false_negative_edges=fn,
    )

