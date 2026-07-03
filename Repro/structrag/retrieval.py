from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, Iterable, List, Tuple

from .features import cosine_similarity, structural_features
from .graph import Edge, GraphData, canonical_edge


@dataclass
class RetrievalResult:
    template: GraphData
    coarse_candidates: List[Tuple[str, float]]
    normalized_ged: float
    delta_edges: List[Edge]


def _degree_rank_mapping(source: GraphData, target: GraphData) -> Dict[str, str]:
    source_degrees = source.degree()
    target_degrees = target.degree()
    source_nodes = sorted(source.nodes, key=lambda node: (-source_degrees[node], node))
    target_nodes = sorted(target.nodes, key=lambda node: (-target_degrees[node], node))
    return {node: target_nodes[index] for index, node in enumerate(source_nodes[: len(target_nodes)])}


def _mapped_edges(graph: GraphData, mapping: Dict[str, str]) -> set[Edge]:
    output: set[Edge] = set()
    for source, target in graph.edges:
        if source in mapping and target in mapping:
            output.add(canonical_edge(mapping[source], mapping[target], directed=False))
    return output


def _edit_cost_with_mapping(source: GraphData, target: GraphData, mapping: Dict[str, str]) -> float:
    source_degrees = source.degree()
    target_degrees = target.degree()
    mapped_edges = _mapped_edges(source, mapping)
    target_edges = target.edge_set()

    node_cost = abs(len(source.nodes) - len(target.nodes))
    edge_cost = len(mapped_edges.symmetric_difference(target_edges))
    substitution_cost = 0.0
    for source_node, target_node in mapping.items():
        substitution_cost += abs(source_degrees[source_node] - target_degrees[target_node]) / max(
            max(source_degrees.values(), default=1),
            max(target_degrees.values(), default=1),
            1,
        )

    source_hub_degree = max(source_degrees.values(), default=0)
    target_hub_degree = max(target_degrees.values(), default=0)
    hub_penalty = 0.0
    if abs(source_hub_degree - target_hub_degree) > 1:
        hub_penalty = 2.0

    return node_cost + edge_cost + substitution_cost + hub_penalty


def approximate_ged(source: GraphData, target: GraphData) -> float:
    if len(source.nodes) <= 8 and len(target.nodes) <= 8 and len(source.nodes) == len(target.nodes):
        best = float("inf")
        for ordering in permutations(target.nodes):
            mapping = dict(zip(source.nodes, ordering))
            best = min(best, _edit_cost_with_mapping(source, target, mapping))
        return best

    mapping = _degree_rank_mapping(source, target)
    return _edit_cost_with_mapping(source, target, mapping)


def normalized_graph_edit_distance(source: GraphData, target: GraphData) -> float:
    denominator = len(source.nodes) + len(target.nodes) + len(source.edges) + len(target.edges)
    if denominator == 0:
        return 0.0
    return approximate_ged(source, target) / denominator


def retrieve_template(graph: GraphData, templates: Iterable[GraphData], top_k: int = 5) -> RetrievalResult:
    template_list = list(templates)
    if not template_list:
        raise ValueError("Template library is empty.")

    query_features = structural_features(graph)
    scored = [
        (template, cosine_similarity(query_features, structural_features(template)))
        for template in template_list
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    coarse = scored[:top_k]

    best_template = coarse[0][0]
    best_distance = float("inf")
    for template, _score in coarse:
        distance = normalized_graph_edit_distance(graph, template)
        if distance < best_distance:
            best_distance = distance
            best_template = template

    delta_edges = sorted(best_template.edge_set().difference(graph.edge_set()))
    return RetrievalResult(
        template=best_template,
        coarse_candidates=[(template.name or template.topology or "template", score) for template, score in coarse],
        normalized_ged=best_distance,
        delta_edges=delta_edges,
    )

