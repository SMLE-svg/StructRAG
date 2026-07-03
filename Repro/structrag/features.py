from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence
import math

from .graph import GraphData


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def degree_histogram(graph: GraphData, max_degree: int | None = None) -> List[float]:
    degrees = graph.degree()
    if max_degree is None:
        max_degree = max(degrees.values(), default=0)
    histogram = [0.0 for _ in range(max_degree + 1)]
    for degree in degrees.values():
        histogram[min(degree, max_degree)] += 1.0
    total = max(len(graph.nodes), 1)
    return [value / total for value in histogram]


def largest_component_nodes(graph: GraphData) -> List[str]:
    adjacency = graph.adjacency()
    seen: set[str] = set()
    components: List[List[str]] = []
    for node in graph.nodes:
        if node in seen:
            continue
        queue = deque([node])
        seen.add(node)
        component: List[str] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    queue.append(neighbor)
        components.append(component)
    return max(components, key=len) if components else []


def average_shortest_path_length(graph: GraphData) -> float:
    nodes = largest_component_nodes(graph)
    if len(nodes) <= 1:
        return 0.0
    adjacency = graph.adjacency()
    total_distance = 0
    pair_count = 0
    node_set = set(nodes)
    for source in nodes:
        distances = {source: 0}
        queue = deque([source])
        while queue:
            current = queue.popleft()
            for neighbor in adjacency.get(current, set()):
                if neighbor in node_set and neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        for target, distance in distances.items():
            if target > source:
                total_distance += distance
                pair_count += 1
    return total_distance / pair_count if pair_count else 0.0


def clustering_coefficient(graph: GraphData) -> float:
    adjacency = graph.adjacency()
    values: List[float] = []
    for node, neighbors in adjacency.items():
        degree = len(neighbors)
        if degree < 2:
            values.append(0.0)
            continue
        possible = degree * (degree - 1) / 2
        observed = 0
        neighbor_list = sorted(neighbors)
        for i, left in enumerate(neighbor_list):
            for right in neighbor_list[i + 1 :]:
                if right in adjacency.get(left, set()):
                    observed += 1
        values.append(observed / possible)
    return sum(values) / len(values) if values else 0.0


def structural_features(graph: GraphData, max_degree: int = 8) -> List[float]:
    n_nodes = len(graph.nodes)
    n_edges = len(graph.edges)
    degrees = graph.degree()
    leaf_prop = sum(1 for value in degrees.values() if value == 1) / max(n_nodes, 1)
    avg_path = average_shortest_path_length(graph)
    features = [
        n_nodes / 20.0,
        n_edges / 40.0,
        leaf_prop,
        clustering_coefficient(graph),
        avg_path / 20.0,
    ]
    features.extend(degree_histogram(graph, max_degree=max_degree))
    return features


def node_level_features(graph: GraphData, max_degree: int = 8) -> List[float]:
    return [len(graph.nodes) / 20.0] + degree_histogram(graph, max_degree=max_degree)

