from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


Edge = Tuple[str, str]


def _as_str_list(values: Iterable[Any]) -> List[str]:
    return [str(value) for value in values]


def canonical_edge(source: str, target: str, directed: bool = False) -> Edge:
    source = str(source)
    target = str(target)
    if directed or source <= target:
        return source, target
    return target, source


@dataclass
class EdgeCandidate:
    source: str
    target: str
    confidence: float = 0.0
    signals: Dict[str, float] = field(default_factory=dict)
    reason: str = ""

    @property
    def edge(self) -> Edge:
        return canonical_edge(self.source, self.target, directed=False)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EdgeCandidate":
        return cls(
            source=str(payload["source"]),
            target=str(payload["target"]),
            confidence=float(payload.get("confidence", 0.0)),
            signals={str(k): float(v) for k, v in payload.get("signals", {}).items()},
            reason=str(payload.get("reason", "")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "confidence": self.confidence,
            "signals": self.signals,
            "reason": self.reason,
        }


@dataclass
class GraphData:
    nodes: List[str]
    edges: List[Edge]
    directed: bool = False
    name: str = ""
    topology: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.nodes = sorted(set(_as_str_list(self.nodes)))
        self.edges = sorted(
            {
                canonical_edge(source, target, self.directed)
                for source, target in self.edges
                if str(source) != str(target)
            }
        )

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "GraphData":
        return cls(
            nodes=_as_str_list(payload.get("nodes", [])),
            edges=[tuple(map(str, edge)) for edge in payload.get("edges", [])],
            directed=bool(payload.get("directed", False)),
            name=str(payload.get("name", "")),
            topology=str(payload.get("topology", "")),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        output = {
            "nodes": self.nodes,
            "edges": [[source, target] for source, target in self.edges],
            "directed": self.directed,
        }
        if self.name:
            output["name"] = self.name
        if self.topology:
            output["topology"] = self.topology
        if self.metadata:
            output["metadata"] = self.metadata
        return output

    def edge_set(self) -> set[Edge]:
        return set(self.edges)

    def degree(self) -> Dict[str, int]:
        degree = {node: 0 for node in self.nodes}
        for source, target in self.edges:
            degree[source] = degree.get(source, 0) + 1
            degree[target] = degree.get(target, 0) + 1
        return degree

    def adjacency(self) -> Dict[str, set[str]]:
        adjacency = {node: set() for node in self.nodes}
        for source, target in self.edges:
            adjacency.setdefault(source, set()).add(target)
            adjacency.setdefault(target, set()).add(source)
        return adjacency

    def with_edges(self, extra_edges: Iterable[Edge]) -> "GraphData":
        return GraphData(
            nodes=self.nodes,
            edges=list(self.edge_set().union({canonical_edge(a, b, self.directed) for a, b in extra_edges})),
            directed=self.directed,
            name=self.name,
            topology=self.topology,
            metadata=self.metadata,
        )


def load_structrag_input(path: str | Path) -> tuple[GraphData, List[EdgeCandidate]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    graph = GraphData.from_dict(payload["graph"])
    uncertain_edges = [EdgeCandidate.from_dict(item) for item in payload.get("uncertain_edges", [])]
    return graph, uncertain_edges


def load_graph(path: str | Path) -> GraphData:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "graph" in payload:
        payload = payload["graph"]
    return GraphData.from_dict(payload)


def load_templates(path: str | Path) -> List[GraphData]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("templates", [])
    return [GraphData.from_dict(item) for item in payload]


def save_graph(path: str | Path, graph: GraphData) -> None:
    Path(path).write_text(json.dumps(graph.to_dict(), indent=2), encoding="utf-8")


def edge_difference(template: GraphData, graph: GraphData) -> set[Edge]:
    return template.edge_set().difference(graph.edge_set())

