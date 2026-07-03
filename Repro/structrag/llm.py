from __future__ import annotations

from collections import Counter
import json
import os
from typing import Iterable, List

from .graph import Edge, EdgeCandidate, GraphData, canonical_edge


def parse_graph_response(text: str, directed: bool = False) -> GraphData:
    payload = json.loads(text)
    return GraphData(
        nodes=[str(node) for node in payload.get("nodes", [])],
        edges=[tuple(edge) for edge in payload.get("edges", [])],
        directed=directed,
        metadata={
            "status": payload.get("status", ""),
            "added_edges": payload.get("added_edges", []),
            "justification": payload.get("justification", ""),
        },
    )


def deterministic_validation(
    graph: GraphData,
    uncertain_edges: Iterable[EdgeCandidate],
    delta_edges: Iterable[Edge],
    threshold: float = 0.40,
    allow_structural_only: bool = False,
) -> GraphData:
    uncertain = {candidate.edge: candidate for candidate in uncertain_edges}
    accepted: List[Edge] = []
    for source, target in delta_edges:
        edge = canonical_edge(source, target, directed=graph.directed)
        candidate = uncertain.get(edge)
        if candidate and candidate.confidence >= threshold:
            accepted.append(edge)
        elif allow_structural_only:
            accepted.append(edge)
    corrected = graph.with_edges(accepted)
    corrected.metadata["status"] = "error" if accepted else "ok"
    corrected.metadata["added_edges"] = accepted
    corrected.metadata["justification"] = "Dry-run structural validation."
    return corrected


def aggregate_graphs(graphs: List[GraphData]) -> GraphData:
    if not graphs:
        raise ValueError("No graphs to aggregate.")

    serialized = [json.dumps(graph.to_dict(), sort_keys=True) for graph in graphs]
    graph_text, count = Counter(serialized).most_common(1)[0]
    if count >= 2:
        return GraphData.from_dict(json.loads(graph_text))

    nodes = sorted(set().union(*(set(graph.nodes) for graph in graphs)))
    edge_counter: Counter[Edge] = Counter()
    for graph in graphs:
        edge_counter.update(graph.edge_set())
    retained_edges = [edge for edge, value in edge_counter.items() if value >= 2]
    return GraphData(nodes=nodes, edges=retained_edges, directed=graphs[0].directed)


def call_openai_graph_validator(
    prompt: str,
    model: str = "gpt-4-0613",
    n_completions: int = 3,
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 800,
) -> List[str]:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Use --dry-run to avoid API calls.")

    from openai import OpenAI

    client = OpenAI()
    outputs: List[str] = []
    for _ in range(n_completions):
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a structural graph validation assistant. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs.append(response.choices[0].message.content or "{}")
    return outputs

