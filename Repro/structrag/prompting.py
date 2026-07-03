from __future__ import annotations

import json
from typing import Iterable, List

from .graph import Edge, EdgeCandidate, GraphData


def build_structrag_prompt(
    graph: GraphData,
    uncertain_edges: Iterable[EdgeCandidate],
    template: GraphData,
    delta_edges: Iterable[Edge],
) -> str:
    payload = {
        "task": "Validate and correct the diagram-derived graph.",
        "rules": [
            "Return only valid JSON.",
            "Treat edges as undirected unless an arrow is explicitly present.",
            "Use the retrieved template as structural evidence, not as an automatic answer.",
            "Only add candidate edges when supported by structural plausibility and visual evidence.",
        ],
        "input_graph": graph.to_dict(),
        "uncertain_visual_edges": [edge.to_dict() for edge in uncertain_edges],
        "retrieved_template": template.to_dict(),
        "candidate_missing_edges": [[source, target] for source, target in delta_edges],
        "required_output_schema": {
            "status": "ok or error",
            "nodes": ["node_id"],
            "edges": [["source", "target"]],
            "added_edges": [["source", "target"]],
            "justification": "short explanation",
        },
    }
    return json.dumps(payload, indent=2)

