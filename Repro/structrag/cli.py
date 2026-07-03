from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from .evaluation import evaluate_graph
from .graph import GraphData, load_graph, load_structrag_input, load_templates
from .llm import (
    aggregate_graphs,
    call_openai_graph_validator,
    deterministic_validation,
    parse_graph_response,
)
from .prompting import build_structrag_prompt
from .retrieval import retrieve_template


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the lightweight StructRAG pipeline.")
    parser.add_argument("--input", required=True, help="Path to StructRAG input JSON.")
    parser.add_argument("--templates", required=True, help="Path to template library JSON.")
    parser.add_argument("--gold", help="Optional ground-truth graph JSON.")
    parser.add_argument("--output", help="Optional output path for predicted graph JSON.")
    parser.add_argument("--model", default="gpt-4-0613", help="OpenAI model name for GPT validation.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of coarse retrieval candidates.")
    parser.add_argument("--dry-run", action="store_true", help="Use deterministic validation instead of GPT.")
    parser.add_argument(
        "--allow-structural-only",
        action="store_true",
        help="In dry-run mode, accept template-only candidate edges.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    graph, uncertain_edges = load_structrag_input(args.input)
    templates = load_templates(args.templates)
    retrieval = retrieve_template(graph, templates, top_k=args.top_k)
    prompt = build_structrag_prompt(graph, uncertain_edges, retrieval.template, retrieval.delta_edges)

    if args.dry_run:
        predicted = deterministic_validation(
            graph,
            uncertain_edges,
            retrieval.delta_edges,
            allow_structural_only=args.allow_structural_only,
        )
    else:
        responses = call_openai_graph_validator(prompt, model=args.model)
        parsed: list[GraphData] = []
        for response in responses:
            try:
                parsed.append(parse_graph_response(response, directed=graph.directed))
            except Exception:
                continue
        predicted = aggregate_graphs(parsed) if parsed else graph

    report = {
        "selected_template": retrieval.template.name,
        "selected_topology": retrieval.template.topology,
        "normalized_ged": retrieval.normalized_ged,
        "coarse_candidates": retrieval.coarse_candidates,
        "candidate_missing_edges": retrieval.delta_edges,
        "prediction": predicted.to_dict(),
    }

    if args.gold:
        gold = load_graph(args.gold)
        metrics = evaluate_graph(predicted, gold)
        report["metrics"] = metrics.__dict__

    output_text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(output_text, encoding="utf-8")
    print(output_text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

