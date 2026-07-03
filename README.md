# StructRAG: Structure-Aware Diagram Reasoning for STEM Education

StructRAG is a framework for improving AI interpretation of diagram-based STEM questions. Many AI tutoring systems can read text reasonably well, but they often struggle with diagrams that contain missing connections, curved lines, overlapping edges, unlabeled junctions, or non-standard visual layouts. This limitation is important because many engineering, computing, and systems problems depend not only on recognizing labels, but also on understanding how components are connected. StructRAG addresses this problem by converting diagrams into graph structures and using structural retrieval to support more reliable reasoning.

The main significance of StructRAG is that it shifts diagram understanding from surface-level visual recognition to structure-aware reasoning. Instead of asking a language model to interpret a diagram directly from OCR text or raw visual cues, StructRAG builds an intermediate graph representation and compares it with canonical topology patterns. This makes the reasoning process more explicit, easier to inspect, and more suitable for educational feedback. The framework is especially useful for STEM diagrams such as circuits, network topologies, bus structures, bridge or mesh layouts, and hierarchical systems.

The StructRAG pipeline contains four main stages. First, the diagram-to-graph conversion module processes a student-uploaded diagram image using OCR and computer vision methods. It identifies candidate nodes, extracts straight or curved connections, assigns edge confidence scores, and produces an initial graph \(G=(V,E)\) together with uncertain candidate edges. Second, the structural pattern retrieval module compares the extracted graph with a curated template library of canonical graph structures. It uses coarse structural features for initial filtering and normalized graph edit distance for fine-grained matching.

Third, StructRAG constructs a pattern-aware prompt for the language model. This prompt includes the extracted graph, the retrieved template \(T*\), uncertain visual evidence, and candidate missing edges derived from the difference between the template and the extracted graph. Rather than relying only on labels or visual appearance, the prompt asks the model to reason about whether the diagram is structurally complete. Fourth, GPT-based reasoning is used to validate candidate corrections and return a corrected graph in a structured JSON format. Multiple completions are aggregated through a fixed voting procedure to improve stability.

The annotation and evaluation process is designed to make the graph-based task reproducible. Each diagram is manually converted into a ground-truth graph by STEM educators, with nodes representing structurally meaningful entities and edges representing valid connections. Diagrams are also assigned topology categories such as star, ring, chain/bus, bridge/mesh, tree, hybrid graph, cross-layer/layered, or irregular. Predicted graphs are evaluated using question-level accuracy and the edge-level F1 score, allowing the study to measure both complete-graph correctness and partial recovery of missing or incorrect edges.

In the empirical study, StructRAG is evaluated on 1,650 STEM diagram-based questions and compared with several baselines, including OCR+CV only, GPT-4 graph-only reasoning, direct-image GPT-4o, and ablated StructRAG variants. The results show that StructRAG improves both macro-average question-level accuracy and micro-averaged edge-level F1 score. Additional retrieval comparisons and component-level ablations show that the gains come from the combined effect of visual parsing, graph abstraction, template retrieval, graph edit distance matching, pattern-aware prompting, LLM reasoning, and output aggregation.

Overall, StructRAG provides a practical approach for building more reliable diagram-aware AI tutoring systems. Its value lies not only in improving recognition accuracy, but also in making diagram interpretation more transparent and structurally grounded. By explicitly modeling diagrams as graphs and using retrieved topology patterns as reasoning support, StructRAG can help AI systems identify missing connections, explain structural errors, and provide more useful feedback for diagram-heavy STEM learning tasks.

# StructRAG Reproduction Code

This repository provides a lightweight, reproducible implementation of the core StructRAG pipeline described in the manuscript. It is designed for transparency and review rather than maximum production performance. The code reproduces the main stages of StructRAG: graph construction, structural template retrieval, graph edit distance matching, pattern-aware prompting, optional GPT-based validation, output aggregation, and evaluation.

## What This Code Reproduces

StructRAG converts a diagram-derived graph into a corrected structured graph by combining visual evidence with structural priors. In the full study, the initial graph is produced from OCR and computer vision. In this release, the pipeline accepts the extracted graph directly as JSON, and the `vision.py` module provides an optional OpenCV-based scaffold for users who want to connect image parsing. This keeps the reproduction independent from proprietary diagram images while preserving the core reasoning pipeline.

The implemented stages are:

1. Load an extracted graph `G` and uncertain visual edge candidates `U`.
2. Retrieve structurally similar templates from a template library.
3. Apply coarse structural filtering followed by normalized graph edit distance.
4. Compute candidate missing edges `DeltaE = E(T*) - E(G)`.
5. Build a pattern-aware prompt containing `G`, `U`, `T*`, and `DeltaE`.
6. Generate or simulate corrected JSON graph outputs.
7. Aggregate three completions by graph-level majority voting, with edge-level voting as fallback.
8. Evaluate question-level accuracy and edge-level F1 score.

## Quick Start

```bash
cd structrag_repro
python -m structrag.cli \
  --input data/demo_input.json \
  --templates data/templates.json \
  --gold data/demo_gold.json \
  --dry-run
```

The dry-run mode does not call the OpenAI API. It deterministically accepts candidate edges that are supported by both the retrieved template and uncertain visual evidence. This is useful for testing the retrieval and evaluation pipeline.

## Optional GPT Mode

To use an OpenAI model for structural validation, install the optional dependency and set an API key:

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...
python -m structrag.cli \
  --input data/demo_input.json \
  --templates data/templates.json \
  --gold data/demo_gold.json \
  --model gpt-4-0613
```

The default decoding configuration follows the revised manuscript: temperature `0.2`, top-p `1.0`, maximum output length `800`, and three completions per question.

## Data Format

Input graphs use a compact JSON schema:

```json
{
  "graph": {
    "nodes": ["u", "v", "w", "x"],
    "edges": [["u", "v"], ["u", "w"]],
    "directed": false
  },
  "uncertain_edges": [
    {
      "source": "u",
      "target": "x",
      "confidence": 0.45,
      "signals": {
        "continuity": 0.50,
        "proximity": 0.42,
        "alignment": 0.41,
        "node_support": 0.35
      },
      "reason": "curved or partially occluded connection"
    }
  ]
}
```

The evaluation script canonicalizes undirected edges before computing metrics. Question-level accuracy requires an exact match between predicted and ground-truth nodes and edges. Edge-level F1 score is computed from true positive, false positive, and false negative edges.

## Notes for Manuscript Reproducibility

The included template library is a small demonstration library, not the full 375-template library used in the paper. To reproduce the full experiment, replace `data/templates.json` with the full anonymized template library and run the same command over all evaluation examples.


