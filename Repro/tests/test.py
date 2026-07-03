import unittest

from structrag.evaluation import evaluate_graph
from structrag.graph import load_graph, load_structrag_input, load_templates
from structrag.llm import deterministic_validation
from structrag.retrieval import retrieve_template


class StructRAGDemoTest(unittest.TestCase):
    def test_demo_pipeline_recovers_missing_edge(self):
        graph, uncertain = load_structrag_input("data/demo_input.json")
        templates = load_templates("data/templates.json")
        result = retrieve_template(graph, templates)
        predicted = deterministic_validation(graph, uncertain, result.delta_edges)
        gold = load_graph("data/demo_gold.json")
        metrics = evaluate_graph(predicted, gold)
        self.assertEqual(result.template.name, "star_4")
        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.edge_f1, 1.0)


if __name__ == "__main__":
    unittest.main()
