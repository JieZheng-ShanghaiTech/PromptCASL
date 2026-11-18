"""Regression checks for runtime and analysis pathway selection.

Run from the repository root:
    python -B -m unittest discover -s tests -v
"""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


RUNTIME = load_module("pathway_runtime_under_test", ROOT / "src/pair_pathway.py")
ANALYSIS_PATH = ROOT / "ana/attn_substrate.py"
if not ANALYSIS_PATH.exists():
    ANALYSIS_PATH = ROOT / "ana/pair_pw_validation.py"
ANALYSIS = load_module("pathway_analysis_under_test", ANALYSIS_PATH)


class PathwaySelectionTests(unittest.TestCase):
    def assert_selected(self, expression, shared, expected, k=2, union=None, sizes=None):
        activity = {"test": np.asarray(expression, dtype=np.float32)[:, None]}
        union = np.arange(len(expression), dtype=np.int64) if union is None else np.asarray(union, dtype=np.int64)
        shared = np.asarray(shared, dtype=np.int64)
        sizes = np.asarray(sizes if sizes is not None else [1] * len(expression), dtype=np.int64)
        builder = RUNTIME.PairPathwayEvidenceBuilder.__new__(RUNTIME.PairPathwayEvidenceBuilder)
        builder.k = k
        builder.activity = activity
        builder.pathway_sizes = sizes
        with self.subTest(selector="runtime"):
            self.assertEqual(builder._select_topk(union, shared, "test").tolist(), expected)
        with self.subTest(selector="analysis"):
            with patch.dict(ANALYSIS.__dict__, ACT=activity, ACTIVITY=activity, PSIZE=sizes):
                self.assertEqual(ANALYSIS._select_topk(union, shared, "test", k).tolist(), expected)

    def test_expression_ties_use_fixed_index_despite_different_sizes(self):
        self.assert_selected([2, -2, 2, -2], [], [0, 1], sizes=[100, 3, 2, 1])

    def test_shared_membership_precedes_expression_magnitude(self):
        self.assert_selected([100, 0.1, 50, 20], [1, 3], [3, 1])

    def test_expression_ranks_within_membership_groups(self):
        self.assert_selected([100, 50, -3, 2], [2, 3], [2, 3, 0], k=3)

    def test_expression_ranking_uses_absolute_values(self):
        self.assert_selected([0.1, -5, 3, -4], [], [1, 3])

    def test_size_changes_cannot_change_a_tied_selection(self):
        self.assert_selected([1, 1, 1, 1], [], [0, 1], sizes=[1, 2, 3, 4])
        self.assert_selected([1, 1, 1, 1], [], [0, 1], sizes=[4, 3, 2, 1])

    def test_all_candidates_are_retained_when_capacity_is_sufficient(self):
        self.assert_selected([0.1, 3, -5, 4], [], [0, 2], k=2, union=[0, 2])

    def test_empty_candidate_set_stays_empty(self):
        self.assert_selected([0.1, 3, -5, 4], [], [], union=[])


if __name__ == "__main__":
    unittest.main()
