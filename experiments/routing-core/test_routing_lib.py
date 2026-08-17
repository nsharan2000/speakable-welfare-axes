"""Fast CPU tests for routing_lib; no model download or fitted lens required.

Run with the jlens environment:
``python experiments/routing-core/test_routing_lib.py``.
"""

import os
import sys
import unittest

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from routing_lib import (
    HISTORICAL_MIN_ID_INVALID,
    JSpace,
    assert_decompositions_converged,
    make_valid_mask,
    valid_mask_provenance,
)


class _FakeLens:
    def __init__(self, d_model=4):
        self.jacobians = {0: torch.eye(d_model)}
        self.d_model = d_model


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(4, 6, bias=False)
        self.name_or_path = "synthetic-test-model"
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [-1.0, -1.0, 0.0, 0.0],
                [5.0, 5.0, 5.0, 5.0],
            ]))


class _FakeTokenizer:
    all_special_ids = [4]

    def __len__(self):
        return 5


class RoutingLibTests(unittest.TestCase):
    def setUp(self):
        self.model = _FakeModel()
        self.lens = _FakeLens()
        self.tok = _FakeTokenizer()
        # warn_on_drift is for real Qwen runs; the synthetic vocab always drifts.
        self.mask = make_valid_mask(self.tok, vocab_size=6, warn_on_drift=False)
        self.space = JSpace(
            self.lens,
            self.model,
            layer=0,
            valid_token_mask=self.mask,
            device="cpu",
            norm_chunk_size=2,
        )

    def test_mask_uses_tokenizer_length_and_special_ids(self):
        self.assertEqual(self.mask.tolist(), [True, True, True, True, False, False])

    def test_known_nonnegative_mixture_reconstructs(self):
        x = np.array([2.0, 0.5, 0.0, 0.0], dtype=np.float32)
        result = self.space.decompose(x, k=4)
        self.assertGreater(result["j_share"], 0.9999)
        self.assertGreater(result["reconstruction_r2"], 0.9999)
        self.assertTrue(np.all(result["coeffs"] >= 0))
        np.testing.assert_allclose(result["x_j"] + result["x_perp"], x, atol=1e-6)
        self.assertTrue(result["diagnostics"]["nnls"]["converged"])
        self.assertLessEqual(len(result["token_ids"]), 2)

    def test_positive_scale_invariance(self):
        x = np.array([2.0, 0.5, 0.0, 0.0], dtype=np.float32)
        result = self.space.decompose(x, k=4)
        scaled = self.space.decompose(10.0 * x, k=4)
        self.assertAlmostEqual(result["j_share"], scaled["j_share"], places=6)
        self.assertEqual(
            result["token_ids_by_coefficient"],
            scaled["token_ids_by_coefficient"],
        )

    def test_compatibility_alias(self):
        result = self.space.decompose(np.ones(4, dtype=np.float32), k=4)
        self.assertEqual(result["var_fraction"], result["j_share"])

    def test_rejects_zero_and_wrong_shape(self):
        with self.assertRaisesRegex(ValueError, "zero-length"):
            self.space.decompose(np.zeros(4, dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shape"):
            self.space.decompose(np.zeros(3, dtype=np.float32))

    def test_rejects_model_lens_dimension_mismatch(self):
        with self.assertRaisesRegex(ValueError, "dimension mismatch"):
            JSpace(_FakeLens(d_model=3), self.model, layer=0, device="cpu")

    def test_records_nnls_convergence(self):
        result = self.space.decompose(
            np.array([2.0, 0.5, 0.0, 0.0], dtype=np.float32), k=4
        )
        self.assertTrue(result["diagnostics"]["nnls_converged"])
        self.assertEqual(result["diagnostics"]["nnls_failures"], [])

    def test_convergence_gate_passes_on_clean_records(self):
        records = {
            "v_mold": {"algorithm_version": "2.0.0", "nnls_converged": True},
            "nested": {"u_gold": {"algorithm_version": "2.0.0",
                                  "nnls_converged": True}},
        }
        self.assertEqual(assert_decompositions_converged(records), 0)

    def test_convergence_gate_raises_and_names_the_bad_cell(self):
        records = {
            "good": {"algorithm_version": "2.0.0", "nnls_converged": True},
            "sweep": [{"algorithm_version": "2.0.0", "nnls_converged": False}],
        }
        with self.assertRaisesRegex(RuntimeError, r"sweep\[0\]"):
            assert_decompositions_converged(records, context="test.json")

    def test_convergence_gate_ignores_unversioned_v1_records(self):
        records = {"legacy": {"var_fraction": 0.05, "tokens": ["a"]}}
        self.assertEqual(assert_decompositions_converged(records), 0)

    def test_mask_provenance_flags_drift_from_frozen_cutoff(self):
        provenance = valid_mask_provenance(self.tok, vocab_size=6)
        self.assertEqual(provenance["min_id_invalid"], 5)
        self.assertEqual(provenance["source"], "len(tokenizer)")
        self.assertFalse(provenance["matches_historical"])
        self.assertEqual(
            provenance["historical_min_id_invalid"], HISTORICAL_MIN_ID_INVALID
        )
        self.assertEqual(provenance["n_special_excluded"], 1)

    def test_explicit_cutoff_is_recorded_as_explicit(self):
        provenance = valid_mask_provenance(self.tok, vocab_size=6, min_id_invalid=3)
        self.assertEqual(provenance["min_id_invalid"], 3)
        self.assertEqual(provenance["source"], "explicit")


if __name__ == "__main__":
    unittest.main()
