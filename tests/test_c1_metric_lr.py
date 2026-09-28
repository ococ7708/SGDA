"""Focused invariants for the E2 metric-only learning-rate change."""
from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.evaluate_c1_checkpoint import shuffle_context_within_subject, zero_metric_generator
from experiments.seediv_c1_end_to_end import _build_optimizer
from models.context_affective_metric import AffectiveMetricHead, MultiSourceAffectiveMetric, bound_symmetric
from utils.seediv_c1_c2_protocol import canonical_hash, load_config, resolve_c1_config


class C1MetricLRTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(901)

    def test_default_multiplier_is_one_and_optimizer_is_numerically_compatible(self):
        cfg = resolve_c1_config(load_config(ROOT / "configs/seediv_c1_full45.json"))
        self.assertEqual(cfg["metric_lr_multiplier"], 1.0)
        backbone, metric = torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)
        bcopy, mcopy = copy.deepcopy(backbone), copy.deepcopy(metric)
        grouped = _build_optimizer(backbone, metric, cfg)
        original = torch.optim.Adam(list(bcopy.parameters()) + list(mcopy.parameters()),
                                    lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
        for step in range(3):
            x = torch.full((4, 3), float(step + 1))
            y = torch.tensor([0, 1, 0, 1])
            grouped.zero_grad(); original.zero_grad()
            loss1 = F.cross_entropy(metric(backbone(x)), y)
            loss2 = F.cross_entropy(mcopy(bcopy(x)), y)
            loss1.backward(); loss2.backward(); grouped.step(); original.step()
        for left, right in zip(list(backbone.parameters()) + list(metric.parameters()),
                               list(bcopy.parameters()) + list(mcopy.parameters())):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_multiplier_changes_only_mechanism_lr_and_enters_effective_hash(self):
        base = resolve_c1_config(load_config(ROOT / "configs/seediv_c1_full45.json"))
        changed = copy.deepcopy(base)
        changed["metric_lr_multiplier"] = 3.0
        self.assertNotEqual(canonical_hash(base), canonical_hash(changed))
        backbone, metric = torch.nn.Linear(3, 2), torch.nn.Linear(2, 2)
        optimizer = _build_optimizer(backbone, metric, changed)
        self.assertEqual([g["lr"] for g in optimizer.param_groups], [0.001, 0.003])
        self.assertEqual([g["group_name"] for g in optimizer.param_groups], ["backbone", "mechanism"])

    def test_zero_h_recovers_base_cosine_metric_scores(self):
        prototypes = F.normalize(torch.randn(4, 8), dim=-1)
        basis = F.normalize(torch.randn(8, 2), dim=0)
        head = AffectiveMetricHead("E2", 5, prototypes, basis, rank=2, gamma=0.5, tau=0.07)
        mechanism = MultiSourceAffectiveMetric(1, head_sharing="independent", variant="E2",
            context_dim=5, prototypes=prototypes, basis=basis, rank=2, gamma=0.5, tau=0.07)
        context, z = torch.randn(11, 5), torch.randn(11, 8)
        expected = F.normalize(z, dim=-1) @ F.normalize(prototypes, dim=-1).T / 0.07
        zero_metric_generator(mechanism)
        observed = mechanism.source_outputs([context], [z])[0]
        torch.testing.assert_close(observed.logits, expected, rtol=1e-7, atol=1e-6)
        self.assertEqual(float(observed.diagnostics["H_spectral_norm"].max()), 0.0)

    def test_bound_metric_is_positive_definite_for_registered_gammas(self):
        raw = torch.randn(128, 2, 2) * 100
        for gamma in (0.5, 0.25):
            h = bound_symmetric(raw, gamma=gamma)
            eigenvalues = torch.linalg.eigvalsh(torch.eye(2)[None] + h)
            self.assertTrue(torch.all(eigenvalues > 1.0 - gamma - 1e-6))
            self.assertTrue(torch.all(torch.linalg.matrix_norm(h, ord=2) < gamma))

    def test_context_shuffle_is_seeded_and_only_returns_reindexed_context(self):
        context = torch.arange(40, dtype=torch.float32).reshape(10, 4)
        shuffled_a, perm_a = shuffle_context_within_subject(context, 20260928)
        shuffled_b, perm_b = shuffle_context_within_subject(context, 20260928)
        self.assertTrue(torch.equal(perm_a, perm_b))
        self.assertTrue(torch.equal(shuffled_a, shuffled_b))
        self.assertTrue(torch.equal(shuffled_a, context[perm_a]))
        self.assertEqual(sorted(perm_a.tolist()), list(range(len(context))))

    def test_a_and_b_configs_hold_all_non_target_parameters_fixed(self):
        a = resolve_c1_config(json.loads((ROOT / "configs/seediv_c1_e2_metric_lr_A.json").read_text(encoding="utf-8")))
        b = resolve_c1_config(json.loads((ROOT / "configs/seediv_c1_e2_metric_lr_B.json").read_text(encoding="utf-8")))
        self.assertEqual(a["metric_lr_multiplier"], 3.0)
        self.assertEqual(a["gamma"], 0.5)
        self.assertEqual(b["metric_lr_multiplier"], 3.0)
        self.assertEqual(b["gamma"], 0.25)
        for key in ("learning_rate", "tau", "rank", "fusion_tau", "fusion_mode", "weight_decay", "epochs", "batch_size"):
            self.assertEqual(a[key], b[key], key)


if __name__ == "__main__":
    unittest.main(verbosity=2)
