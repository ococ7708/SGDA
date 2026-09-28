"""Synthetic C2 acceptance checks; no EEG model is trained here."""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.seediv_c2_experiment import (
    Evidence,
    _oracle_from_evidence,
    _oracle_metrics,
    _sample_weighted_loss,
    _validate_full_matrix_scope,
    _validate_pilot_acceptance,
    train_router_method,
)
from models.causal_evidence_history import (
    CausalMarginHistory,
    CausalProbabilityEMA,
    build_history_features,
    probability_ema,
)
from models.pairwise_utility_router import PairwiseUtilityRouter, build_router_features
from utils.pairwise_consistency import (
    apply_best_single_action,
    class_pairs,
    pairwise_residuals,
    single_action_logits,
    utility_labels,
)
from utils.seediv_c2_protocol import PARENT_CONFIG_HASH, c2_config_hash, inner_split, load_c2_config, sha256_file, validate_split


class SeedIVC2PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20260928)
        self.rng = np.random.default_rng(20260928)

    def _temporal_fixture(self, n=8):
        reference = self.rng.normal(size=(n, 4)).astype(np.float32)
        experts = self.rng.normal(size=(n, 3, 4)).astype(np.float32)
        subjects = np.asarray([2] * 4 + [5] * 4)
        sessions = np.ones(n, dtype=np.int8)
        trials = np.asarray([1, 1, 2, 2, 1, 1, 1, 2], dtype=np.int8)
        starts = np.asarray([0, 1, 0, 1, 0, 1, 2, 0], dtype=np.int32)
        return reference, experts, subjects, sessions, trials, starts

    def test_parent_and_c2_hashes_are_separate(self):
        config = load_c2_config(ROOT / "configs/seediv_c2_e0_s1s2s3.json")
        self.assertEqual(config["parent_config_hash"], PARENT_CONFIG_HASH)
        self.assertNotEqual(c2_config_hash(config), PARENT_CONFIG_HASH)
        self.assertEqual(config["sessions"], [1, 2, 3])
        self.assertEqual(config["pilot_methods"], ["B0", "B1", "C2-current", "C2-history"])

    def test_inner_split_is_disjoint_stable_and_has_required_counts(self):
        first = inner_split(1, 1)
        second = inner_split(1, 1)
        self.assertEqual(first, second)
        validate_split(first)
        self.assertEqual(len(first["source_dev_subject_ids"]), 2)
        self.assertEqual(len(first["router_oof_pool_subject_ids"]), 12)
        self.assertEqual([len(x["heldout_subject_ids"]) for x in first["oof_folds"]], [4, 4, 4])
        self.assertEqual([len(x["teacher_train_subject_ids"]) for x in first["oof_folds"]], [8, 8, 8])
        self.assertNotEqual(first, inner_split(2, 1))

    def test_history_future_isolation_and_first_window_reset(self):
        ref, exp, sid, ses, trial, starts = self._temporal_fixture()
        baseline = build_history_features(ref, exp, sid, ses, trial, starts)
        changed_ref, changed_exp = ref.copy(), exp.copy()
        changed_ref[1] += 1000
        changed_exp[1] -= 500
        changed_ref[6:] *= -99
        changed_exp[6:] += 200
        changed = build_history_features(changed_ref, changed_exp, sid, ses, trial, starts)
        # A future value cannot affect itself or any preceding window in its trial.
        np.testing.assert_allclose(baseline[[0, 4, 5]], changed[[0, 4, 5]], atol=0, rtol=0)
        # Each new trial begins without past information.
        for index in (0, 2, 4, 7):
            self.assertEqual(float(baseline[index, 0, 0, 5]), 0.0)
            np.testing.assert_array_equal(baseline[index, ..., :5], 0.0)

    def test_history_full_sequence_equals_chunked_state_updates(self):
        ref, exp, sid, ses, trial, starts = self._temporal_fixture()
        full = build_history_features(ref, exp, sid, ses, trial, starts, alpha=0.8)
        pairs = class_pairs(4)
        m0 = np.stack([ref[:, a] - ref[:, b] for a, b in pairs], axis=-1)
        mk = np.stack([exp[:, :, a] - exp[:, :, b] for a, b in pairs], axis=-1)
        dk = mk - m0[:, None, :]
        state = CausalMarginHistory(alpha=0.8)
        chunked = np.zeros_like(full)
        # Deliberately split within trial, preserving carried state.
        for part in (range(0, 3), range(3, 6), range(6, 8)):
            for index in part:
                key = (sid[index], ses[index], trial[index])
                chunked[index] = state.update(m0[index], mk[index], dk[index], key, starts[index])
        np.testing.assert_allclose(full, chunked, atol=5e-7, rtol=5e-7)

    def test_probability_ema_full_sequence_equals_chunked_state_updates(self):
        p = self.rng.random((8, 4), dtype=np.float32)
        p /= p.sum(axis=1, keepdims=True)
        sid = np.asarray([1] * 4 + [2] * 4)
        ses = np.ones(8, dtype=np.int8)
        trials = np.asarray([1, 1, 2, 2, 1, 1, 1, 2])
        starts = np.asarray([0, 1, 0, 1, 0, 1, 2, 0])
        full = probability_ema(p, sid, ses, trials, starts, alpha=0.8)
        state = CausalProbabilityEMA(0.8)
        chunked = np.zeros_like(full)
        for part in (range(0, 2), range(2, 6), range(6, 8)):
            for index in part:
                chunked[index] = state.update(p[index], (sid[index], ses[index], trials[index]), starts[index])
        np.testing.assert_allclose(full, chunked, atol=1e-7, rtol=1e-7)

    def test_noop_preserves_probabilities_and_predictions(self):
        ref = torch.randn(17, 4)
        experts = torch.randn(17, 8, 4)
        residual = pairwise_residuals(ref, experts, clip_limit=2.0)
        negative_utilities = torch.full((17, 8, 6), -0.1)
        revised, action = apply_best_single_action(ref, residual, negative_utilities, alpha=0.25)
        self.assertTrue(torch.equal(action, torch.zeros_like(action)))
        torch.testing.assert_close(ref.softmax(-1), revised.softmax(-1), atol=1e-7, rtol=1e-7)
        self.assertTrue(torch.equal(ref.argmax(-1), revised.argmax(-1)))

    def test_full_multiclass_ce_utility_matches_direct_recomputation(self):
        ref = torch.randn(13, 4)
        expert = torch.randn(13, 5, 4)
        labels = torch.arange(13) % 4
        alpha, clip = 0.25, 2.0
        values = utility_labels(ref, expert, labels, alpha=alpha, clip_limit=clip)
        residual = pairwise_residuals(ref, expert, clip_limit=clip)
        pairs = class_pairs(4)
        base_ce = F.cross_entropy(ref, labels, reduction="none")
        for source in range(expert.shape[1]):
            for pair_index, pair in enumerate(pairs):
                action = single_action_logits(ref, residual[:, source, pair_index], pair, alpha)
                expected = base_ce - F.cross_entropy(action, labels, reduction="none")
                torch.testing.assert_close(values[:, source, pair_index], expected, atol=1e-7, rtol=1e-7)

    def test_pairwise_action_changes_only_selected_margin_by_alpha_residual(self):
        ref = torch.randn(9, 4)
        residual = torch.full((9,), 1.2)
        a, b = (1, 3)
        action = single_action_logits(ref, residual, (a, b), alpha=0.25)
        old_margin = ref[:, a] - ref[:, b]
        new_margin = action[:, a] - action[:, b]
        torch.testing.assert_close(new_margin - old_margin, torch.full_like(old_margin, 0.3), atol=1e-6, rtol=1e-6)
        for other in range(4):
            if other not in (a, b):
                torch.testing.assert_close(action[:, other], (ref - ref.mean(-1, keepdim=True))[:, other], atol=1e-6, rtol=1e-6)

    def test_shared_router_accepts_8_12_and_14_sources(self):
        n, classes = 7, 4
        reference = torch.randn(n, classes)
        pair_features = torch.randn(6, 7)
        train_experts = torch.randn(n, 8, classes)
        train_distances = torch.rand(n, 8)
        train_features = build_router_features(reference, train_experts, pair_features, train_distances)
        router = PairwiseUtilityRouter(train_features.shape[-1])
        router.standardizer.fit(train_features)
        for sources in (8, 12, 14):
            experts = torch.randn(n, sources, classes)
            distances = torch.rand(n, sources)
            features = build_router_features(reference, experts, pair_features, distances)
            output = router(features)
            self.assertEqual(tuple(output.shape), (n, sources, 6))

    def test_mse_and_huber_objectives_are_distinct_and_candidate_balanced(self):
        predicted = torch.tensor([[[0.0, 0.0]], [[0.0, 0.0]]])
        target = torch.tensor([[[1.0, 1.0]], [[10.0, 10.0]]])
        mse = _sample_weighted_loss(predicted, target, "MSE")
        huber = _sample_weighted_loss(predicted, target, "Huber")
        self.assertNotEqual(float(mse), float(huber))
        # A broadcast candidate count does not alter the scalar objective.
        duplicated = target.repeat(1, 3, 4)
        duplicated_prediction = predicted.repeat(1, 3, 4)
        self.assertAlmostEqual(float(_sample_weighted_loss(duplicated_prediction, duplicated, "MSE")), float(mse), places=6)

    def test_oracle_action_is_recorded_as_sample_level_label_diagnostic(self):
        n, sources = 11, 3
        ref = self.rng.normal(size=(n, 4)).astype(np.float32)
        experts = self.rng.normal(size=(n, sources, 4)).astype(np.float32)
        labels = np.arange(n, dtype=np.int64) % 4
        evidence = Evidence(
            reference_logits=ref,
            expert_logits=experts,
            labels=labels,
            source_distances=np.zeros((n, sources), dtype=np.float32),
            prototype_pair_features=np.zeros((6, 4), dtype=np.float32),
            subject_ids=np.ones(n, dtype=np.int16),
            session_ids=np.ones(n, dtype=np.int8),
            trial_ids=np.repeat([1, 2], [6, 5]).astype(np.int8),
            window_starts=np.concatenate([np.arange(6), np.arange(5)]).astype(np.int32),
            sample_ids=np.asarray([f"sample-{index}" for index in range(n)]),
            teacher_ids=np.full(n, "synthetic", dtype="U32"),
        )
        config = {"router": {"residual_clip": 2.0, "action_alpha": 0.25}}
        payload = _oracle_from_evidence(evidence, config, torch.device("cpu"))
        metrics = _oracle_metrics(evidence, payload)
        self.assertEqual(payload["oracle_action_code"].shape, (n,))
        self.assertEqual(payload["oracle_source_index"].shape, (n,))
        self.assertEqual(payload["oracle_pair_index"].shape, (n,))
        self.assertEqual(len(metrics["confusion_matrix"]), 4)
        self.assertEqual(metrics["label_use"], "diagnostic upper bound only; not available to router inference or training decisions")

    def test_full_matrix_requires_owner_acceptance_bound_to_exact_pilot(self):
        required = {
            "b0_matches_parent", "oof_sample_exclusion_verified", "history_future_isolation_verified",
            "trial_reset_and_chunking_verified", "noop_invariant_verified", "cached_ce_utility_verified",
            "final_fusion_sample_diagnostics_verified", "teachers_frozen_during_router_training",
            "method_controls_independently_verified", "resume_consistency_verified_or_not_supported",
        }
        config_hash = "c2-test-config"
        implementations = {"runner.py": "abc123"}
        with tempfile.TemporaryDirectory() as folder:
            pilot = Path(folder) / "pilot.json"
            acceptance = Path(folder) / "acceptance.json"
            pilot.write_text(json.dumps({
                "status": "PASS", "config_hash": config_hash,
                "implementation_hashes": implementations,
            }), encoding="utf-8")
            record = {
                "status": "PASS", "accepted_by": "owner", "accepted_at": "2026-09-28",
                "config_hash": config_hash, "implementation_hashes": implementations,
                "pilot_report_sha256": sha256_file(pilot),
                "checklist": {item: True for item in required},
            }
            acceptance.write_text(json.dumps(record), encoding="utf-8")
            _validate_pilot_acceptance(pilot, acceptance, config_hash, implementations)
            record["pilot_report_sha256"] = "stale"
            acceptance.write_text(json.dumps(record), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "exact pilot report"):
                _validate_pilot_acceptance(pilot, acceptance, config_hash, implementations)

    def test_full_mode_rejects_partial_targets_or_methods(self):
        config = {
            "sessions": [1, 2, 3],
            "subject_ids": list(range(1, 16)),
            "seeds": [42, 43, 44],
            "methods": ["B0", "B1", "C2-current", "C2-current-EMA", "C2-history", "C2-current-matched", "C2-current-Huber"],
        }
        _validate_full_matrix_scope("full", [1, 2, 3], list(range(1, 16)), [42, 43, 44], config["methods"], config)
        with self.assertRaisesRegex(RuntimeError, "complete frozen matrix"):
            _validate_full_matrix_scope("full", [1], [1], [42], ["B0"], config)

    def test_one_epoch_router_smoke_writes_source_dev_and_target_reports(self):
        def evidence(n, subject_id, seed):
            rng = np.random.default_rng(seed)
            return Evidence(
                reference_logits=rng.normal(size=(n, 4)).astype(np.float32),
                expert_logits=rng.normal(size=(n, 8, 4)).astype(np.float32),
                labels=(np.arange(n) % 4).astype(np.int64),
                source_distances=rng.uniform(size=(n, 8)).astype(np.float32),
                prototype_pair_features=rng.normal(size=(6, 7)).astype(np.float32),
                subject_ids=np.full(n, subject_id, dtype=np.int16),
                session_ids=np.ones(n, dtype=np.int8),
                trial_ids=np.repeat([1, 2], [n // 2, n - n // 2]).astype(np.int8),
                window_starts=np.concatenate([np.arange(n // 2), np.arange(n - n // 2)]).astype(np.int32),
                sample_ids=np.asarray([f"{subject_id}-{index}" for index in range(n)]),
                teacher_ids=np.full(n, "synthetic", dtype="U32"),
            )

        config = {
            "router": {
                "epochs": 1, "batch_size": 16, "learning_rate": 1e-3,
                "action_alpha": 0.25, "residual_clip": 2.0,
                "history_alpha": 0.8, "output_ema_alpha": 0.8,
                "source_dev_checkpoint_rule": "mean_subject_accuracy",
                "target_checkpoint_rule": "target_accuracy_report_only",
            }
        }
        oof = evidence(24, 2, 11)
        dev = evidence(8, 3, 12)
        target = evidence(12, 1, 13)
        with tempfile.TemporaryDirectory() as folder:
            summary = train_router_method(
                method="C2-current", session=1, target=1, seed=42,
                oof=oof, dev=dev, target_evidence=target,
                config=config, device=torch.device("cpu"), run_dir=Path(folder), run_manifest={},
            )
            self.assertEqual(summary["status"], "PASS")
            self.assertIn("oracle_single_action", summary["target_best_metrics_report_only"])
            self.assertTrue((Path(folder) / "source_dev_selected.pt").is_file())
            self.assertTrue((Path(folder) / "target_best.pt").is_file())
            self.assertTrue((Path(folder) / "epoch_metrics.csv").is_file())


if __name__ == "__main__":
    unittest.main(verbosity=2)
