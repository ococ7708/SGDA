"""Auditable SEED-IV C2 pipeline built around the completed E0 checkpoints.

The default operation is a read-only parent-artifact audit.  Real EEG work is
only launched by the explicit ``pilot --execute`` or ``full --execute`` modes.
The full matrix is intentionally not run by this module during import/tests.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SGDA_SEEDIV_LOAD_WORKERS", "1")

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

from config.setting import Setting
from data_utils.constants.path_mapper import path_mapper
from data_utils.constants.label_text_mapper import getLabelMapper
from data_utils.load_data import get_data
from data_utils.text_to_vector import label_to_vector
from models.causal_evidence_history import (
    append_history_features,
    build_history_features,
    probability_ema,
)
from models.context_affective_metric import MultiSourceAffectiveMetric
from models.geosem_stda import GeoSemSTDA, compute_source_domain_centroids
from models.pairwise_utility_router import PairwiseUtilityRouter, build_router_features
from utils.mix_utils import setup_seed, zscore_subject_wise
from utils.pairwise_consistency import (
    apply_best_single_action,
    class_pairs,
    pairwise_residuals,
    utility_labels,
)
from utils.seediv_c2_protocol import (
    DEFAULT_CONFIG,
    PARENT_CONFIG_HASH,
    array_fingerprint,
    build_sample_ids,
    c2_config_hash,
    inner_split,
    json_fingerprint,
    load_c2_config,
    parent_checkpoint_path,
    parent_run_dir,
    run_identity,
    sha256_file,
    validate_split,
)
from utils.seediv_c1_c2_protocol import canonical_hash, resolve_c1_config


@dataclass
class SubjectData:
    subject_id: int
    x: np.ndarray
    y: np.ndarray
    trial_ids: np.ndarray
    window_starts: np.ndarray
    sample_ids: np.ndarray
    input_fingerprint: str
    raw_feature_fingerprint: str
    label_fingerprint: str
    source_file_path: str
    source_file_sha256: str


@dataclass
class Evidence:
    reference_logits: np.ndarray
    expert_logits: np.ndarray
    labels: np.ndarray
    source_distances: np.ndarray
    prototype_pair_features: np.ndarray
    subject_ids: np.ndarray
    session_ids: np.ndarray
    trial_ids: np.ndarray
    window_starts: np.ndarray
    sample_ids: np.ndarray
    teacher_ids: np.ndarray

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "reference_logits": self.reference_logits.astype(np.float32),
            "expert_logits": self.expert_logits.astype(np.float32),
            "labels": self.labels.astype(np.int64),
            "source_distances": self.source_distances.astype(np.float32),
            "prototype_pair_features": self.prototype_pair_features.astype(np.float32),
            "subject_ids": self.subject_ids.astype(np.int16),
            "session_ids": self.session_ids.astype(np.int8),
            "trial_ids": self.trial_ids.astype(np.int8),
            "window_starts": self.window_starts.astype(np.int32),
            "sample_ids": self.sample_ids.astype("U32"),
            "teacher_ids": self.teacher_ids.astype("U32"),
        }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _config_file_hash(path: Path) -> str:
    return sha256_file(path)


def _implementation_hashes() -> dict[str, str]:
    files = (
        "experiments/seediv_c2_experiment.py",
        "experiments/seediv_c1_end_to_end.py",
        "models/causal_evidence_history.py",
        "models/pairwise_utility_router.py",
        "models/context_affective_metric.py",
        "models/geosem_stda.py",
        "utils/pairwise_consistency.py",
        "utils/seediv_c2_protocol.py",
        "utils/seediv_c1_c2_protocol.py",
        "utils/mix_utils.py",
        "data_utils/load_data.py",
    )
    return {relative: sha256_file(ROOT / relative) for relative in files}


def _same_number(left: Any, right: Any, tolerance: float = 1e-10) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return False


def _validate_pilot_acceptance(
    pilot_report_path: Path, acceptance_path: Path, config_hash: str,
    implementation_hashes: dict[str, str],
) -> None:
    if not pilot_report_path.is_file():
        raise RuntimeError(f"full matrix blocked: pilot report missing at {pilot_report_path}")
    pilot_report = json.loads(pilot_report_path.read_text(encoding="utf-8"))
    if pilot_report.get("status") != "PASS":
        raise RuntimeError("full matrix blocked: the pilot report is not PASS")
    if pilot_report.get("config_hash") != config_hash:
        raise RuntimeError("full matrix blocked: pilot config hash does not match current C2 config")
    if pilot_report.get("implementation_hashes") != implementation_hashes:
        raise RuntimeError("full matrix blocked: pilot implementation hashes do not match current code")
    if not acceptance_path.is_file():
        raise RuntimeError(f"full matrix blocked: project-owner acceptance record missing at {acceptance_path}")
    acceptance = json.loads(acceptance_path.read_text(encoding="utf-8"))
    required_acceptance_items = {
        "b0_matches_parent",
        "oof_sample_exclusion_verified",
        "history_future_isolation_verified",
        "trial_reset_and_chunking_verified",
        "noop_invariant_verified",
        "cached_ce_utility_verified",
        "final_fusion_sample_diagnostics_verified",
        "teachers_frozen_during_router_training",
        "method_controls_independently_verified",
        "resume_consistency_verified_or_not_supported",
    }
    checklist = acceptance.get("checklist", {})
    if acceptance.get("status") != "PASS" or not acceptance.get("accepted_by") or not acceptance.get("accepted_at"):
        raise RuntimeError("full matrix blocked: acceptance record must be PASS and signed/dated by project owner")
    if acceptance.get("config_hash") != config_hash:
        raise RuntimeError("full matrix blocked: acceptance record config hash differs from current C2 config")
    if acceptance.get("implementation_hashes") != implementation_hashes:
        raise RuntimeError("full matrix blocked: acceptance record implementation hashes differ from current code")
    if acceptance.get("pilot_report_sha256") != sha256_file(pilot_report_path):
        raise RuntimeError("full matrix blocked: acceptance record is not bound to this exact pilot report")
    if set(checklist) != required_acceptance_items or not all(checklist.values()):
        raise RuntimeError("full matrix blocked: acceptance checklist is incomplete or contains failed items")


def _validate_full_matrix_scope(mode: str, sessions: list[int], targets: list[int], seeds: list[int],
                                methods: list[str], config: dict[str, Any]) -> None:
    if mode != "full":
        return
    expected = (
        list(config["sessions"]), list(config["subject_ids"]),
        list(config["seeds"]), list(config["methods"]),
    )
    observed = (sessions, targets, seeds, methods)
    if observed != expected:
        raise RuntimeError(
            "full mode requires the complete frozen matrix: Sessions 1/2/3, all 15 targets, "
            "seeds 42/43/44, and every registered method in config order"
        )


def audit_parent_matrix(config_path: Path, output: Path, *, load_weights: bool = True) -> dict[str, Any]:
    """Recompute the reviewed C1 hash and inspect all configured E0 parent runs."""
    cfg = load_c2_config(config_path)
    parent_config_path = ROOT / cfg["parent_config_path"]
    parent_file_config = json.loads(parent_config_path.read_text(encoding="utf-8"))
    expected_effective = resolve_c1_config(parent_file_config)
    expected_hash = canonical_hash(expected_effective)
    if expected_hash != PARENT_CONFIG_HASH:
        raise RuntimeError(
            f"reviewed C1 config no longer resolves to {PARENT_CONFIG_HASH}: got {expected_hash}"
        )

    cells: dict[tuple[int, int, int], dict[str, Any]] = {}
    failures: list[str] = []
    for session in cfg["sessions"]:
        for target in cfg["subject_ids"]:
            for seed in cfg["seeds"]:
                key = (int(session), int(target), int(seed))
                run_dir = parent_run_dir(*key, root=ROOT)
                manifest_path = run_dir / "run_manifest.json"
                metrics_path = run_dir / "metrics.json"
                history_path = run_dir / "epoch_metrics.json"
                checkpoint_path = run_dir / "target_best.pt"
                complete_path = run_dir / "COMPLETE"
                absent = [p.name for p in (manifest_path, metrics_path, history_path, checkpoint_path, complete_path) if not p.is_file()]
                if absent:
                    failures.append(f"{key}: missing {','.join(absent)}")
                    continue
                try:
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                    history = json.loads(history_path.read_text(encoding="utf-8"))
                    if manifest.get("variant") != "E0" or tuple(map(int, (manifest.get("session"), manifest.get("target"), manifest.get("seed")))) != key:
                        raise ValueError("manifest run identity mismatch")
                    effective = manifest.get("effective_config")
                    computed_hash = canonical_hash(effective)
                    if computed_hash != PARENT_CONFIG_HASH or manifest.get("effective_config_hash") != computed_hash:
                        raise ValueError(f"effective config hash mismatch: {computed_hash}")
                    if effective != expected_effective:
                        raise ValueError("manifest effective config differs from reviewed parent config")
                    expected_sources = [sid for sid in cfg["subject_ids"] if sid != target]
                    if list(map(int, manifest.get("source_ids", []))) != expected_sources:
                        raise ValueError("parent E0 source IDs do not match outer LOSO")
                    if complete_path.read_text(encoding="ascii").strip() != "PASS":
                        raise ValueError("COMPLETE marker is not PASS")
                    if len(history) != 200:
                        raise ValueError(f"expected 200 epoch rows, got {len(history)}")
                    hist_acc = [float(row["accuracy"]) for row in history]
                    max_acc = max(hist_acc)
                    first_max_epoch = next(i + 1 for i, value in enumerate(hist_acc) if value == max_acc)
                    if not _same_number(metrics.get("accuracy"), max_acc) or int(metrics.get("best_epoch", -1)) != first_max_epoch:
                        raise ValueError("metrics do not agree with the first maximum-accuracy history epoch")
                    digest = sha256_file(checkpoint_path)
                    if load_weights:
                        state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                        embedded_manifest = state.get("manifest", {})
                        if int(state.get("epoch", -1)) != int(metrics.get("best_epoch", -2)):
                            raise ValueError("checkpoint epoch does not match metrics")
                        if int(embedded_manifest.get("session", -1)) != session or int(embedded_manifest.get("target", -1)) != target or int(embedded_manifest.get("seed", -1)) != seed:
                            raise ValueError("checkpoint embedded run identity mismatch")
                        if not _same_number(state.get("metrics", {}).get("accuracy"), metrics.get("accuracy")):
                            raise ValueError("checkpoint accuracy does not match metrics.json")
                        if "model" not in state or "mechanism" not in state or "centroids" not in state:
                            raise ValueError("checkpoint is missing model/mechanism/source-centroid state")
                        del state
                    cells[key] = {
                        "session": session,
                        "target": target,
                        "seed": seed,
                        "relative_checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
                        "parent_config_hash": computed_hash,
                        "parent_checkpoint_sha256": digest,
                        "parent_best_epoch": int(metrics["best_epoch"]),
                        "parent_accuracy": float(metrics["accuracy"]),
                        "parent_macro_f1": float(metrics["macro_f1"]),
                        "parent_balanced_accuracy": float(metrics["balanced_accuracy"]),
                    }
                except Exception as exc:  # evidence report must contain every broken cell
                    failures.append(f"{key}: {type(exc).__name__}: {exc}")
                finally:
                    if load_weights and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()

    expected_cells = len(cfg["sessions"]) * len(cfg["subject_ids"]) * len(cfg["seeds"])
    if len(cells) != expected_cells:
        failures.append(f"valid parent run cells={len(cells)}, expected={expected_cells}")
    missing_cells = [
        {"session": session, "target": target, "seed": seed}
        for session in cfg["sessions"]
        for target in cfg["subject_ids"]
        for seed in cfg["seeds"]
        if (int(session), int(target), int(seed)) not in cells
    ]
    payload = {
        "status": "PASS" if not failures else "BLOCKED_MISSING_OR_INVALID_PARENT",
        "scope": "existing E0 parent runs only; no training performed",
        "parent_config_path": str(parent_config_path.relative_to(ROOT)),
        "parent_config_file_sha256": _config_file_hash(parent_config_path),
        "parent_config_hash_recomputed": expected_hash,
        "parent_config_hash_expected": PARENT_CONFIG_HASH,
        "c2_config_path": str(config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path),
        "c2_config_hash": c2_config_hash(cfg),
        "expected_cells": expected_cells,
        "valid_cells": len(cells),
        "missing_cells": missing_cells,
        "checkpoint_weight_loads": bool(load_weights),
        "cells": [cells[key] for key in sorted(cells)],
        "failures": failures,
    }
    _write_json(output, payload)
    return payload


def _load_seediv_session_data(config: dict[str, Any]) -> dict[int, dict[int, SubjectData]]:
    """Load DE-LDS using the repository loader and preserve trial/window IDs."""
    prep = config["preprocessing"]
    setting = Setting(
        dataset="seediv_de_lds",
        dataset_path=path_mapper["seediv_de_lds"],
        pass_band=[0.3, 50],
        extract_bands=None,
        time_window=1,
        overlap=0,
        sample_length=int(prep["sample_length"]),
        stride=int(prep["stride"]),
        seed=42,
        feature_type="de_lds",
        only_seg=False,
        experiment_mode="subject-independent",
        sessions=list(config["sessions"]),
        onehot=False,
    )
    all_data, all_labels, channels, bands, num_classes = get_data(setting)
    if channels != 62 or bands != 5 or num_classes != 4:
        raise ValueError(f"unexpected SEED-IV shape/classes: {channels=}, {bands=}, {num_classes=}")
    class_order = [getLabelMapper("seediv_de_lds", False)[index] for index in range(4)]
    if class_order != ["neutral", "sad", "fear", "happy"]:
        raise ValueError(f"unexpected SEED-IV class order: {class_order}")
    sessions: dict[int, dict[int, SubjectData]] = {}
    for session in config["sessions"]:
        session_idx = list(config["sessions"]).index(int(session))
        subjects: dict[int, SubjectData] = {}
        for subject_idx in range(15):
            trial_x = [np.asarray(value) for value in all_data[session_idx][subject_idx]]
            trial_y = [np.asarray(value, dtype=np.int64).reshape(-1) for value in all_labels[session_idx][subject_idx]]
            if len(trial_x) != 24 or len(trial_y) != 24:
                raise ValueError(f"expected 24 trials for session={session}, subject={subject_idx + 1}")
            flat_x = np.concatenate(trial_x, axis=0)
            flat_y = np.concatenate(trial_y, axis=0)
            # Match the C1 helper exactly: subject-wise z-score over all flattened
            # windows and all three time positions, separately for each session.
            normalized = np.stack(zscore_subject_wise([[flat_x]])[0][0], axis=0).astype(np.float32)
            trial_ids = np.concatenate([
                np.full(len(tx), trial_idx + 1, dtype=np.int8) for trial_idx, tx in enumerate(trial_x)
            ])
            window_starts = np.concatenate([
                np.arange(len(tx), dtype=np.int32) * int(prep["stride"]) for tx in trial_x
            ])
            sample_ids = build_sample_ids(session, subject_idx + 1, trial_ids, window_starts)
            if len(normalized) != len(flat_y) or len(normalized) != len(sample_ids):
                raise ValueError("feature/label/time-index lengths differ")
            subject_id = subject_idx + 1
            session_dir = Path(path_mapper["seediv_de_lds"]) / "eeg_feature_smooth" / str(session)
            source_files = list(session_dir.glob(f"{subject_id}_*.mat"))
            if len(source_files) != 1:
                raise FileNotFoundError(f"expected one source MAT file for S{subject_id} in {session_dir}, found {len(source_files)}")
            subjects[subject_idx + 1] = SubjectData(
                subject_id=subject_idx + 1,
                x=normalized,
                y=flat_y.astype(np.int64),
                trial_ids=trial_ids,
                window_starts=window_starts,
                sample_ids=sample_ids,
                input_fingerprint=array_fingerprint(normalized),
                raw_feature_fingerprint=array_fingerprint(flat_x),
                label_fingerprint=array_fingerprint(flat_y.astype(np.int64), trial_ids),
                source_file_path=str(source_files[0]),
                source_file_sha256=sha256_file(source_files[0]),
            )
        sessions[int(session)] = subjects
    return sessions


def _class_weights(labels: np.ndarray, classes: int, device: torch.device) -> torch.Tensor:
    counts = np.bincount(np.asarray(labels, dtype=np.int64), minlength=classes)
    values = len(labels) / (classes * np.maximum(counts, 1))
    return torch.as_tensor(values, dtype=torch.float32, device=device)


def _model_from_config(n_sources: int, config: dict[str, Any], device: torch.device) -> GeoSemSTDA:
    model_cfg = config["backbone"]
    model = GeoSemSTDA(
        n_sources=n_sources,
        num_electrodes=int(model_cfg["channels"]),
        num_freq_bands=int(model_cfg["frequency_bands"]),
        st_dim=int(model_cfg["st_dim"]),
        graph_dim=int(model_cfg["graph_dim"]),
        adapter_bottleneck=int(model_cfg["adapter_bottleneck"]),
        text_dim=512,
        heads=int(model_cfg["graph_heads"]),
        graph_heads=int(model_cfg["graph_heads"]),
        topk=int(model_cfg["topk"]),
        dropout=float(model_cfg["dropout"]),
        sample_length=int(model_cfg["sample_length"]),
        representation_mode=str(model_cfg["representation_mode"]),
        classifier_type=str(model_cfg["classifier_type"]),
        num_classes=4,
        cast_variant=str(model_cfg["cast_variant"]),
    ).to(device)
    return model


def _mechanism_from_state(state_dict: dict[str, torch.Tensor], n_sources: int, device: torch.device) -> MultiSourceAffectiveMetric:
    prototypes = state_dict["heads.0.prototypes"].detach().to(device)
    basis = state_dict["heads.0.basis"].detach().to(device)
    mechanism = MultiSourceAffectiveMetric(
        n_sources=n_sources,
        head_sharing="independent",
        variant="E0",
        context_dim=128,
        prototypes=prototypes,
        basis=basis,
        rank=2,
        hidden_dim=64,
        gamma=0.5,
        tau=0.07,
        lambda_regularization=1e-4,
    ).to(device)
    mechanism.load_state_dict(state_dict, strict=True)
    return mechanism


def _independent_e0_fixed_buffers(config: dict[str, Any], parent_state: dict[str, Any], device: torch.device):
    """Rebuild frozen CLIP prototypes and E0 semantic basis independently.

    The outer E0 checkpoint is not an OOF teacher initialization. Its immutable
    buffers are checked against independently reconstructed values so it can
    supply only the target-side frozen reference in this pipeline.
    """
    class_names = ["neutral", "sad", "fear", "happy"]
    text_dim, vectors = label_to_vector("seediv_de_lds", "clip", device=device)
    prototypes = torch.as_tensor(
        np.asarray([vectors[index] for index in range(4)], dtype=np.float32),
        dtype=torch.float32, device=device,
    )
    va_path = ROOT / config["semantic_basis_config"]
    va_config = json.loads(va_path.read_text(encoding="utf-8"))
    if not va_config.get("source") or not va_config.get("scale"):
        raise ValueError("reviewed valence-arousal basis provenance is missing")
    va_values = []
    for name in class_names:
        row = va_config.get("classes", {}).get(name, {})
        if row.get("valence") is None or row.get("arousal") is None:
            raise ValueError(f"semantic basis config lacks coordinates for {name}")
        va_values.append([row["valence"], row["arousal"]])
    va = torch.as_tensor(va_values, dtype=torch.float32, device=device)
    x = va - va.mean(dim=0, keepdim=True)
    normalized = F.normalize(prototypes, dim=-1)
    y = normalized - normalized.mean(dim=0, keepdim=True)
    directions = (torch.linalg.pinv(x) @ y).T
    basis = torch.linalg.qr(directions, mode="reduced").Q[:, :2]
    fixed = parent_state["mechanism"]
    if not torch.allclose(fixed["heads.0.prototypes"].to(device), F.normalize(prototypes, dim=-1), atol=1e-6, rtol=1e-6):
        raise RuntimeError("independently reconstructed CLIP prototypes disagree with parent E0 frozen buffer")
    if not torch.allclose(fixed["heads.0.basis"].to(device), basis, atol=1e-5, rtol=1e-5):
        raise RuntimeError("independently reconstructed semantic basis disagrees with parent E0 frozen buffer")
    return text_dim, prototypes, basis, sha256_file(va_path)


def _parent_bundle(session: int, target: int, seed: int, config: dict[str, Any], device: torch.device):
    checkpoint_path = parent_checkpoint_path(session, target, seed, ROOT)
    parent_sha = sha256_file(checkpoint_path)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = _model_from_config(14, config, device)
    model.load_state_dict(state["model"], strict=True)
    mechanism = _mechanism_from_state(state["mechanism"], 14, device)
    model.eval(); mechanism.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in mechanism.parameters():
        parameter.requires_grad_(False)
    return checkpoint_path, parent_sha, state, model, mechanism


def _make_loader(subject: SubjectData, batch_size: int, *, shuffle: bool = False, seed: int | None = None):
    x = torch.from_numpy(subject.x.astype(np.float32, copy=False))
    y = torch.from_numpy(subject.y.astype(np.int64, copy=False))
    r = torch.zeros((len(x), 1, 1), dtype=torch.float32)
    generator = None
    if shuffle and seed is not None:
        generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(x, r, y),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        drop_last=False,
        generator=generator,
    )


def _metrics_from_probabilities(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    p = np.asarray(probabilities, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pred = p.argmax(axis=-1)
    return {
        "accuracy": float(np.mean(pred == y)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2, 3], average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "cross_entropy": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)).mean()),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1, 2, 3]).tolist(),
        "predictions": pred,
    }


def _metrics_from_logits(logits: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    logits_t = torch.as_tensor(np.asarray(logits), dtype=torch.float64)
    probabilities = torch.softmax(logits_t, dim=-1).numpy()
    return _metrics_from_probabilities(probabilities, labels)


@torch.inference_mode()
def evaluate_parent_e0(model, mechanism, centroids, subject: SubjectData, session: int, device: torch.device, config: dict[str, Any]):
    loader = _make_loader(subject, int(config["backbone"]["batch_size"]))
    refs, experts, distances, weights_all = [], [], [], []
    tau = float(config["backbone"]["fusion_tau"])
    model.eval(); mechanism.eval()
    for x, r, _ in loader:
        x, r = x.to(device), r.to(device)
        _, z_all, _, h, _, _ = model([], [], x, r, return_features=True)
        branch_outputs = mechanism.source_outputs([h] * len(z_all), z_all)
        branch_logits = torch.stack([out.logits for out in branch_outputs], dim=0)
        z_stack = torch.stack(z_all, dim=0)
        dist = torch.linalg.vector_norm(z_stack - centroids[:, None, :], dim=-1)
        weight = torch.softmax(-dist / tau, dim=0)
        reference = (weight.unsqueeze(-1) * branch_logits).sum(dim=0)
        refs.append(reference.cpu().numpy())
        experts.append(branch_logits.permute(1, 0, 2).cpu().numpy())
        distances.append(dist.transpose(0, 1).cpu().numpy())
        weights_all.append(weight.transpose(0, 1).cpu().numpy())
    return {
        "reference_logits": np.concatenate(refs),
        "expert_logits": np.concatenate(experts),
        "source_distances": np.concatenate(distances),
        "fusion_weights": np.concatenate(weights_all),
    }


def _prototype_pair_features(mechanism: MultiSourceAffectiveMetric) -> np.ndarray:
    prototypes = F.normalize(mechanism.heads[0].prototypes.detach(), dim=-1)
    result = []
    pairs = class_pairs(prototypes.shape[0])
    for pair_index, (a, b) in enumerate(pairs):
        one_hot = np.eye(len(pairs), dtype=np.float32)[pair_index]
        cosine = float(torch.dot(prototypes[a], prototypes[b]).cpu())
        result.append(np.concatenate((one_hot, np.asarray([cosine], dtype=np.float32))))
    return np.asarray(result, dtype=np.float32)


@torch.inference_mode()
def export_evidence(model, mechanism, centroids, subjects: list[SubjectData], session: int, teacher_id: str,
                    source_subject_ids: list[int], device: torch.device, config: dict[str, Any]) -> Evidence:
    refs, experts, dists, labels = [], [], [], []
    subject_ids, session_ids, trial_ids, starts, sample_ids, teacher_ids = [], [], [], [], [], []
    pair_features = _prototype_pair_features(mechanism)
    tau = float(config["backbone"]["fusion_tau"])
    model.eval(); mechanism.eval()
    for subject in subjects:
        loader = _make_loader(subject, int(config["backbone"]["batch_size"]))
        local_refs, local_experts, local_dists = [], [], []
        for x, r, _ in loader:
            x, r = x.to(device), r.to(device)
            _, z_all, _, h, _, _ = model([], [], x, r, return_features=True)
            branch_outputs = mechanism.source_outputs([h] * len(z_all), z_all)
            branch_logits = torch.stack([out.logits for out in branch_outputs], dim=0)
            z_stack = torch.stack(z_all, dim=0)
            dist = torch.linalg.vector_norm(z_stack - centroids[:, None, :], dim=-1)
            weight = torch.softmax(-dist / tau, dim=0)
            reference = (weight.unsqueeze(-1) * branch_logits).sum(dim=0)
            local_refs.append(reference.cpu().numpy())
            local_experts.append(branch_logits.permute(1, 0, 2).cpu().numpy())
            local_dists.append(dist.transpose(0, 1).cpu().numpy())
        n = len(subject.x)
        refs.append(np.concatenate(local_refs)); experts.append(np.concatenate(local_experts)); dists.append(np.concatenate(local_dists))
        labels.append(subject.y)
        subject_ids.append(np.full(n, subject.subject_id, dtype=np.int16))
        session_ids.append(np.full(n, int(session), dtype=np.int8))
        trial_ids.append(subject.trial_ids)
        starts.append(subject.window_starts)
        sample_ids.append(subject.sample_ids)
        teacher_ids.append(np.full(n, teacher_id, dtype="U32"))
        if len(source_subject_ids) != centroids.shape[0]:
            raise ValueError("source subject order does not match teacher adapter/centroid count")
    return Evidence(
        reference_logits=np.concatenate(refs).astype(np.float32),
        expert_logits=np.concatenate(experts).astype(np.float32),
        labels=np.concatenate(labels).astype(np.int64),
        source_distances=np.concatenate(dists).astype(np.float32),
        prototype_pair_features=pair_features,
        subject_ids=np.concatenate(subject_ids),
        session_ids=np.concatenate(session_ids),
        trial_ids=np.concatenate(trial_ids),
        window_starts=np.concatenate(starts),
        sample_ids=np.concatenate(sample_ids),
        teacher_ids=np.concatenate(teacher_ids),
    )


def _evidence_path(cache_root: Path, role: str) -> Path:
    return cache_root / f"{role}_evidence.npz"


def _save_evidence(path: Path, evidence: Evidence) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **evidence.arrays())
    tmp.replace(path)
    return sha256_file(path)


def _load_evidence(path: Path) -> Evidence:
    data = np.load(path, allow_pickle=False)
    return Evidence(**{key: data[key] for key in Evidence.__dataclass_fields__})


def _subject_input_hash(subjects: dict[int, SubjectData], ids: list[int]) -> str:
    return json_fingerprint({str(sid): {
        "sample_ids": sha256_file_bytes(subjects[sid].sample_ids.astype("U32").tobytes()),
        "features": subjects[sid].input_fingerprint,
        "raw_features": subjects[sid].raw_feature_fingerprint,
        "labels": subjects[sid].label_fingerprint,
        "source_file_sha256": subjects[sid].source_file_sha256,
    } for sid in ids})


def sha256_file_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _teacher_loaders(subjects: dict[int, SubjectData], train_ids: list[int], batch_size: int, seed: int):
    loaders, centroid_loaders, weights = [], [], []
    for subject_id in train_ids:
        subject = subjects[subject_id]
        # Match the parent runner's explicit source-specific sampler seed
        # (source IDs there are zero-based).
        loader = _make_loader(subject, batch_size, shuffle=True, seed=seed + subject_id - 1)
        loaders.append(loader)
        centroid_loaders.append(_make_loader(subject, batch_size, shuffle=False))
        weights.append(_class_weights(subject.y, 4, torch.device("cpu")))
    return loaders, centroid_loaders, weights


def train_teacher(
    *, session: int, outer_target: int, seed: int, teacher_id: str, train_ids: list[int],
    eval_ids: list[int], subjects: dict[int, SubjectData], parent_state: dict[str, Any],
    config: dict[str, Any], device: torch.device, output_dir: Path,
    implementation_hashes: dict[str, str],
) -> tuple[Evidence, dict[str, Any]]:
    """Train fixed-epoch E0 source teacher; evaluation labels do not select it."""
    split = inner_split(session, outer_target, split_seed=int(config["inner_oof"]["split_seed"]))
    validate_split(split)
    train_set, eval_set = set(train_ids), set(eval_ids)
    if not train_set or train_set & eval_set or outer_target in train_set | eval_set:
        raise ValueError("invalid teacher train/evaluation subject partition")
    if train_set & set(split["source_dev_subject_ids"]):
        # Only the dedicated D-teacher may evaluate on D and it trains on R.
        raise ValueError("source-dev D leaked into a teacher training set")
    if not train_set <= set(split["router_oof_pool_subject_ids"]):
        raise ValueError("teacher train subjects must be drawn from R")
    if len(train_ids) not in (8, 12):
        raise ValueError("OOF teachers must train on 8 people and D teacher on R=12")
    if len(train_ids) == 8 and len(eval_ids) != 4:
        raise ValueError("an OOF teacher must evaluate exactly its four heldout subjects")
    if len(train_ids) == 12 and (train_set != set(split["router_oof_pool_subject_ids"]) or eval_set != set(split["source_dev_subject_ids"])):
        raise ValueError("the source-dev teacher must train on all R and evaluate only on D")
    if not eval_set.isdisjoint(train_set) or set(eval_ids) & set(split["source_dev_subject_ids"]) and len(train_ids) == 8:
        raise ValueError("heldout/dev leakage detected")

    setup_seed(int(seed))
    model = _model_from_config(len(train_ids), config, device)
    text_dim, prototypes, basis, basis_config_sha256 = _independent_e0_fixed_buffers(config, parent_state, device)
    if int(text_dim) != int(config["backbone"].get("text_dim", 512)):
        raise RuntimeError(f"unexpected CLIP text dimension: {text_dim}")
    metric_cfg = config["source_metric"]
    mechanism = MultiSourceAffectiveMetric(
        n_sources=len(train_ids), head_sharing=str(metric_cfg["head_sharing"]),
        variant=str(metric_cfg["variant"]), context_dim=int(config["backbone"]["st_dim"]),
        prototypes=prototypes, basis=basis, rank=int(metric_cfg["rank"]),
        hidden_dim=int(metric_cfg["metric_hidden_dim"]), gamma=float(metric_cfg["gamma"]),
        bound_eps=float(metric_cfg["metric_epsilon"]), tau=float(metric_cfg["tau"]),
        lambda_regularization=float(metric_cfg["lambda_H"]),
    ).to(device)
    # An E0 teacher includes the trainable source adapters/metric heads as well
    # as the shared encoder.  Freezing (or omitting) the mechanism optimizer
    # would no longer be the parent E0 model family.
    model.train(); mechanism.train()
    loaders, centroid_loaders, class_weights_cpu = _teacher_loaders(
        subjects, train_ids, int(config["backbone"]["batch_size"]), int(seed)
    )
    for subject_id, loader in zip(train_ids, loaders):
        loaded_x, _, loaded_y = loader.dataset.tensors
        subject = subjects[subject_id]
        if len(loaded_x) != len(subject.sample_ids) or not np.array_equal(loaded_y.cpu().numpy(), subject.y):
            raise RuntimeError(f"teacher loader sample/label set differs from recorded source subject {subject_id}")
        if not np.array_equal(loaded_x.cpu().numpy(), subject.x):
            raise RuntimeError(f"teacher loader feature rows differ from recorded source subject {subject_id}")
    class_weights = [value.to(device) for value in class_weights_cpu]
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(mechanism.parameters()),
        lr=float(config["backbone"]["learning_rate"]),
        weight_decay=float(config["backbone"]["weight_decay"]),
    )
    steps = min(len(loader) for loader in loaders)
    iterators = [iter(loader) for loader in loaders]
    train_log = []
    start_time = time.time()
    epochs = int(config["inner_oof"]["teacher_epochs"])
    for epoch in range(1, epochs + 1):
        model.train(); mechanism.train()
        losses = []
        for _ in range(steps):
            batches = []
            for i, iterator in enumerate(iterators):
                try:
                    batches.append(next(iterator))
                except StopIteration:
                    iterators[i] = iter(loaders[i])
                    batches.append(next(iterators[i]))
            xs = [batch[0].to(device) for batch in batches]
            rs = [batch[1].to(device) for batch in batches]
            ys = [batch[2].to(device) for batch in batches]
            optimizer.zero_grad(set_to_none=True)
            z_sources, _, h_sources, _, _, _ = model(xs, rs, return_features=True)
            outputs = mechanism.source_outputs(h_sources, z_sources)
            per_source = [
                F.cross_entropy(out.logits, labels, weight=class_weights[i]) + out.regularization
                for i, (out, labels) in enumerate(zip(outputs, ys))
            ]
            loss = torch.stack(per_source).mean()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_log.append({"epoch": epoch, "mean_source_ce": float(np.mean(losses))})
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"teacher={teacher_id} epoch={epoch}/{epochs} train_ce={np.mean(losses):.6f}", flush=True)

    model.eval(); mechanism.eval()
    with torch.inference_mode():
        centroids = compute_source_domain_centroids(model, centroid_loaders, device)
    evidence = export_evidence(
        model, mechanism, centroids,
        [subjects[sid] for sid in eval_ids], session, teacher_id, train_ids, device, config,
    )
    teacher_dir = output_dir / "teachers"
    teacher_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = teacher_dir / f"{teacher_id}.pt"
    sample_set_path = teacher_dir / f"{teacher_id}_sample_sets.npz"
    train_sample_ids = np.concatenate([subjects[sid].sample_ids for sid in train_ids]).astype("U32")
    train_sample_subject_ids = np.concatenate([
        np.full(len(subjects[sid].sample_ids), sid, dtype=np.int16) for sid in train_ids
    ])
    train_sample_labels = np.concatenate([subjects[sid].y for sid in train_ids]).astype(np.int64)
    evaluation_sample_ids = np.concatenate([subjects[sid].sample_ids for sid in eval_ids]).astype("U32")
    evaluation_sample_subject_ids = np.concatenate([
        np.full(len(subjects[sid].sample_ids), sid, dtype=np.int16) for sid in eval_ids
    ])
    evaluation_sample_labels = np.concatenate([subjects[sid].y for sid in eval_ids]).astype(np.int64)
    np.savez_compressed(
        sample_set_path,
        train_sample_ids=train_sample_ids,
        train_subject_ids=train_sample_subject_ids,
        train_labels=train_sample_labels,
        evaluation_sample_ids=evaluation_sample_ids,
        evaluation_subject_ids=evaluation_sample_subject_ids,
        evaluation_labels=evaluation_sample_labels,
    )
    sample_set_sha256 = sha256_file(sample_set_path)
    state = {
        "model": model.state_dict(),
        "mechanism": mechanism.state_dict(),
        "centroids": centroids.detach().cpu(),
        "epoch": epochs,
        "optimizer": optimizer.state_dict(),
        "teacher_id": teacher_id,
        "config_hash": c2_config_hash(config),
        "parent_config_hash": PARENT_CONFIG_HASH,
        "seed": int(seed),
        "session": int(session),
        "outer_target": int(outer_target),
        "train_subject_ids": list(train_ids),
        "evaluation_subject_ids": list(eval_ids),
        "prototype_sha256": array_fingerprint(prototypes.detach().cpu().numpy()),
        "semantic_basis_config_path": config["semantic_basis_config"],
        "semantic_basis_config_sha256": basis_config_sha256,
        "fixed_buffers_reconstructed_independently": True,
        "implementation_hashes": implementation_hashes,
        "sample_set_manifest_path": str(sample_set_path.relative_to(ROOT)),
        "sample_set_manifest_sha256": sample_set_sha256,
    }
    torch.save(state, ckpt_path)
    checkpoint_sha = sha256_file(ckpt_path)
    provenance = {
        "teacher_id": teacher_id,
        "teacher_type": "oof_8_source" if len(train_ids) == 8 else "source_dev_12_source",
        "session": int(session),
        "outer_target": int(outer_target),
        "seed": int(seed),
        "teacher_seed": int(seed),
        "fixed_epochs": epochs,
        "implementation_hashes": implementation_hashes,
        "checkpoint_selection": "fixed final epoch; heldout labels were not used for checkpoint selection",
        "teacher_train_subject_ids": list(train_ids),
        "evaluation_subject_ids": list(eval_ids),
        "actual_fit_subject_ids": list(train_ids),
        "normalization_fit_scope": "per-subject self-zscore; no shared statistics; each evaluated subject only contributes its own unlabeled normalization moments",
        "train_sample_count": int(sum(len(subjects[sid].y) for sid in train_ids)),
        "evaluation_sample_count": int(sum(len(subjects[sid].y) for sid in eval_ids)),
        "actual_sample_set_manifest_path": str(sample_set_path.relative_to(ROOT)),
        "actual_sample_set_manifest_sha256": sample_set_sha256,
        "train_sample_ids_sha256": json_fingerprint({str(sid): subjects[sid].sample_ids.tolist() for sid in train_ids}),
        "train_feature_label_fingerprint": _subject_input_hash(subjects, train_ids),
        "evaluation_sample_ids_sha256": json_fingerprint({sid: subjects[sid].sample_ids.tolist() for sid in eval_ids}),
        "evaluation_feature_fingerprints": {str(sid): subjects[sid].input_fingerprint for sid in eval_ids},
        "evaluation_label_fingerprints": {str(sid): subjects[sid].label_fingerprint for sid in eval_ids},
        "evaluation_source_files": {str(sid): {"path": subjects[sid].source_file_path, "sha256": subjects[sid].source_file_sha256} for sid in eval_ids},
        "source_files": {str(sid): {"path": subjects[sid].source_file_path, "sha256": subjects[sid].source_file_sha256} for sid in train_ids},
        "checkpoint_path": str(ckpt_path.relative_to(ROOT)),
        "checkpoint_sha256": checkpoint_sha,
        "training_log_path": str((teacher_dir / f"{teacher_id}.jsonl").relative_to(ROOT)),
        "training_elapsed_seconds": time.time() - start_time,
        "optimizer": "Adam",
        "learning_rate": float(config["backbone"]["learning_rate"]),
        "weight_decay": float(config["backbone"]["weight_decay"]),
        "batch_size": int(config["backbone"]["batch_size"]),
        "steps_per_epoch": steps,
        "teacher_source_count": len(train_ids),
        "excluded_outer_target": int(outer_target),
        "excluded_source_dev_from_oof_fit": list(split["source_dev_subject_ids"]),
        "router_training_graph_includes_teacher": False,
        "rows": int(len(evidence.labels)),
    }
    with (teacher_dir / f"{teacher_id}.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in train_log:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    del model, mechanism, optimizer, loaders, centroid_loaders
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return evidence, provenance


def _concat_evidence(items: list[Evidence]) -> Evidence:
    if not items:
        raise ValueError("cannot concatenate an empty evidence list")
    pair_features = items[0].prototype_pair_features
    for item in items[1:]:
        if not np.array_equal(pair_features, item.prototype_pair_features):
            raise ValueError("teacher CLIP prototype-pair features differ within one outer fold")
    return Evidence(
        reference_logits=np.concatenate([x.reference_logits for x in items]),
        expert_logits=np.concatenate([x.expert_logits for x in items]),
        labels=np.concatenate([x.labels for x in items]),
        source_distances=np.concatenate([x.source_distances for x in items]),
        prototype_pair_features=pair_features,
        subject_ids=np.concatenate([x.subject_ids for x in items]),
        session_ids=np.concatenate([x.session_ids for x in items]),
        trial_ids=np.concatenate([x.trial_ids for x in items]),
        window_starts=np.concatenate([x.window_starts for x in items]),
        sample_ids=np.concatenate([x.sample_ids for x in items]),
        teacher_ids=np.concatenate([x.teacher_ids for x in items]),
    )


def _evidence_from_cache(cache_dir: Path, role: str) -> Evidence:
    return _load_evidence(_evidence_path(cache_dir, role))


def _ensure_fold_cache(
    *, session: int, target: int, seed: int, subjects: dict[int, SubjectData], parent_state: dict[str, Any],
    config: dict[str, Any], device: torch.device, cache_dir: Path, run_manifest: dict[str, Any],
) -> tuple[Evidence, Evidence, dict[str, Any]]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "evidence_manifest.json"
    oof_path = _evidence_path(cache_dir, "oof")
    dev_path = _evidence_path(cache_dir, "source_dev")
    split = inner_split(session, target, split_seed=int(config["inner_oof"]["split_seed"]))
    validate_split(split)
    split_path = cache_dir / "split_manifest.json"
    if split_path.exists():
        previous_split = json.loads(split_path.read_text(encoding="utf-8"))
        if previous_split != split:
            raise RuntimeError(f"cached split does not equal the frozen protocol: {split_path}")
    else:
        _write_json(split_path, split)

    if manifest_path.exists() and oof_path.exists() and dev_path.exists():
        cached = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_identity = {
            "config_hash": c2_config_hash(config),
            "parent_config_hash": PARENT_CONFIG_HASH,
            "parent_checkpoint_sha256": run_manifest["parent_checkpoint_sha256"],
            "session": int(session),
            "outer_target": int(target),
            "seed": int(seed),
            "split": split,
            "implementation_hashes": run_manifest["implementation_hashes"],
        }
        if any(cached.get(key) != value for key, value in expected_identity.items()):
            raise RuntimeError("existing evidence cache identity/provenance does not match this run")
        if cached.get("oof_evidence_sha256") != sha256_file(oof_path) or cached.get("source_dev_evidence_sha256") != sha256_file(dev_path):
            raise RuntimeError("evidence cache file hash mismatch; refusing to reuse it")
        for record in cached.get("teacher_records", []):
            teacher_checkpoint = ROOT / record["checkpoint_path"]
            if not teacher_checkpoint.is_file() or sha256_file(teacher_checkpoint) != record.get("checkpoint_sha256"):
                raise RuntimeError(f"cached teacher checkpoint missing or hash-mismatched: {teacher_checkpoint}")
            sample_set_manifest = ROOT / record["actual_sample_set_manifest_path"]
            if not sample_set_manifest.is_file() or sha256_file(sample_set_manifest) != record.get("actual_sample_set_manifest_sha256"):
                raise RuntimeError(f"cached exact teacher sample-set manifest missing or hash-mismatched: {sample_set_manifest}")
        return _evidence_from_cache(cache_dir, "oof"), _evidence_from_cache(cache_dir, "source_dev"), cached

    teacher_records: list[dict[str, Any]] = []
    oof_items: list[Evidence] = []
    for fold in split["oof_folds"]:
        fold_id = int(fold["fold_id"])
        train_ids = sorted(map(int, fold["teacher_train_subject_ids"]))
        eval_ids = sorted(map(int, fold["heldout_subject_ids"]))
        teacher_id = f"oof_fold_{fold_id}"
        evidence, record = train_teacher(
            session=session, outer_target=target, seed=seed, teacher_id=teacher_id,
            train_ids=train_ids, eval_ids=eval_ids, subjects=subjects,
            parent_state=parent_state, config=config, device=device, output_dir=cache_dir,
            implementation_hashes=run_manifest["implementation_hashes"],
        )
        record.update({"fold_id": fold_id, "fold_subjects": fold})
        teacher_records.append(record)
        oof_items.append(evidence)
    oof_evidence = _concat_evidence(oof_items)

    dev_train_ids = sorted(map(int, split["dev_teacher_train_subject_ids"]))
    dev_ids = sorted(map(int, split["source_dev_subject_ids"]))
    dev_evidence, dev_record = train_teacher(
        session=session, outer_target=target, seed=seed, teacher_id="source_dev_teacher",
        train_ids=dev_train_ids, eval_ids=dev_ids, subjects=subjects,
        parent_state=parent_state, config=config, device=device, output_dir=cache_dir,
        implementation_hashes=run_manifest["implementation_hashes"],
    )
    dev_record["fold_id"] = "source_dev"
    teacher_records.append(dev_record)
    oof_sha = _save_evidence(oof_path, oof_evidence)
    dev_sha = _save_evidence(dev_path, dev_evidence)
    payload = {
        "status": "PASS",
        "config_hash": c2_config_hash(config),
        "parent_config_hash": PARENT_CONFIG_HASH,
        "parent_checkpoint_sha256": run_manifest["parent_checkpoint_sha256"],
        "session": int(session),
        "outer_target": int(target),
        "seed": int(seed),
        "split": split,
        "implementation_hashes": run_manifest["implementation_hashes"],
        "oof_teacher_count": 3,
        "source_dev_teacher_count": 1,
        "teacher_records": teacher_records,
        "oof_evidence_sha256": oof_sha,
        "source_dev_evidence_sha256": dev_sha,
        "oof_rows": int(len(oof_evidence.labels)),
        "source_dev_rows": int(len(dev_evidence.labels)),
        "oof_subjects_in_router_train": sorted(set(map(int, oof_evidence.subject_ids))),
        "source_dev_subjects_are_not_router_train": not bool(set(dev_ids) & set(map(int, oof_evidence.subject_ids))),
        "outer_target_excluded_from_all_teacher_fit": all(target not in r["actual_fit_subject_ids"] for r in teacher_records),
        "d_excluded_from_oof_teacher_fit": all(
            not (set(dev_ids) & set(r["actual_fit_subject_ids"])) for r in teacher_records if r["teacher_type"] == "oof_8_source"
        ),
        "teacher_training_sample_fingerprints": {r["teacher_id"]: r["train_feature_label_fingerprint"] for r in teacher_records},
        "data_fingerprints": {
            str(sid): {
                "features_sha256": subjects[sid].input_fingerprint,
                "raw_features_sha256": subjects[sid].raw_feature_fingerprint,
                "labels_and_trials_sha256": subjects[sid].label_fingerprint,
                "source_file_path": subjects[sid].source_file_path,
                "source_file_sha256": subjects[sid].source_file_sha256,
            }
            for sid in split["source_subject_ids"] + [target]
        },
        "normalization": config["preprocessing"],
        "notes": "OOF utility labels use each R subject's true labels only after its teacher excluded that subject; D is router source-dev only; outer target labels are not in these caches.",
    }
    if not payload["outer_target_excluded_from_all_teacher_fit"] or not payload["d_excluded_from_oof_teacher_fit"] or not payload["source_dev_subjects_are_not_router_train"]:
        raise RuntimeError("actual teacher/cache isolation audit failed")
    _write_json(manifest_path, payload)
    return oof_evidence, dev_evidence, payload


def _current_features(evidence: Evidence, device: torch.device, row_slice: slice | None = None) -> torch.Tensor:
    sl = row_slice or slice(None)
    ref = torch.as_tensor(evidence.reference_logits[sl], dtype=torch.float32, device=device)
    exp = torch.as_tensor(evidence.expert_logits[sl], dtype=torch.float32, device=device)
    pair = torch.as_tensor(evidence.prototype_pair_features, dtype=torch.float32, device=device)
    dist = torch.as_tensor(evidence.source_distances[sl], dtype=torch.float32, device=device)
    return build_router_features(ref, exp, pair, dist)


def _history_features(evidence: Evidence, config: dict[str, Any], *, matched: bool = False) -> np.ndarray:
    return build_history_features(
        evidence.reference_logits,
        evidence.expert_logits,
        evidence.subject_ids,
        evidence.session_ids,
        evidence.trial_ids,
        evidence.window_starts,
        alpha=float(config["router"]["history_alpha"]),
        matched_current=matched,
    )


def _build_features(evidence: Evidence, rows: slice, method: str, config: dict[str, Any], device: torch.device,
                    history_cache: np.ndarray | None = None, matched_cache: np.ndarray | None = None) -> torch.Tensor:
    current = _current_features(evidence, device, rows)
    if method == "C2-history":
        if history_cache is None:
            history_cache = _history_features(evidence, config)
        history = torch.as_tensor(history_cache[rows], dtype=current.dtype, device=device)
        return append_history_features(current, history)
    if method == "C2-current-matched":
        if matched_cache is None:
            matched_cache = _history_features(evidence, config, matched=True)
        matched = torch.as_tensor(matched_cache[rows], dtype=current.dtype, device=device)
        return append_history_features(current, matched)
    return current


def _fit_standardizer(router: PairwiseUtilityRouter, evidence: Evidence, method: str,
                      config: dict[str, Any], device: torch.device, *, chunk_size: int = 256) -> None:
    history = _history_features(evidence, config) if method == "C2-history" else None
    matched = _history_features(evidence, config, matched=True) if method == "C2-current-matched" else None
    total = torch.zeros(router.standardizer.mean.numel(), dtype=torch.float64, device=device)
    total_sq = torch.zeros_like(total)
    count = 0
    for start in range(0, len(evidence.labels), chunk_size):
        rows = slice(start, min(start + chunk_size, len(evidence.labels)))
        features = _build_features(evidence, rows, method, config, device, history, matched)
        flat = features.reshape(-1, features.shape[-1]).to(torch.float64)
        total += flat.sum(dim=0)
        total_sq += flat.square().sum(dim=0)
        count += int(flat.shape[0])
    if not count:
        raise ValueError("router training evidence is empty")
    mean = total / count
    variance = (total_sq / count - mean.square()).clamp_min(1e-12)
    router.standardizer.mean.copy_(mean.to(router.standardizer.mean.dtype))
    router.standardizer.scale.copy_(torch.sqrt(variance).clamp_min(1e-6).to(router.standardizer.scale.dtype))
    router.standardizer.fitted.fill_(True)


def _utility_targets(evidence: Evidence, rows: slice, cfg: dict[str, Any], device: torch.device) -> torch.Tensor:
    ref = torch.as_tensor(evidence.reference_logits[rows], dtype=torch.float32, device=device)
    exp = torch.as_tensor(evidence.expert_logits[rows], dtype=torch.float32, device=device)
    labels = torch.as_tensor(evidence.labels[rows], dtype=torch.long, device=device)
    return utility_labels(
        ref, exp, labels,
        alpha=float(cfg["router"]["action_alpha"]),
        clip_limit=float(cfg["router"]["residual_clip"]),
    )


def _sample_weighted_loss(predicted: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
    # K and six class pairs are constant inside one evidence package. Averaging
    # over candidates therefore assigns equal total weight to each sample.
    if predicted.shape != target.shape:
        raise ValueError("router predictions and utility labels must have identical shape")
    if loss_name == "MSE":
        per_candidate = (predicted - target).square()
    elif loss_name == "Huber":
        per_candidate = F.huber_loss(predicted, target, reduction="none")
    else:
        raise ValueError(f"unsupported router loss: {loss_name}")
    return per_candidate.mean(dim=(-2, -1)).mean()


@torch.inference_mode()
def _action_payload(evidence: Evidence, router: PairwiseUtilityRouter, config: dict[str, Any], device: torch.device,
                    method: str, history: np.ndarray | None = None, matched: np.ndarray | None = None,
                    chunk_size: int = 256) -> dict[str, np.ndarray]:
    predicted_rows, logits_rows = [], []
    action_codes, predicted_values = [], []
    residual_rows, actual_values, oracle_values = [], [], []
    oracle_logits_rows, oracle_codes, oracle_values_selected = [], [], []
    pairs = class_pairs(4)
    for start in range(0, len(evidence.labels), chunk_size):
        rows = slice(start, min(start + chunk_size, len(evidence.labels)))
        features = _build_features(evidence, rows, method, config, device, history, matched)
        predicted = router(features).detach()
        ref = torch.as_tensor(evidence.reference_logits[rows], dtype=torch.float32, device=device)
        exp = torch.as_tensor(evidence.expert_logits[rows], dtype=torch.float32, device=device)
        residual = pairwise_residuals(ref, exp, float(config["router"]["residual_clip"]))
        adjusted, action = apply_best_single_action(
            ref, residual, predicted, alpha=float(config["router"]["action_alpha"])
        )
        pred_index = predicted.reshape(len(ref), -1).argmax(dim=-1)
        max_value = predicted.reshape(len(ref), -1).gather(1, pred_index[:, None]).squeeze(1)
        max_value = torch.where(max_value > 0, max_value, torch.zeros_like(max_value))
        # Labels are deliberately read only after the complete inference action
        # has been chosen; they are used below for realized-utility reporting.
        y = torch.as_tensor(evidence.labels[rows], dtype=torch.long, device=device)
        actual_candidates = utility_labels(
            ref, exp, y,
            alpha=float(config["router"]["action_alpha"]),
            clip_limit=float(config["router"]["residual_clip"]),
        )
        actual = actual_candidates.reshape(len(ref), -1).gather(1, pred_index[:, None]).squeeze(1)
        actual = torch.where(action > 0, actual, torch.zeros_like(actual))
        # Diagnostic-only upper bound: choose using realized CE utility after
        # the deployable predicted-utility action has already been fixed.
        oracle_index = actual_candidates.reshape(len(ref), -1).argmax(dim=-1)
        oracle_max = actual_candidates.reshape(len(ref), -1).gather(1, oracle_index[:, None]).squeeze(1)
        oracle_logits, oracle_code = apply_best_single_action(
            ref, residual, actual_candidates, alpha=float(config["router"]["action_alpha"])
        )
        oracle_selected = torch.where(oracle_code > 0, oracle_max, torch.zeros_like(oracle_max))
        predicted_rows.append(predicted.cpu().numpy())
        logits_rows.append(adjusted.cpu().numpy())
        action_codes.append(action.cpu().numpy())
        oracle_logits_rows.append(oracle_logits.cpu().numpy())
        oracle_codes.append(oracle_code.cpu().numpy())
        oracle_values_selected.append(oracle_selected.cpu().numpy())
        predicted_values.append(max_value.cpu().numpy())
        actual_values.append(actual.cpu().numpy())
        oracle_values.append(actual_candidates.cpu().numpy())
        residual_rows.append(residual.cpu().numpy())
    codes = np.concatenate(action_codes).astype(np.int32)
    edge_count = len(pairs)
    active = codes > 0
    source_index = np.where(active, (codes - 1) // edge_count, -1).astype(np.int16)
    pair_index = np.where(active, (codes - 1) % edge_count, -1).astype(np.int8)
    return {
        "predicted_utility": np.concatenate(predicted_rows).astype(np.float32),
        "action_logits": np.concatenate(logits_rows).astype(np.float32),
        "oracle_action_logits": np.concatenate(oracle_logits_rows).astype(np.float32),
        "oracle_action_code": np.concatenate(oracle_codes).astype(np.int32),
        "oracle_source_index": np.where(np.concatenate(oracle_codes) > 0, (np.concatenate(oracle_codes) - 1) // len(pairs), -1).astype(np.int16),
        "oracle_pair_index": np.where(np.concatenate(oracle_codes) > 0, (np.concatenate(oracle_codes) - 1) % len(pairs), -1).astype(np.int8),
        "oracle_realized_selected_utility": np.concatenate(oracle_values_selected).astype(np.float32),
        "action_code": codes,
        "source_index": source_index,
        "pair_index": pair_index,
        "predicted_selected_utility": np.concatenate(predicted_values).astype(np.float32),
        "realized_selected_utility": np.concatenate(actual_values).astype(np.float32),
        "oracle_candidate_utility": np.concatenate(oracle_values).astype(np.float32),
        "residuals": np.concatenate(residual_rows).astype(np.float32),
    }


def _evaluate_probabilities(probabilities: np.ndarray, labels: np.ndarray, base_probabilities: np.ndarray | None = None,
                            action_codes: np.ndarray | None = None, predicted_utility: np.ndarray | None = None,
                            realized_utility: np.ndarray | None = None, oracle_utility: np.ndarray | None = None,
                            subject_ids: np.ndarray | None = None) -> dict[str, Any]:
    scored = _metrics_from_probabilities(probabilities, labels)
    predictions = scored.pop("predictions")
    scored["action_rate"] = float(np.mean(action_codes > 0)) if action_codes is not None else 0.0
    scored["no_op_rate"] = 1.0 - scored["action_rate"]
    if subject_ids is not None:
        subject_ids = np.asarray(subject_ids)
        per_subject = {
            str(int(subject)): float(np.mean(predictions[subject_ids == subject] == np.asarray(labels)[subject_ids == subject]))
            for subject in np.unique(subject_ids)
        }
        scored["mean_subject_accuracy"] = float(np.mean(list(per_subject.values())))
        scored["per_subject_accuracy"] = per_subject
    if realized_utility is not None:
        active = action_codes > 0
        scored["mean_realized_utility"] = float(np.mean(realized_utility))
        scored["negative_utility_rate"] = float(np.mean(realized_utility[active] < 0)) if active.any() else 0.0
        scored["mean_predicted_selected_utility"] = float(np.mean(predicted_utility))
    if base_probabilities is not None:
        base_pred = np.asarray(base_probabilities).argmax(axis=-1)
        scored["corrected_errors"] = int(np.sum((base_pred != labels) & (predictions == labels)))
        scored["harmed_correct"] = int(np.sum((base_pred == labels) & (predictions != labels)))
        scored["unchanged_correct"] = int(np.sum((base_pred == labels) & (predictions == labels)))
        scored["unchanged_wrong"] = int(np.sum((base_pred != labels) & (predictions != labels)))
        scored["base_accuracy"] = float(np.mean(base_pred == labels))
    if realized_utility is not None and oracle_utility is not None:
        oracle = np.maximum(np.asarray(oracle_utility).reshape(len(labels), -1).max(axis=-1), 0.0)
        scored["realized_oracle_gap"] = float(np.mean(oracle - realized_utility))
    return scored


def _evaluate_router(router: PairwiseUtilityRouter, evidence: Evidence, config: dict[str, Any], device: torch.device,
                     method: str, *, history: np.ndarray | None = None, matched: np.ndarray | None = None,
                     output_ema: bool = False) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    router.eval()
    payload = _action_payload(evidence, router, config, device, method, history, matched)
    action_prob = torch.softmax(torch.as_tensor(payload["action_logits"], dtype=torch.float64), dim=-1).numpy()
    if output_ema:
        action_prob = probability_ema(
            action_prob, evidence.subject_ids, evidence.session_ids, evidence.trial_ids, evidence.window_starts,
            alpha=float(config["router"]["output_ema_alpha"]),
        )
    base_prob = torch.softmax(torch.as_tensor(evidence.reference_logits, dtype=torch.float64), dim=-1).numpy()
    metrics = _evaluate_probabilities(
        action_prob,
        evidence.labels,
        base_probabilities=base_prob,
        action_codes=payload["action_code"],
        predicted_utility=payload["predicted_selected_utility"],
        realized_utility=payload["realized_selected_utility"],
        oracle_utility=payload["oracle_candidate_utility"],
        subject_ids=evidence.subject_ids,
    )
    oracle_prob = torch.softmax(torch.as_tensor(payload["oracle_action_logits"], dtype=torch.float64), dim=-1).numpy()
    metrics["oracle_single_action"] = _oracle_metrics(evidence, payload)
    return metrics, {**payload, "probabilities": action_prob.astype(np.float32), "oracle_probabilities": oracle_prob.astype(np.float32)}


def _parent_evidence_from_arrays(subject: SubjectData, session: int, outputs: dict[str, np.ndarray], pair_features: np.ndarray) -> Evidence:
    n = len(subject.y)
    return Evidence(
        reference_logits=outputs["reference_logits"].astype(np.float32),
        expert_logits=outputs["expert_logits"].astype(np.float32),
        labels=subject.y,
        source_distances=outputs["source_distances"].astype(np.float32),
        prototype_pair_features=pair_features,
        subject_ids=np.full(n, subject.subject_id, dtype=np.int16),
        session_ids=np.full(n, int(session), dtype=np.int8),
        trial_ids=subject.trial_ids,
        window_starts=subject.window_starts,
        sample_ids=subject.sample_ids,
        teacher_ids=np.full(n, "parent_E0", dtype="U32"),
    )


@torch.inference_mode()
def _oracle_from_evidence(evidence: Evidence, config: dict[str, Any], device: torch.device, chunk_size: int = 256):
    logits_rows, code_rows, utility_rows = [], [], []
    for start in range(0, len(evidence.labels), chunk_size):
        rows = slice(start, min(start + chunk_size, len(evidence.labels)))
        ref = torch.as_tensor(evidence.reference_logits[rows], dtype=torch.float32, device=device)
        experts = torch.as_tensor(evidence.expert_logits[rows], dtype=torch.float32, device=device)
        labels = torch.as_tensor(evidence.labels[rows], dtype=torch.long, device=device)
        residual = pairwise_residuals(ref, experts, float(config["router"]["residual_clip"]))
        utility = utility_labels(
            ref, experts, labels,
            alpha=float(config["router"]["action_alpha"]),
            clip_limit=float(config["router"]["residual_clip"]),
        )
        adjusted, action = apply_best_single_action(
            ref, residual, utility, alpha=float(config["router"]["action_alpha"])
        )
        best_value = utility.reshape(len(ref), -1).max(dim=-1).values
        selected = torch.where(action > 0, best_value, torch.zeros_like(best_value))
        logits_rows.append(adjusted.cpu().numpy())
        code_rows.append(action.cpu().numpy())
        utility_rows.append(selected.cpu().numpy())
    return {
        "oracle_action_logits": np.concatenate(logits_rows).astype(np.float32),
        "oracle_action_code": np.concatenate(code_rows).astype(np.int32),
        "oracle_source_index": np.where(np.concatenate(code_rows) > 0, (np.concatenate(code_rows) - 1) // len(class_pairs(4)), -1).astype(np.int16),
        "oracle_pair_index": np.where(np.concatenate(code_rows) > 0, (np.concatenate(code_rows) - 1) % len(class_pairs(4)), -1).astype(np.int8),
        "oracle_realized_selected_utility": np.concatenate(utility_rows).astype(np.float32),
    }


def _oracle_metrics(evidence: Evidence, oracle_payload: dict[str, np.ndarray]) -> dict[str, Any]:
    probabilities = torch.softmax(torch.as_tensor(oracle_payload["oracle_action_logits"], dtype=torch.float64), dim=-1).numpy()
    scored = _metrics_from_probabilities(probabilities, evidence.labels)
    predictions = scored.pop("predictions")
    base_pred = np.asarray(evidence.reference_logits).argmax(axis=-1)
    labels = np.asarray(evidence.labels)
    return {
        **scored,
        "action_rate": float(np.mean(oracle_payload["oracle_action_code"] > 0)),
        "no_op_rate": float(np.mean(oracle_payload["oracle_action_code"] == 0)),
        "corrected_errors": int(np.sum((base_pred != labels) & (predictions == labels))),
        "harmed_correct": int(np.sum((base_pred == labels) & (predictions != labels))),
        "mean_realized_selected_utility": float(np.mean(oracle_payload["oracle_realized_selected_utility"])),
        "label_use": "diagnostic upper bound only; not available to router inference or training decisions",
    }


def _write_fixed_method(method: str, evidence: Evidence, probabilities: np.ndarray, *, parent_metrics: dict[str, Any] | None,
                        run_dir: Path, parent_best_epoch: int, config: dict[str, Any], device: torch.device,
                        output_ema_alpha: float | None = None) -> dict[str, Any]:
    probs = np.asarray(probabilities, dtype=np.float32)
    metrics = _evaluate_probabilities(
        probs,
        evidence.labels,
        base_probabilities=torch.softmax(torch.as_tensor(evidence.reference_logits, dtype=torch.float64), dim=-1).numpy(),
        action_codes=np.zeros(len(evidence.labels), dtype=np.int32),
        subject_ids=evidence.subject_ids,
    )
    oracle_payload = _oracle_from_evidence(evidence, config, device)
    metrics["oracle_single_action"] = _oracle_metrics(evidence, oracle_payload)
    if parent_metrics is not None:
        metrics["parent_reproduction_delta"] = {
            key: float(metrics[key] - float(parent_metrics[key]))
            for key in ("accuracy", "macro_f1", "balanced_accuracy")
        }
        metrics["parent_reproduction_pass"] = all(
            abs(metrics[key] - float(parent_metrics[key])) <= 1e-10
            for key in ("accuracy", "macro_f1", "balanced_accuracy")
        ) and metrics["confusion_matrix"] == parent_metrics["confusion_matrix"]
        metrics["parent_reproduction_tolerance"] = {
            "accuracy": 1e-10,
            "macro_f1": 1e-10,
            "balanced_accuracy": 1e-10,
            "confusion_matrix": "exact integer match",
        }
        if not metrics["parent_reproduction_pass"]:
            metrics["parent_reproduction_difference_reason"] = (
                "Check data loader version/order, per-subject normalization, target batch order, checkpoint/model reconstruction, "
                "source centroid state, branch-logit fusion temperature, and torch/sklearn version; C2 pilot halted before teacher training."
            )
    if output_ema_alpha is not None:
        metrics["output_ema_alpha"] = float(output_ema_alpha)
    metrics["method"] = method
    metrics["parent_best_epoch"] = int(parent_best_epoch)
    metrics["evaluation_type"] = "frozen_parent_checkpoint_diagnostic"
    payload = {
        "probabilities": probs,
        "action_code": np.zeros(len(evidence.labels), dtype=np.int32),
        "source_index": np.full(len(evidence.labels), -1, dtype=np.int16),
        "pair_index": np.full(len(evidence.labels), -1, dtype=np.int8),
        **oracle_payload,
        "oracle_probabilities": torch.softmax(torch.as_tensor(oracle_payload["oracle_action_logits"], dtype=torch.float64), dim=-1).numpy().astype(np.float32),
    }
    _save_sample_outputs(run_dir, method, evidence, metrics, payload)
    _write_json(run_dir / "metrics.json", metrics)
    (run_dir / "COMPLETE").write_text("PASS\n", encoding="ascii")
    return metrics


def _distribution_report(oof: Evidence, dev: Evidence, target: Evidence, router: PairwiseUtilityRouter,
                         config: dict[str, Any], device: torch.device, output: Path) -> dict[str, Any]:
    report: dict[str, Any] = {"feature_dimensions": {}, "standardized_mean_absolute_shift_from_oof": {}}
    for name, evidence in (("K8_oof", oof), ("K12_source_dev", dev), ("K14_outer_target", target)):
        chunks = []
        for start in range(0, len(evidence.labels), 256):
            features = _current_features(evidence, device, slice(start, min(start + 256, len(evidence.labels))))
            chunks.append(features.detach().cpu().numpy().reshape(-1, features.shape[-1]))
        matrix = np.concatenate(chunks)
        report["feature_dimensions"][name] = {
            "candidate_rows": int(len(matrix)),
            "mean": matrix.mean(axis=0).tolist(),
            "std": matrix.std(axis=0).tolist(),
        }
        standardized_mean = (matrix.mean(axis=0) - router.standardizer.mean.detach().cpu().numpy()) / router.standardizer.scale.detach().cpu().numpy()
        report["standardized_mean_absolute_shift_from_oof"][name] = float(np.mean(np.abs(standardized_mean)))
    _write_json(output, report)
    return report


def execute_fold(
    *, session: int, target: int, seed: int, methods: list[str], subjects: dict[int, SubjectData],
    config: dict[str, Any], device: torch.device, config_hash: str,
    implementation_hashes: dict[str, str],
) -> list[dict[str, Any]]:
    fold_started_at = time.time()
    parent_path, parent_sha, parent_state, model, mechanism = _parent_bundle(session, target, seed, config, device)
    parent_metrics = json.loads((parent_path.parent / "metrics.json").read_text(encoding="utf-8"))
    centroids = parent_state["centroids"].to(device)
    target_subject = subjects[target]
    implementation_hash = json_fingerprint(implementation_hashes)
    outputs = evaluate_parent_e0(model, mechanism, centroids, target_subject, session, device, config)
    target_evidence = _parent_evidence_from_arrays(
        target_subject, session, outputs, _prototype_pair_features(mechanism)
    )
    parent_probabilities = torch.softmax(torch.as_tensor(outputs["reference_logits"], dtype=torch.float64), dim=-1).numpy()
    revision_root = ROOT / "results" / "seediv_c2_e0_s1s2s3" / config_hash / f"impl_{implementation_hash[:12]}"
    target_cache_root = revision_root / "shared_target_evidence"
    target_cache_path = target_cache_root / f"session{session}_target{target:02d}_seed{seed}.npz"
    target_cache_sha = _save_evidence(target_cache_path, target_evidence)

    fold_root = revision_root / f"session{session}_target{target:02d}" / f"seed{seed}"
    split = inner_split(session, target, split_seed=int(config["inner_oof"]["split_seed"]))
    run_manifest = {
        "config_hash": config_hash,
        "implementation_hashes": implementation_hashes,
        "implementation_hash": implementation_hash,
        "implementation_revision_dir": f"impl_{implementation_hash[:12]}",
        "run_hash": None,
        "parent_config_hash": PARENT_CONFIG_HASH,
        "parent_checkpoint_sha256": parent_sha,
        "parent_checkpoint_path": str(parent_path.relative_to(ROOT)),
        "parent_best_epoch": int(parent_state["epoch"]),
        "parent_metrics": {key: parent_metrics[key] for key in ("accuracy", "macro_f1", "balanced_accuracy", "confusion_matrix")},
        "session": int(session),
        "outer_target": int(target),
        "seed": int(seed),
        "outer_source_subject_ids": [sid for sid in range(1, 16) if sid != target],
        "data_fingerprint_target": {
            "features_sha256": target_subject.input_fingerprint,
            "labels_and_trials_sha256": target_subject.label_fingerprint,
        },
        "target_sample_id_sha256": json_fingerprint(target_subject.sample_ids.tolist()),
        "target_evidence_cache_path": str(target_cache_path.relative_to(ROOT)),
        "target_evidence_cache_sha256": target_cache_sha,
        "split": split,
        "normalization": config["preprocessing"]["normalization"],
        "whole_system_online_causal_claim": False,
        "device": str(device),
    }
    _write_json(fold_root / "parent_evaluation_manifest.json", run_manifest)

    recorded = []
    if "B0" in methods:
        b0_dir = fold_root / "B0"
        b0_manifest = dict(run_manifest)
        b0_manifest.update({"method": "B0", "run_hash": run_identity(config_hash, "B0", session, target, seed, parent_sha, implementation_hash)})
        _write_json(b0_dir / "run_manifest.json", b0_manifest)
        b0_metrics = _write_fixed_method(
            "B0", target_evidence, parent_probabilities, parent_metrics=parent_metrics,
            run_dir=b0_dir, parent_best_epoch=int(parent_state["epoch"]), config=config, device=device,
        )
        recorded.append({"method": "B0", **{k: v for k, v in b0_metrics.items() if k != "confusion_matrix"}})
        if not b0_metrics.get("parent_reproduction_pass", False):
            _write_json(fold_root / "PILOT_BLOCKED_B0_MISMATCH.json", {
                "status": "FAIL", "reason": b0_metrics.get("parent_reproduction_difference_reason"),
                "parent_metrics": parent_metrics, "recomputed_metrics": b0_metrics,
            })
            raise RuntimeError("B0 did not reproduce the corresponding E0 run; stopped before OOF teacher training")

    if "B1" in methods:
        b1_manifest = dict(run_manifest)
        b1_manifest.update({"method": "B1", "run_hash": run_identity(config_hash, "B1", session, target, seed, parent_sha, implementation_hash)})
        _write_json(fold_root / "B1" / "run_manifest.json", b1_manifest)
        b1_probabilities = probability_ema(
            parent_probabilities,
            target_evidence.subject_ids,
            target_evidence.session_ids,
            target_evidence.trial_ids,
            target_evidence.window_starts,
            alpha=float(config["router"]["output_ema_alpha"]),
        )
        b1_metrics = _write_fixed_method(
            "B1", target_evidence, b1_probabilities, parent_metrics=None,
            run_dir=fold_root / "B1", parent_best_epoch=int(parent_state["epoch"]),
            config=config, device=device,
            output_ema_alpha=float(config["router"]["output_ema_alpha"]),
        )
        recorded.append({"method": "B1", **{k: v for k, v in b1_metrics.items() if k != "confusion_matrix"}})

    cache_dir = fold_root / "shared_oof_cache"
    evidence_manifest_path = cache_dir / "evidence_manifest.json"
    if any(method not in ("B0", "B1") for method in methods):
        oof, dev, evidence_manifest = _ensure_fold_cache(
            session=session, target=target, seed=seed, subjects=subjects,
            parent_state=parent_state, config=config, device=device,
            cache_dir=cache_dir, run_manifest=run_manifest,
        )
        run_manifest["evidence_manifest_sha256"] = sha256_file(evidence_manifest_path)
        for method in methods:
            if method in ("B0", "B1"):
                continue
            method_dir = fold_root / method
            method_manifest = dict(run_manifest)
            method_manifest["method"] = method
            method_manifest["run_hash"] = run_identity(config_hash, method, session, target, seed, parent_sha, implementation_hash)
            method_manifest["teacher_checkpoint_sha256s_before_router"] = {
                r["teacher_id"]: r["checkpoint_sha256"] for r in evidence_manifest["teacher_records"]
            }
            _write_json(method_dir / "run_manifest.json", method_manifest)
            result = train_router_method(
                method=method, session=session, target=target, seed=seed,
                oof=oof, dev=dev, target_evidence=target_evidence,
                config=config, device=device, run_dir=method_dir,
                run_manifest=method_manifest,
            )
            after = {r["teacher_id"]: sha256_file(ROOT / r["checkpoint_path"]) for r in evidence_manifest["teacher_records"]}
            if after != method_manifest["teacher_checkpoint_sha256s_before_router"]:
                raise RuntimeError("router training changed a frozen teacher checkpoint")
            result["teacher_checkpoints_unchanged_after_router"] = True
            _write_json(method_dir / "metrics.json", result)
            recorded.append({"method": method, **{k: v for k, v in result.items() if k not in ("target_best_metrics_report_only", "source_dev_selected_metrics", "source_dev_reload_metrics")},
                             "target_accuracy": result["target_best_metrics_report_only"]["accuracy"],
                             "target_macro_f1": result["target_best_metrics_report_only"]["macro_f1"],
                             "target_balanced_accuracy": result["target_best_metrics_report_only"]["balanced_accuracy"],
                             "target_cross_entropy": result["target_best_metrics_report_only"]["cross_entropy"],
                             "source_dev_selected_accuracy": result["source_dev_selected_metrics"].get("mean_subject_accuracy", result["source_dev_selected_metrics"]["accuracy"])})
        # A small K-shift summary is stored once and reuses the current router's
        # train-fitted scaler. This checks K=8/12/14 without adding embeddings.
        router_file = next((fold_root / method / "source_dev_selected.pt" for method in methods if method in ("C2-current", "C2-current-EMA", "C2-current-Huber")), None)
        if router_file is not None:
            model_state = torch.load(router_file, map_location=device, weights_only=False)["router"]
            feature_method = "C2-current"
            example = _build_features(oof, slice(0, 1), feature_method, config, device)
            router_for_drift = PairwiseUtilityRouter(example.shape[-1], hidden=(64, 32)).to(device)
            router_for_drift.load_state_dict(model_state, strict=True)
            _distribution_report(oof, dev, target_evidence, router_for_drift, config, device, fold_root / "source_count_feature_shift.json")
            del router_for_drift

    _write_json(fold_root / "fold_summary.json", {
        "status": "PASS", "runs": recorded,
        "elapsed_seconds": time.time() - fold_started_at,
    })
    del parent_state, model, mechanism
    if "oof" in locals():
        del oof, dev, evidence_manifest
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    return recorded


def _parse_id_list(value: list[int] | None, default: list[int], name: str) -> list[int]:
    result = default if value is None else [int(x) for x in value]
    if not result or len(set(result)) != len(result):
        raise ValueError(f"{name} must be a non-empty list without duplicates")
    return result


def run_experiment(args) -> dict[str, Any]:
    config_path = (ROOT / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config)
    config = load_c2_config(config_path)
    config_hash = c2_config_hash(config)
    implementation_hashes = _implementation_hashes()
    implementation_hash = json_fingerprint(implementation_hashes)
    audit_path = ROOT / "artifacts" / "seediv_c2_s1s2s3_parent_audit.json"
    audit = audit_parent_matrix(config_path, audit_path, load_weights=True)
    if args.mode == "full" and audit["status"] != "PASS":
        raise RuntimeError(f"full C2 requires all 135 valid E0 parents; see {audit_path}")
    if args.mode == "pilot":
        pilot_key = (int(config["pilot"]["session"]), int(config["pilot"]["target"]), int(config["pilot"]["seed"]))
        if not any((int(cell["session"]), int(cell["target"]), int(cell["seed"])) == pilot_key for cell in audit["cells"]):
            raise RuntimeError(f"pilot E0 parent {pilot_key} is missing or invalid; see {audit_path}")

    if args.mode == "plan":
        plan = {
            "status": "READY_NOT_STARTED" if audit["status"] == "PASS" else "BLOCKED_MISSING_PARENT_E0",
            "config_hash": config_hash,
            "implementation_hashes": implementation_hashes,
            "implementation_hash": implementation_hash,
            "audit_report": str(audit_path.relative_to(ROOT)),
            "parent_cells": audit["valid_cells"],
            "pilot": config["pilot"],
            "pilot_methods": config["pilot_methods"],
            "full_scope": {
                "sessions": config["sessions"],
                "targets": config["subject_ids"],
                "seeds": config["seeds"],
                "methods": config["methods"],
                "runs_per_method": int(config["execution"]["runs_per_method"]),
                "teacher_fits_per_fold_seed": 4,
                "parent_audit_status": audit["status"],
                "missing_parent_cells": audit["missing_cells"],
                "full_matrix_blocker": None if audit["status"] == "PASS" else "complete and audit Session 3 C1 E0 parent checkpoints before full C2",
            },
            "explicit_execute_flag_required": True,
        }
        out = ROOT / "artifacts" / "seediv_c2_execution_plan.json"
        _write_json(out, plan)
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return plan

    if args.mode in ("pilot", "full") and not args.execute:
        raise ValueError(f"{args.mode} performs EEG fitting; add --execute to launch it")
    if args.mode == "full":
        pilot_report_path = ROOT / args.pilot_report
        acceptance_path = ROOT / args.acceptance_record
        _validate_pilot_acceptance(pilot_report_path, acceptance_path, config_hash, implementation_hashes)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"requested {args.device}, but CUDA is unavailable; refusing silent CPU fallback")
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        gpu_name = torch.cuda.get_device_name(device)
    else:
        gpu_name = None
    experiment_started_at = time.time()

    if args.mode == "pilot":
        sessions = [int(config["pilot"]["session"])]
        targets = [int(config["pilot"]["target"])]
        seeds = [int(config["pilot"]["seed"])]
        methods = list(config["pilot_methods"])
    else:
        sessions = _parse_id_list(args.sessions, list(config["sessions"]), "sessions")
        targets = _parse_id_list(args.targets, list(config["subject_ids"]), "targets")
        seeds = _parse_id_list(args.seeds, list(config["seeds"]), "seeds")
        methods = list(config["methods"] if args.methods is None else args.methods)
    if any(value not in config["sessions"] for value in sessions):
        raise ValueError("only Sessions 1, 2, and 3 are in this C2 experiment")
    if any(value not in config["subject_ids"] for value in targets):
        raise ValueError("targets must be one-based subject IDs 1..15")
    if any(value not in config["seeds"] for value in seeds):
        raise ValueError("seeds must be 42, 43, or 44")
    unknown_methods = set(methods) - set(config["methods"])
    if unknown_methods:
        raise ValueError(f"unregistered methods requested: {sorted(unknown_methods)}")

    _validate_full_matrix_scope(args.mode, sessions, targets, seeds, methods, config)
    run_count = len(sessions) * len(targets) * len(seeds)
    report_name = f"seediv_c2_s1s2s3_{args.mode}_report.json"
    report_path = ROOT / "artifacts" / report_name
    _write_json(report_path, {
        "status": "RUNNING",
        "mode": args.mode,
        "config_hash": config_hash,
        "implementation_hashes": implementation_hashes,
        "implementation_hash": implementation_hash,
        "config_path": str(config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path),
        "parent_audit": str(audit_path.relative_to(ROOT)),
        "device": str(device),
        "gpu_name": gpu_name,
        "sessions": sessions,
        "targets": targets,
        "seeds": seeds,
        "methods": methods,
        "planned_outer_cells": run_count,
        "planned_teacher_fits": run_count * (4 if any(m.startswith("C2-") for m in methods) else 0),
        "started_at_unix": experiment_started_at,
    })
    # Read only the requested sessions. Each LOSO run is evaluated within one
    # session; session samples are never pooled across sessions.
    selected_config = dict(config)
    selected_config["sessions"] = sessions
    all_subjects = _load_seediv_session_data(selected_config)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for session in sessions:
        for target in targets:
            for seed in seeds:
                print(f"=== C2 fold: Session {session}, target S{target}, seed {seed} ===", flush=True)
                try:
                    fold_rows = execute_fold(
                        session=session,
                        target=target,
                        seed=seed,
                        methods=methods,
                        subjects=all_subjects[session],
                        config=config,
                        device=device,
                        config_hash=config_hash,
                        implementation_hashes=implementation_hashes,
                    )
                    rows.extend(fold_rows)
                except Exception as exc:
                    failures.append({
                        "session": session,
                        "target": target,
                        "seed": seed,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    })
                    print(f"FOLD FAILED S{session}/P{target}/seed{seed}: {type(exc).__name__}: {exc}", flush=True)
                    if args.mode == "pilot":
                        break
            if failures and args.mode == "pilot":
                break
        if failures and args.mode == "pilot":
            break

    expected_method_runs = run_count * len(methods)
    payload = {
        "status": "PASS" if not failures and len(rows) == expected_method_runs else "FAIL",
        "mode": args.mode,
        "config_hash": config_hash,
        "parent_config_hash": PARENT_CONFIG_HASH,
        "parent_audit_status": audit["status"],
        "device": str(device),
        "gpu_name": gpu_name,
        "sessions": sessions,
        "targets": targets,
        "seeds": seeds,
        "methods": methods,
        "expected_method_runs": expected_method_runs,
        "recorded_method_runs": len(rows),
        "folds": rows,
        "failures": failures,
        "finished_at_unix": time.time(),
        "elapsed_seconds": time.time() - experiment_started_at,
        "peak_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None,
        "peak_cuda_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else None,
        "limitations": [
            "E0 target-run checkpoints were frozen and not retrained.",
            "Per-subject normalization uses each evaluated subject's full unlabeled session as in the parent pipeline; full-system online causality is not claimed.",
            "Target-checkpoint accuracy records are descriptive only; source-dev-selected checkpoints are separately saved.",
        ],
    }
    _write_json(report_path, payload)
    summary_rows = []
    for row in rows:
        summary_rows.append({key: value for key, value in row.items() if isinstance(value, (str, int, float, bool)) or value is None})
    summary_csv = ROOT / "results" / "seediv_c2_e0_s1s2s3" / config_hash / f"{args.mode}_summary.csv"
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    if summary_rows:
        keys = list(dict.fromkeys(key for row in summary_rows for key in row))
        with summary_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=keys, extrasaction="ignore")
            writer.writeheader(); writer.writerows(summary_rows)
    print(json.dumps({key: payload[key] for key in ("status", "mode", "config_hash", "recorded_method_runs", "expected_method_runs", "failures")}, ensure_ascii=False, indent=2))
    if payload["status"] != "PASS":
        raise RuntimeError(f"C2 {args.mode} did not pass; inspect {report_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("audit", "plan", "pilot", "full"))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true", help="required to start the actual pilot/full fitting")
    parser.add_argument("--sessions", nargs="+", type=int)
    parser.add_argument("--targets", nargs="+", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--pilot-report", default="artifacts/seediv_c2_s1s2s3_pilot_report.json")
    parser.add_argument("--acceptance-record", default="artifacts/seediv_c2_s1s2s3_pilot_acceptance.json")
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    if args.mode == "audit":
        output = ROOT / "artifacts" / "seediv_c2_s1s2s3_parent_audit.json"
        report = audit_parent_matrix(config_path, output, load_weights=True)
        print(json.dumps({key: report[key] for key in ("status", "parent_config_hash_recomputed", "c2_config_hash", "expected_cells", "valid_cells", "missing_cells", "failures")}, ensure_ascii=False, indent=2))
        if report["status"] != "PASS":
            raise SystemExit(1)
    else:
        run_experiment(args)


def _history_cache_pair(oof: Evidence, dev: Evidence, target: Evidence, cfg: dict[str, Any], matched: bool):
    fn = lambda evidence: _history_features(evidence, cfg, matched=matched)
    return fn(oof), fn(dev), fn(target)


def _save_sample_outputs(directory: Path, method: str, evidence: Evidence, metrics: dict[str, Any], payload: dict[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        directory / "sample_outputs.npz",
        sample_ids=evidence.sample_ids,
        labels=evidence.labels,
        subject_ids=evidence.subject_ids,
        session_ids=evidence.session_ids,
        trial_ids=evidence.trial_ids,
        window_starts=evidence.window_starts,
        probabilities=payload["probabilities"],
        predictions=np.asarray(payload["probabilities"]).argmax(axis=-1).astype(np.int64),
        reference_logits=evidence.reference_logits,
        expert_logits=evidence.expert_logits,
        source_distances=evidence.source_distances,
        action_code=payload["action_code"],
        source_index=payload["source_index"],
        pair_index=payload["pair_index"],
        predicted_utility=payload.get("predicted_utility", np.zeros((len(evidence.labels), 0, 0), dtype=np.float32)),
        predicted_selected_utility=payload.get("predicted_selected_utility", np.zeros(len(evidence.labels), dtype=np.float32)),
        realized_selected_utility=payload.get("realized_selected_utility", np.zeros(len(evidence.labels), dtype=np.float32)),
        oracle_probabilities=payload.get("oracle_probabilities", np.zeros((len(evidence.labels), 4), dtype=np.float32)),
        oracle_predictions=np.asarray(payload.get("oracle_probabilities", np.zeros((len(evidence.labels), 4), dtype=np.float32))).argmax(axis=-1).astype(np.int64),
        oracle_action_logits=payload.get("oracle_action_logits", np.zeros((len(evidence.labels), 4), dtype=np.float32)),
        oracle_action_code=payload.get("oracle_action_code", np.zeros(len(evidence.labels), dtype=np.int32)),
        oracle_source_index=payload.get("oracle_source_index", np.full(len(evidence.labels), -1, dtype=np.int16)),
        oracle_pair_index=payload.get("oracle_pair_index", np.full(len(evidence.labels), -1, dtype=np.int8)),
        oracle_realized_selected_utility=payload.get("oracle_realized_selected_utility", np.zeros(len(evidence.labels), dtype=np.float32)),
    )
    _write_json(directory / "metrics.json", {key: value for key, value in metrics.items() if key != "confusion_matrix"} | {"confusion_matrix": metrics["confusion_matrix"]})
    with (directory / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["true\\pred", "neutral", "sad", "fear", "happy"])
        for label, row in enumerate(metrics["confusion_matrix"]):
            writer.writerow([label, *row])


def _best_key(metrics: dict[str, Any], epoch: int, *, source_dev: bool = False) -> tuple[float, float, int]:
    accuracy_key = "mean_subject_accuracy" if source_dev else "accuracy"
    return float(metrics[accuracy_key]), -float(metrics["cross_entropy"]), -int(epoch)


def train_router_method(
    *, method: str, session: int, target: int, seed: int, oof: Evidence, dev: Evidence,
    target_evidence: Evidence, config: dict[str, Any], device: torch.device, run_dir: Path,
    run_manifest: dict[str, Any],
) -> dict[str, Any]:
    loss_name = "Huber" if method == "C2-current-Huber" else "MSE"
    feature_method = "C2-current" if method in ("C2-current-EMA", "C2-current-Huber") else method
    if method not in ("C2-current", "C2-current-EMA", "C2-history", "C2-current-matched", "C2-current-Huber"):
        raise ValueError(f"not a trainable router method: {method}")
    history_caches = _history_cache_pair(oof, dev, target_evidence, config, matched=False) if method == "C2-history" else (None, None, None)
    matched_caches = _history_cache_pair(oof, dev, target_evidence, config, matched=True) if method == "C2-current-matched" else (None, None, None)

    setup_seed(int(seed))
    current_example = _build_features(
        oof, slice(0, min(2, len(oof.labels))), feature_method, config, device,
        history_caches[0], matched_caches[0],
    )
    router = PairwiseUtilityRouter(current_example.shape[-1], hidden=(64, 32)).to(device)
    _fit_standardizer(router, oof, feature_method, config, device, chunk_size=int(config["router"]["batch_size"]))
    optimizer = torch.optim.Adam(router.parameters(), lr=float(config["router"]["learning_rate"]))
    rng = torch.Generator(device="cpu").manual_seed(int(seed))
    batch_size = int(config["router"]["batch_size"])
    epochs = int(config["router"]["epochs"])
    history_rows = []
    source_dev_best = None
    target_best = None
    start_time = time.time()
    run_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        router.train()
        order = torch.randperm(len(oof.labels), generator=rng).numpy()
        train_losses = []
        for start in range(0, len(order), batch_size):
            rows_np = order[start:start + batch_size]
            ref = torch.as_tensor(oof.reference_logits[rows_np], dtype=torch.float32, device=device)
            exp = torch.as_tensor(oof.expert_logits[rows_np], dtype=torch.float32, device=device)
            dist = torch.as_tensor(oof.source_distances[rows_np], dtype=torch.float32, device=device)
            pair = torch.as_tensor(oof.prototype_pair_features, dtype=torch.float32, device=device)
            current_features = build_router_features(ref, exp, pair, dist)
            if feature_method == "C2-history":
                h = torch.as_tensor(history_caches[0][rows_np], dtype=current_features.dtype, device=device)
                features = append_history_features(current_features, h)
            elif feature_method == "C2-current-matched":
                h = torch.as_tensor(matched_caches[0][rows_np], dtype=current_features.dtype, device=device)
                features = append_history_features(current_features, h)
            else:
                features = current_features
            y = torch.as_tensor(oof.labels[rows_np], dtype=torch.long, device=device)
            targets = utility_labels(
                ref, exp, y,
                alpha=float(config["router"]["action_alpha"]),
                clip_limit=float(config["router"]["residual_clip"]),
            )
            optimizer.zero_grad(set_to_none=True)
            predicted = router(features)
            loss = _sample_weighted_loss(predicted, targets, loss_name)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        router.eval()
        dev_metrics, dev_payload = _evaluate_router(
            router, dev, config, device, feature_method,
            history=history_caches[1], matched=matched_caches[1],
            output_ema=method == "C2-current-EMA",
        )
        target_metrics, target_payload = _evaluate_router(
            router, target_evidence, config, device, feature_method,
            history=history_caches[2], matched=matched_caches[2],
            output_ema=method == "C2-current-EMA",
        )
        row = {
            "epoch": epoch,
            "train_loss_name": loss_name,
            "train_loss": float(np.mean(train_losses)),
            "source_dev_mean_subject_accuracy": dev_metrics["mean_subject_accuracy"],
            "source_dev_cross_entropy": dev_metrics["cross_entropy"],
            "target_accuracy": target_metrics["accuracy"],
            "target_macro_f1": target_metrics["macro_f1"],
            "target_balanced_accuracy": target_metrics["balanced_accuracy"],
            "target_cross_entropy": target_metrics["cross_entropy"],
            "target_action_rate": target_metrics["action_rate"],
            "target_negative_utility_rate": target_metrics["negative_utility_rate"],
        }
        history_rows.append(row)
        if source_dev_best is None or _best_key(dev_metrics, epoch, source_dev=True) > source_dev_best["selection_key"]:
            source_dev_best = {
                "epoch": epoch,
                "metrics": dev_metrics,
                "selection_key": _best_key(dev_metrics, epoch, source_dev=True),
                "model_state": {key: value.detach().cpu().clone() for key, value in router.state_dict().items()},
                "sample_payload": dev_payload,
            }
        # Target labels are used only in this predeclared descriptive report
        # record. This selected checkpoint is not the deployable selection.
        if target_best is None or target_metrics["accuracy"] > target_best["metrics"]["accuracy"]:
            target_best = {
                "epoch": epoch,
                "metrics": target_metrics,
                "model_state": {key: value.detach().cpu().clone() for key, value in router.state_dict().items()},
                "sample_payload": target_payload,
            }
        torch.save({
            "epoch": epoch,
            "router": router.state_dict(),
            "optimizer": optimizer.state_dict(),
            "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "sampler_rng": rng.get_state(),
            "history": history_rows,
            "manifest": run_manifest,
        }, run_dir / "last.pt")
        print(
            f"{method} S{session} target={target} seed={seed} epoch={epoch}/{epochs} "
            f"loss={row['train_loss']:.6f} dev_acc={row['source_dev_mean_subject_accuracy']:.4f} "
            f"eval_acc={row['target_accuracy']:.4f}", flush=True,
        )

    if source_dev_best is None or target_best is None:
        raise RuntimeError("router produced no epoch records")
    torch.save({
        "router": source_dev_best["model_state"],
        "router_epoch": source_dev_best["epoch"],
        "selection_rule": config["router"]["source_dev_checkpoint_rule"],
        "source_dev_metrics": source_dev_best["metrics"],
        "manifest": run_manifest,
    }, run_dir / "source_dev_selected.pt")
    torch.save({
        "router": target_best["model_state"],
        "router_epoch": target_best["epoch"],
        "selection_rule": config["router"]["target_checkpoint_rule"],
        "target_metrics_for_report_only": target_best["metrics"],
        "deployment_selection": False,
        "manifest": run_manifest,
    }, run_dir / "target_best.pt")
    (run_dir / "epoch_metrics.json").write_text(json.dumps(history_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    with (run_dir / "epoch_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history_rows[0]))
        writer.writeheader(); writer.writerows(history_rows)
    _save_sample_outputs(run_dir / "target_best", method, target_evidence, target_best["metrics"], target_best["sample_payload"])

    router.load_state_dict(source_dev_best["model_state"], strict=True)
    dev_recheck_metrics, dev_recheck_payload = _evaluate_router(
        router, dev, config, device, feature_method,
        history=history_caches[1], matched=matched_caches[1],
        output_ema=method == "C2-current-EMA",
    )
    if not _same_number(dev_recheck_metrics["mean_subject_accuracy"], source_dev_best["metrics"]["mean_subject_accuracy"]):
        raise RuntimeError("saved source-dev router checkpoint failed reload/re-evaluation")
    _save_sample_outputs(run_dir / "source_dev_selected", method, dev, dev_recheck_metrics, dev_recheck_payload)
    summary = {
        "status": "PASS",
        "method": method,
        "loss": loss_name,
        "session": int(session),
        "target": int(target),
        "seed": int(seed),
        "router_epochs": epochs,
        "router_batch_size": batch_size,
        "router_best_epoch_target_report_only": int(target_best["epoch"]),
        "target_best_metrics_report_only": target_best["metrics"],
        "source_dev_selected_epoch": int(source_dev_best["epoch"]),
        "source_dev_selected_metrics": source_dev_best["metrics"],
        "source_dev_reload_metrics": dev_recheck_metrics,
        "elapsed_seconds": time.time() - start_time,
        "target_checkpoint_is_deployment_selection": False,
        "teacher_parameters_updated": False,
    }
    _write_json(run_dir / "metrics.json", summary)
    (run_dir / "COMPLETE").write_text("PASS\n", encoding="ascii")
    return summary


if __name__ == "__main__":
    main()
