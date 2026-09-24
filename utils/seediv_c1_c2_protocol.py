"""Frozen split and manifest helpers for taskbook 06 (C1/C2)."""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path


DEV_TARGETS = (1, 6, 11)
SPLIT_SEED = 20260924


def source_roles(target: int, subjects=range(1, 16), seed: int = SPLIT_SEED):
    """Deterministic 8/3/3 T/V/U split among the 14 non-target subjects."""
    import numpy as np
    source = [int(x) for x in subjects if int(x) != int(target)]
    if len(source) != 14:
        raise ValueError("SEED-IV development split requires 15 subjects and one target")
    rng = np.random.default_rng(int(seed) + int(target))
    order = [source[i] for i in rng.permutation(len(source))]
    return {"T": order[:8], "V": order[8:11], "U": order[11:14]}


def oof_plan(subjects):
    """Subject-level leave-one-source-out plan; never mixes windows from a subject."""
    subjects = tuple(sorted(int(x) for x in subjects))
    return [{"held_out": s, "teacher_train": [x for x in subjects if x != s]} for s in subjects]


def canonical_hash(payload) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def resolve_c1_config(config, overrides=None, *, smoke=False):
    """Create the single configuration that both execution and hashing use."""
    effective = deepcopy(config)
    for key, value in (overrides or {}).items():
        if value is not None:
            effective[key] = value
    effective.setdefault("weight_decay", 1e-4)
    effective.setdefault("fusion_tau", 0.07)
    effective.setdefault("fusion_mode", "branch_logits")
    effective.setdefault("head_sharing", "independent")
    effective.setdefault("training_path", "branch_supervision")
    effective.setdefault("backbone", "e3_strong_de_channel_graph")
    effective.setdefault("dropout", 0.3)
    effective.setdefault("representation_mode", "cast_level1")
    effective.setdefault("classifier_type", "clip")
    effective.setdefault("num_classes", 4)
    effective.setdefault("sample_length", 3)
    effective.setdefault("st_dim", 128)
    effective.setdefault("graph_dim", 64)
    effective.setdefault("adapter_bottleneck", 32)
    effective.setdefault("graph_heads", 4)
    effective.setdefault("topk", 6)
    effective.setdefault("metric_hidden_dim", 64)
    effective.setdefault("va_config", "configs/seediv_va_coordinates.json")
    effective["run_kind"] = "smoke" if smoke else "formal"
    if smoke:
        effective["epochs"] = min(int(effective["epochs"]), 2)
    if effective["fusion_mode"] == "shared_fused_embedding" and effective["head_sharing"] != "shared":
        raise ValueError("shared_fused_embedding requires head_sharing=shared")
    if effective["training_path"] == "fused_uniform_direct" and effective["head_sharing"] != "shared":
        raise ValueError("fused_uniform_direct requires head_sharing=shared")
    return effective


def load_config(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    required = {"dataset", "protocol", "variants", "seeds", "epochs", "best_checkpoint_rule"}
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"config missing required keys: {missing}")
    if config["best_checkpoint_rule"] != "target_best_accuracy_same_checkpoint_metrics":
        raise ValueError("taskbook 06 requires target-best accuracy with same-checkpoint metrics")
    return config
