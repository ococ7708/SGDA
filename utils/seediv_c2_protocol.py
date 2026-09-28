"""Frozen protocol, run identity, and provenance helpers for SEED-IV C2."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


PARENT_CONFIG_HASH = "592351bfb649e601"
DEFAULT_CONFIG = Path("configs/seediv_c2_e0_s1s2s3.json")
PARENT_RESULTS = Path("results/seediv_c1") / PARENT_CONFIG_HASH / "E0"


def canonical_hash(payload: Any) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_c2_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {
        "schema_version", "sessions", "subject_ids", "seeds", "parent_config_hash",
        "inner_oof", "router", "methods", "pilot_methods", "action", "history",
    }
    missing = sorted(required - config.keys())
    if missing:
        raise ValueError(f"C2 config is missing keys: {missing}")
    if config["parent_config_hash"] != PARENT_CONFIG_HASH:
        raise ValueError("C2 parent config hash is not the reviewed E0 configuration")
    if config["sessions"] != [1, 2, 3] or config["seeds"] != [42, 43, 44]:
        raise ValueError("C2 scope must cover Sessions 1/2/3 and seeds 42/43/44")
    if config["router"]["action_mode"] != "single":
        raise ValueError("Only single-action C2 is registered in this first phase")
    return config


def c2_config_hash(config: dict[str, Any]) -> str:
    return canonical_hash(config)


def parent_checkpoint_path(session: int, target: int, seed: int, root: str | Path = ".") -> Path:
    return Path(root) / PARENT_RESULTS / f"seed{int(seed)}" / f"session{int(session)}_target{int(target):02d}" / "target_best.pt"


def parent_run_dir(session: int, target: int, seed: int, root: str | Path = ".") -> Path:
    return parent_checkpoint_path(session, target, seed, root).parent


def inner_split(session: int, outer_target: int, *, split_seed: int = 20260927) -> dict[str, Any]:
    """Generate frozen 2-person D / 12-person R and three R folds.

    IDs are the dataset's one-based subject IDs. The same split is independent
    of backbone/router seed, so all variants and seeds share the same subjects.
    """
    target = int(outer_target)
    sources = np.asarray([sid for sid in range(1, 16) if sid != target], dtype=np.int64)
    if target not in range(1, 16) or len(sources) != 14:
        raise ValueError("outer target must be one-based SEED-IV subject ID 1..15")
    rng = np.random.default_rng(np.random.SeedSequence([int(split_seed), int(session), target]))
    permuted = rng.permutation(sources)
    dev = [int(x) for x in permuted[:2]]
    pool = [int(x) for x in rng.permutation(permuted[2:])]
    folds = [pool[i * 4:(i + 1) * 4] for i in range(3)]
    plan = {
        "session_id": int(session),
        "outer_target_id": target,
        "source_subject_ids": [int(x) for x in sources],
        "source_dev_subject_ids": dev,
        "router_oof_pool_subject_ids": pool,
        "oof_folds": [
            {
                "fold_id": fold_id,
                "heldout_subject_ids": heldout,
                "teacher_train_subject_ids": [sid for sid in pool if sid not in heldout],
            }
            for fold_id, heldout in enumerate(folds, start=1)
        ],
        "dev_teacher_train_subject_ids": pool,
        "split_seed": int(split_seed),
        "split_algorithm": "SeedSequence([split_seed, session_id, outer_target_id]); permute sources for D; independently permute R and split sequentially into 3 groups of 4",
    }
    validate_split(plan)
    return plan


def validate_split(plan: dict[str, Any]) -> None:
    target = int(plan["outer_target_id"])
    sources = set(map(int, plan["source_subject_ids"]))
    dev = set(map(int, plan["source_dev_subject_ids"]))
    pool = set(map(int, plan["router_oof_pool_subject_ids"]))
    if target in sources or len(sources) != 14:
        raise ValueError("outer target leaked into source set or source count is not 14")
    if len(dev) != 2 or len(pool) != 12 or dev & pool or dev | pool != sources:
        raise ValueError("D/R split must be disjoint, exhaustive, and have sizes 2/12")
    heldout_union: set[int] = set()
    for fold in plan["oof_folds"]:
        heldout = set(map(int, fold["heldout_subject_ids"]))
        teacher_train = set(map(int, fold["teacher_train_subject_ids"]))
        if len(heldout) != 4 or len(teacher_train) != 8:
            raise ValueError("each OOF fold must have four heldout and eight teacher-train subjects")
        if heldout & teacher_train or heldout | teacher_train != pool:
            raise ValueError("OOF teacher train/heldout sets must partition R")
        if target in teacher_train | heldout or dev & (teacher_train | heldout):
            raise ValueError("outer target or D leaked into an OOF fold")
        if heldout_union & heldout:
            raise ValueError("OOF heldout folds overlap")
        heldout_union |= heldout
    if heldout_union != pool:
        raise ValueError("OOF folds must cover all 12 R subjects exactly once")
    if set(map(int, plan["dev_teacher_train_subject_ids"])) != pool:
        raise ValueError("the D teacher must train on all 12 R subjects")


def build_sample_ids(session: int, subject: int, trial_ids: np.ndarray, window_starts: np.ndarray) -> np.ndarray:
    trial_ids = np.asarray(trial_ids, dtype=np.int16)
    window_starts = np.asarray(window_starts, dtype=np.int32)
    if trial_ids.shape != window_starts.shape:
        raise ValueError("trial IDs and window starts must have equal lengths")
    return np.asarray([
        f"S{int(session)}-P{int(subject):02d}-T{int(trial):02d}-W{int(start):05d}"
        for trial, start in zip(trial_ids, window_starts)
    ], dtype="U32")


def array_fingerprint(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.view(np.uint8))
    return digest.hexdigest()


def run_identity(
    config_hash: str, method: str, session: int, target: int, seed: int,
    parent_sha256: str, implementation_hash: str = "",
) -> str:
    return canonical_hash({
        "config_hash": config_hash,
        "method": method,
        "session": int(session),
        "target": int(target),
        "seed": int(seed),
        "parent_checkpoint_sha256": parent_sha256,
        "implementation_hash": implementation_hash,
    })
