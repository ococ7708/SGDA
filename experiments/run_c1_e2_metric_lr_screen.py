"""Run the paired, all-session E2 metric-LR A/B experiment (135 runs/phase)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.seediv_c1_c2_protocol import canonical_hash, resolve_c1_config

SEEDS = (42, 43, 44)
FOLDS = [(session, target, seed)
         for session in (1, 2, 3)
         for target in range(1, 16)
         for seed in SEEDS]
CONFIGS = {
    "A": "configs/seediv_c1_e2_metric_lr_A.json",
    "B": "configs/seediv_c1_e2_metric_lr_B.json",
}


def config_identity(path: str) -> tuple[dict, str]:
    config = json.loads((ROOT / path).read_text(encoding="utf-8"))
    effective = resolve_c1_config(config)
    return effective, canonical_hash(effective)


def completed_runs(phase: str) -> list[Path]:
    _, config_hash = config_identity(CONFIGS[phase])
    return [
        ROOT / "results" / "seediv_c1" / config_hash / "E2" / f"seed{seed}"
        / f"session{session}_target{target:02d}" / "COMPLETE"
        for session, target, seed in FOLDS
    ]


def missing_parent_runs() -> list[Path]:
    parent = ROOT / "results" / "seediv_c1" / "592351bfb649e601" / "E2"
    missing = []
    for session, target, seed in FOLDS:
        folder = parent / f"seed{seed}" / f"session{session}_target{target:02d}"
        if not (folder / "COMPLETE").is_file() or not (folder / "metrics.json").is_file():
            missing.append(folder)
    return missing


def _metrics_path(root: Path, session: int, target: int, seed: int) -> Path:
    return root / f"seed{seed}" / f"session{session}_target{target:02d}" / "metrics.json"


def review_phase_a() -> dict | None:
    a_hash = config_identity(CONFIGS["A"])[1]
    a_root = ROOT / "results" / "seediv_c1" / a_hash / "E2"
    baseline_root = ROOT / "results" / "seediv_c1" / "592351bfb649e601" / "E2"
    if not all(path.is_file() for path in completed_runs("A")):
        return None
    paired = []
    for session, target, seed in FOLDS:
        base = json.loads(_metrics_path(baseline_root, session, target, seed).read_text(encoding="utf-8"))
        new = json.loads(_metrics_path(a_root, session, target, seed).read_text(encoding="utf-8"))
        paired.append({
            "session": session, "target": target, "seed": seed,
            "delta_accuracy": float(new["accuracy"] - base["accuracy"]),
            "delta_macro_f1": float(new["macro_f1"] - base["macro_f1"]),
            "delta_balanced_accuracy": float(new["balanced_accuracy"] - base["balanced_accuracy"]),
        })
    cell_rows = []
    for session in (1, 2, 3):
        for target in range(1, 16):
            rows = [r for r in paired if r["session"] == session and r["target"] == target]
            cell_rows.append({
                "session": session, "target": target,
                **{key: sum(r[key] for r in rows) / len(rows) for key in
                   ("delta_accuracy", "delta_macro_f1", "delta_balanced_accuracy")},
            })
    session_summary = {}
    for session in (1, 2, 3):
        rows = [r for r in cell_rows if r["session"] == session]
        session_summary[str(session)] = {
            key: sum(r[key] for r in rows) / len(rows)
            for key in ("delta_accuracy", "delta_macro_f1", "delta_balanced_accuracy")
        }
    overall = {
        key: sum(r[key] for r in cell_rows) / len(cell_rows)
        for key in ("delta_accuracy", "delta_macro_f1", "delta_balanced_accuracy")
    }
    positive_cells = {
        key: sum(r[key] > 0 for r in cell_rows)
        for key in ("delta_accuracy", "delta_macro_f1")
    }
    positive_sessions = {
        key: sum(session_summary[str(s)][key] > 0 for s in (1, 2, 3))
        for key in ("delta_accuracy", "delta_macro_f1")
    }
    gate = {
        "scope": "all SEED-IV sessions, all 15 targets/session, seeds 42/43/44; seeds are repeats, not independent subjects",
        "rule_locked_before_A": "overall mean accuracy and macro-F1 deltas positive; at least 23/45 subject-session cells positive for each; at least 2/3 session means positive for each; all-session mean balanced-accuracy delta nonnegative",
        "paired_deltas": paired,
        "subject_session_cell_deltas_mean_over_seeds": cell_rows,
        "session_mean_deltas": session_summary,
        "overall_mean_deltas": overall,
        "positive_subject_session_cells": positive_cells,
        "positive_session_count": positive_sessions,
    }
    gate["continue_to_B"] = (
        overall["delta_accuracy"] > 0
        and overall["delta_macro_f1"] > 0
        and positive_cells["delta_accuracy"] >= 23
        and positive_cells["delta_macro_f1"] >= 23
        and positive_sessions["delta_accuracy"] >= 2
        and positive_sessions["delta_macro_f1"] >= 2
        and all(session_summary[str(s)]["delta_balanced_accuracy"] >= 0 for s in (1, 2, 3))
    )
    gate["recommendation"] = "Run fixed-gamma B" if gate["continue_to_B"] else "Stop after A; report negative/inconsistent screen"
    out = ROOT / "artifacts" / "c1_e2_metric_lr_screen" / "A_screen_gate.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("A", "B"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--execute", action="store_true", help="otherwise print the exact run plan")
    args = parser.parse_args()

    missing_parent = missing_parent_runs()
    if missing_parent:
        missing_text = "\n  ".join(str(path.relative_to(ROOT)) for path in missing_parent)
        raise SystemExit(
            f"Cannot start C1 phase {args.phase}: paired E2 parent is incomplete "
            f"({len(missing_parent)}/135 folds missing). Complete and audit the exact parent first:\n  "
            + missing_text
        )

    if args.phase == "B":
        missing = [str(path.relative_to(ROOT)) for path in completed_runs("A") if not path.is_file()]
        if missing:
            raise SystemExit("B is gated on all A folds completing; missing:\n  " + "\n  ".join(missing))
        gate = review_phase_a()
        if not gate or not gate["continue_to_B"]:
            raise SystemExit("A screen did not meet its predeclared continuation rule; B stopped. See artifacts/c1_e2_metric_lr_screen/A_screen_gate.json")

    config_path = CONFIGS[args.phase]
    effective, config_hash = config_identity(config_path)
    print(json.dumps({
        "phase": args.phase,
        "config": config_path,
        "effective_config_hash": config_hash,
        "metric_lr_multiplier": effective["metric_lr_multiplier"],
        "backbone_lr": effective["learning_rate"],
        "metric_lr": effective["learning_rate"] * effective["metric_lr_multiplier"],
        "gamma": effective["gamma"],
        "folds": FOLDS,
        "execute": args.execute,
    }, ensure_ascii=False, indent=2))

    for session, target, seed in FOLDS:
        complete = ROOT / "results" / "seediv_c1" / config_hash / "E2" / f"seed{seed}" / f"session{session}_target{target:02d}" / "COMPLETE"
        if complete.is_file():
            print(f"SKIP complete: S{session}/T{target}/seed{seed}")
            continue
        cmd = [
            sys.executable, "experiments/seediv_c1_end_to_end.py",
            "--config", config_path, "--variant", "E2", "--session", str(session),
            "--target", str(target), "--seed", str(seed), "--device", args.device, "--resume",
        ]
        print("RUN:", " ".join(cmd), flush=True)
        if args.execute:
            subprocess.run(cmd, cwd=ROOT, check=True)
    if args.phase == "A":
        gate = review_phase_a()
        if gate:
            print("A SCREEN GATE:", json.dumps(gate, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
