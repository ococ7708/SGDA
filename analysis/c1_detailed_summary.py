"""Rebuild compact C1 tables from COMPLETE run artifacts.

The inferential table does not treat seeds or repeated sessions as independent
subjects.  It averages seeds within session/target, then averages completed
sessions within target, and finally performs paired comparisons across the 15
target subjects.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


VARIANTS = tuple(f"E{i}" for i in range(6))
SEEDS = (42, 43, 44)
SESSIONS = (1, 2, 3)
TARGETS = tuple(range(1, 16))
DIAGNOSTICS = (
    "centered_logit_rms",
    "prediction_flip_rate",
    "metric_H_fro_mean",
    "metric_H_sample_variance",
    "source_weight_entropy",
    "mechanism_gradient_norm_mean",
)


def _mean(values):
    return float(np.mean(np.asarray(values, dtype=float)))


def _sample_sd(values):
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def load_rows(root: Path):
    rows, missing = [], []
    for variant in VARIANTS:
        for seed in SEEDS:
            for session in SESSIONS:
                for target in TARGETS:
                    fold = root / variant / f"seed{seed}" / f"session{session}_target{target:02d}"
                    if not (fold / "COMPLETE").is_file() or not (fold / "metrics.json").is_file():
                        missing.append(str(fold.relative_to(root)))
                        continue
                    metrics = json.loads((fold / "metrics.json").read_text(encoding="utf-8"))
                    rows.append(
                        {
                            "variant": variant,
                            "seed": seed,
                            "session": session,
                            "target": target,
                            **metrics,
                        }
                    )
    return rows, missing


def complete_sessions(rows: list[dict]) -> list[int]:
    present = {(r["variant"], r["seed"], r["session"], r["target"]) for r in rows}
    completed = []
    for session in SESSIONS:
        expected = {
            (variant, seed, session, target)
            for variant in VARIANTS
            for seed in SEEDS
            for target in TARGETS
        }
        if expected <= present:
            completed.append(session)
    return completed


def summarize(root: Path, bootstrap_resamples: int, bootstrap_seed: int):
    all_rows, missing = load_rows(root)
    sessions = complete_sessions(all_rows)
    rows = [row for row in all_rows if row["session"] in sessions]
    if not sessions:
        raise SystemExit("No complete C1 session is available for balanced analysis")

    by_variant = defaultdict(list)
    by_session_variant = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
        by_session_variant[(row["session"], row["variant"])].append(row)

    main = []
    for variant in VARIANTS:
        group = by_variant[variant]
        main.append(
            {
                "variant": variant,
                "n_runs": len(group),
                "accuracy_mean_pct": 100 * _mean([r["accuracy"] for r in group]),
                "accuracy_sd_pct": 100 * _sample_sd([r["accuracy"] for r in group]),
                "macro_f1_mean_pct": 100 * _mean([r["macro_f1"] for r in group]),
                "macro_f1_sd_pct": 100 * _sample_sd([r["macro_f1"] for r in group]),
                "balanced_accuracy_mean_pct": 100 * _mean([r["balanced_accuracy"] for r in group]),
                "balanced_accuracy_sd_pct": 100 * _sample_sd([r["balanced_accuracy"] for r in group]),
                "best_epoch_mean": _mean([r["best_epoch"] for r in group]),
            }
        )

    per_session = []
    for session in sessions:
        for variant in VARIANTS:
            group = by_session_variant[(session, variant)]
            per_session.append(
                {
                    "session": session,
                    "variant": variant,
                    "n_runs": len(group),
                    "accuracy_mean_pct": 100 * _mean([r["accuracy"] for r in group]),
                    "accuracy_sd_pct": 100 * _sample_sd([r["accuracy"] for r in group]),
                }
            )

    # Average three seeds within each session-target cell.
    cell_values = defaultdict(list)
    for row in rows:
        cell_values[(row["variant"], row["session"], row["target"])].append(row["accuracy"])
    cell_means = {key: _mean(values) for key, values in cell_values.items()}

    # Average completed sessions within each target subject.
    subject_means = {}
    for variant in VARIANTS:
        for target in TARGETS:
            subject_means[(variant, target)] = _mean(
                [cell_means[(variant, session, target)] for session in sessions]
            )

    rng = np.random.default_rng(bootstrap_seed)
    paired, raw_p = [], []
    reference = np.asarray([subject_means[("E0", target)] for target in TARGETS])
    for variant in VARIANTS[1:]:
        candidate = np.asarray([subject_means[(variant, target)] for target in TARGETS])
        delta_pp = 100 * (candidate - reference)
        indices = rng.integers(0, len(TARGETS), size=(bootstrap_resamples, len(TARGETS)))
        bootstrap = delta_pp[indices].mean(axis=1)
        paired_t = stats.ttest_rel(candidate, reference)
        wilcoxon = stats.wilcoxon(delta_pp, zero_method="wilcox", method="auto")
        raw_p.append(float(paired_t.pvalue))
        paired.append(
            {
                "variant": variant,
                "n_subjects": len(TARGETS),
                "delta_vs_E0_pp": _mean(delta_pp),
                "bootstrap95_low_pp": float(np.percentile(bootstrap, 2.5)),
                "bootstrap95_high_pp": float(np.percentile(bootstrap, 97.5)),
                "subject_wins": int(np.sum(delta_pp > 0)),
                "ties": int(np.sum(delta_pp == 0)),
                "losses": int(np.sum(delta_pp < 0)),
                "cohen_dz": float(delta_pp.mean() / delta_pp.std(ddof=1)),
                "paired_t_p": float(paired_t.pvalue),
                "wilcoxon_p": float(wilcoxon.pvalue),
            }
        )
    for row, adjusted in zip(paired, _holm_adjust(raw_p)):
        row["holm_p"] = adjusted

    diagnostics = []
    for variant in VARIANTS:
        group = by_variant[variant]
        item = {"variant": variant}
        for key in DIAGNOSTICS:
            item[key] = _mean([r[key] for r in group])
        diagnostics.append(item)

    return {
        "results_root": str(root),
        "completed_sessions": sessions,
        "complete_runs_used": len(rows),
        "planned_runs": len(VARIANTS) * len(SEEDS) * len(SESSIONS) * len(TARGETS),
        "missing_runs": len(missing),
        "bootstrap_resamples": bootstrap_resamples,
        "bootstrap_seed": bootstrap_seed,
        "main": main,
        "per_session": per_session,
        "paired_vs_E0": paired,
        "diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    parser.add_argument("--bootstrap-seed", type=int, default=20_260_927)
    args = parser.parse_args()

    result = summarize(args.results_root, args.bootstrap_resamples, args.bootstrap_seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "c1_detailed_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_csv(args.output_dir / "c1_main.csv", result["main"])
    _write_csv(args.output_dir / "c1_per_session.csv", result["per_session"])
    _write_csv(args.output_dir / "c1_paired_vs_E0.csv", result["paired_vs_E0"])
    _write_csv(args.output_dir / "c1_diagnostics.csv", result["diagnostics"])
    print(json.dumps({key: result[key] for key in (
        "completed_sessions", "complete_runs_used", "planned_runs", "missing_runs"
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
