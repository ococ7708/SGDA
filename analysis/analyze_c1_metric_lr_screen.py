"""Pair the 135-cell E2 baseline/A/B matrix and export summaries and figures."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from utils.seediv_c1_c2_protocol import canonical_hash, resolve_c1_config

BASE_HASH = "592351bfb649e601"
FOLDS = [(session, target, seed)
         for session in (1, 2, 3)
         for target in range(1, 16)
         for seed in (42, 43, 44)]
PHASE_CONFIG = {
    "A": "configs/seediv_c1_e2_metric_lr_A.json",
    "B": "configs/seediv_c1_e2_metric_lr_B.json",
}


def _effective_hash(path: str) -> str:
    config = json.loads((ROOT / path).read_text(encoding="utf-8"))
    return canonical_hash(resolve_c1_config(config))


def _read_metrics(root: Path, session: int, target: int, seed: int) -> dict:
    folder = root / f"seed{seed}" / f"session{session}_target{target:02d}"
    if not (folder / "COMPLETE").is_file():
        raise FileNotFoundError(f"incomplete fold: {folder}")
    return json.loads((folder / "metrics.json").read_text(encoding="utf-8"))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def analyze(output_dir: Path) -> dict:
    a_hash, b_hash = (_effective_hash(PHASE_CONFIG[name]) for name in ("A", "B"))
    roots = {
        "baseline_E2": ROOT / "results" / "seediv_c1" / BASE_HASH / "E2",
        "A_lr3_gamma0.5": ROOT / "results" / "seediv_c1" / a_hash / "E2",
        "B_lr3_gamma0.25": ROOT / "results" / "seediv_c1" / b_hash / "E2",
    }
    paired_rows, confusions = [], {name: np.zeros((4, 4), dtype=np.int64) for name in roots}
    for session, target, seed in FOLDS:
        records = {name: _read_metrics(path, session, target, seed) for name, path in roots.items()}
        for name, metric in records.items():
            confusions[name] += np.asarray(metric["confusion_matrix"], dtype=np.int64)
        base = records["baseline_E2"]
        for name in ("A_lr3_gamma0.5", "B_lr3_gamma0.25"):
            row = {
                "session": session, "target": target, "seed": seed, "comparison": name,
                "baseline_accuracy": base["accuracy"], "method_accuracy": records[name]["accuracy"],
                "delta_accuracy": records[name]["accuracy"] - base["accuracy"],
                "baseline_macro_f1": base["macro_f1"], "method_macro_f1": records[name]["macro_f1"],
                "delta_macro_f1": records[name]["macro_f1"] - base["macro_f1"],
                "baseline_balanced_accuracy": base["balanced_accuracy"],
                "method_balanced_accuracy": records[name]["balanced_accuracy"],
                "delta_balanced_accuracy": records[name]["balanced_accuracy"] - base["balanced_accuracy"],
                "accuracy_best_epoch": records[name]["best_epoch"],
                "training_ce": records[name].get("training_ce_mean"),
                "training_regularization": records[name].get("training_regularization_mean"),
                "metric_head_gradient_norm": records[name].get("metric_head_gradient_norm_mean"),
                "H_spectral_norm_max": records[name].get("metric_H_spectral_norm_max"),
                "H_spectral_ratio_max": records[name].get("metric_H_spectral_ratio_max"),
                "final_fused_flip_rate": records[name].get("final_fused_prediction_flip_rate"),
                "final_fused_net_correction_rate": records[name].get("final_fused_net_correction_rate"),
                "corrected_errors": records[name].get("final_fused_corrected_errors"),
                "harmed_correct": records[name].get("final_fused_harmed_correct"),
            }
            paired_rows.append(row)

    pair_changes = []
    for true_class in range(4):
        for predicted_class in range(4):
            baseline = int(confusions["baseline_E2"][true_class, predicted_class])
            a = int(confusions["A_lr3_gamma0.5"][true_class, predicted_class])
            b = int(confusions["B_lr3_gamma0.25"][true_class, predicted_class])
            pair_changes.append({
                "true_class": true_class, "predicted_class": predicted_class,
                "baseline_count": baseline, "A_count": a, "A_minus_baseline": a - baseline,
                "B_count": b, "B_minus_baseline": b - baseline, "B_minus_A": b - a,
            })
    _write_csv(output_dir / "paired_results.csv", paired_rows)
    _write_csv(output_dir / "confusion_pair_changes.csv", pair_changes)

    metrics = ("delta_accuracy", "delta_macro_f1", "delta_balanced_accuracy")
    summaries = {}
    for name in ("A_lr3_gamma0.5", "B_lr3_gamma0.25"):
        selected = [row for row in paired_rows if row["comparison"] == name]
        summaries[name] = {metric: {
            "n_paired_runs_descriptive_only": len(selected),
            "mean_delta": float(np.mean([row[metric] for row in selected])),
            "sd_across_runs_descriptive_only": float(np.std([row[metric] for row in selected], ddof=1)) if len(selected) > 1 else 0.0,
            "positive_run_count": int(sum(row[metric] > 0 for row in selected)),
        } for metric in metrics}
        summaries[name]["per_session"] = {}
        for session in (1, 2, 3):
            session_rows = [row for row in selected if row["session"] == session]
            summaries[name]["per_session"][str(session)] = {
                metric: float(np.mean([row[metric] for row in session_rows]))
                for metric in metrics
            }
    payload = {
        "scope": "all SEED-IV Sessions 1-3 separately; 15 targets/session; seeds 42/43/44",
        "baseline_config_hash": BASE_HASH,
        "A_effective_config_hash": a_hash,
        "B_effective_config_hash": b_hash,
        "confusion_class_order": ["neutral", "sad", "fear", "happy"],
        "confusion_matrices_summed_over_seeds": {name: matrix.tolist() for name, matrix in confusions.items()},
        "paired_descriptive_summary": summaries,
        "inferential_warning": "Runs and seeds are repeated measures, not independent subjects. Use subject-clustered inference for any population claim; the summary SDs are descriptive only.",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _plot_figures(output_dir, paired_rows)
    return payload


def _plot_figures(output_dir: Path, rows: list[dict]) -> None:
    import matplotlib.pyplot as plt

    methods = ["baseline_E2", "A_lr3_gamma0.5", "B_lr3_gamma0.25"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    phases = ["A_lr3_gamma0.5", "B_lr3_gamma0.25"]
    colors = ["#2878B5", "#E07A26"]
    for ax, key, title in ((axes[0], "delta_accuracy", "Accuracy Δ vs E2"),
                           (axes[1], "delta_macro_f1", "Macro-F1 Δ vs E2")):
        x = np.arange(3); width = 0.34
        for i, (phase, color) in enumerate(zip(phases, colors)):
            means = [np.mean([r[key] for r in rows if r["comparison"] == phase and r["session"] == s]) for s in (1, 2, 3)]
            ax.bar(x + (i - 0.5) * width, means, width, label=phase, color=color)
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x, ["Session 1", "Session 2", "Session 3"]); ax.set_ylabel(title)
        ax.set_title(title); ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"figure1_session_paired_metric_deltas.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    keys = ("H_spectral_ratio_max", "final_fused_flip_rate", "final_fused_net_correction_rate")
    titles = ("max ||H||₂ / γ", "Final fused flip rate", "Net correction rate")
    x = np.arange(3); width = 0.34
    for ax, key, title in zip(axes, keys, titles):
        for i, (phase, color) in enumerate(zip(phases, colors)):
            means = [np.mean([r[key] for r in rows if r["comparison"] == phase and r["session"] == s]) for s in (1, 2, 3)]
            ax.bar(x + (i - 0.5) * width, means, width, label=phase, color=color)
        ax.set_xticks(x, ["S1", "S2", "S3"]); ax.set_title(title); ax.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    for ext in ("png", "svg"):
        fig.savefig(output_dir / f"figure2_mechanism_diagnostics.{ext}", dpi=300 if ext == "png" else None)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/c1_e2_metric_lr_screen")
    args = parser.parse_args()
    output = Path(args.output_dir)
    if not output.is_absolute():
        output = ROOT / output
    print(json.dumps(analyze(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    import argparse
    main()
