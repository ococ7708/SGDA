"""Independently audit a completed single-fold SEED-IV C2 engineering pilot.

This is a read-only verifier. It never opens EEG MAT files or edits results.
Run from the repository root after the pilot report has status PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_METHODS = ["B0", "B1", "C2-current", "C2-history"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def json_fingerprint(payload) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def audit(report_path: Path) -> dict:
    report = read_json(report_path)
    require(report.get("status") == "PASS", "pilot report is not PASS")
    require(report.get("sessions") == [1] and report.get("targets") == [1] and report.get("seeds") == [42],
            "report does not describe the fixed S1/target-1/seed-42 pilot")
    require(report.get("methods") == EXPECTED_METHODS and report.get("recorded_method_runs") == 4,
            "pilot method/run coverage is incomplete or unexpected")
    require(report.get("expected_method_runs") == 4 and report.get("failures") == [],
            "pilot report has failures or missing method cells")
    require(report.get("parent_config_hash") == "592351bfb649e601" and report.get("parent_audit_status") == "PASS",
            "parent identity/audit is missing or failed")

    impl_hashes = report.get("implementation_hashes", {})
    require(bool(impl_hashes), "report lacks implementation file hashes")
    require(json_fingerprint(impl_hashes) == report.get("implementation_hash"),
            "implementation hash does not match its file-hash map")
    for relative, expected_hash in impl_hashes.items():
        path = ROOT / relative
        require(path.is_file() and sha256_file(path) == expected_hash,
                f"implementation file missing/changed since pilot: {relative}")

    parent_audit = read_json(ROOT / "artifacts" / "seediv_c2_parent_audit.json")
    require(parent_audit.get("status") == "PASS" and parent_audit.get("valid_cells") == 90,
            "parent E0 artifact audit is incomplete")

    implementation_dir = f"impl_{report['implementation_hash'][:12]}"
    fold_root = ROOT / "results" / "seediv_c2_e0_s1s2" / report["config_hash"] / implementation_dir / "session1_target01" / "seed42"
    run_manifests = {}
    for method in EXPECTED_METHODS:
        method_dir = fold_root / method
        require((method_dir / "COMPLETE").is_file(), f"{method} lacks COMPLETE")
        manifest = read_json(method_dir / "run_manifest.json")
        run_manifests[method] = manifest
        require(manifest.get("implementation_hash") == report["implementation_hash"],
                f"{method} implementation fingerprint differs from report")
        require(manifest.get("implementation_hashes") == impl_hashes,
                f"{method} implementation file hashes differ from report")
        require(manifest.get("parent_config_hash") == report["parent_config_hash"],
                f"{method} parent config differs from report")
    parent_shas = {m["parent_checkpoint_sha256"] for m in run_manifests.values()}
    require(len(parent_shas) == 1, "methods do not share the exact same parent checkpoint")

    b0 = read_json(fold_root / "B0" / "metrics.json")
    require(b0.get("parent_reproduction_pass") is True, "B0 does not reproduce parent E0")
    require(all(abs(b0["parent_reproduction_delta"][key]) <= 1e-10 for key in ("accuracy", "macro_f1", "balanced_accuracy")),
            "B0 metric delta exceeds the recorded tolerance")

    cache_dir = fold_root / "shared_oof_cache"
    evidence = read_json(cache_dir / "evidence_manifest.json")
    split = read_json(cache_dir / "split_manifest.json")
    require(evidence.get("status") == "PASS", "OOF evidence manifest is not PASS")
    require(evidence.get("implementation_hashes") == impl_hashes, "evidence cache code fingerprint differs")
    d_ids = set(map(int, split["source_dev_subject_ids"]))
    r_ids = set(map(int, split["router_oof_pool_subject_ids"]))
    outer_target = int(split["outer_target_id"])
    require(len(d_ids) == 2 and len(r_ids) == 12 and d_ids.isdisjoint(r_ids), "inner D/R split is invalid")
    require(outer_target not in d_ids | r_ids, "outer target entered inner source split")
    require(evidence.get("outer_target_excluded_from_all_teacher_fit") is True, "outer target not excluded from teacher fit")
    require(evidence.get("d_excluded_from_oof_teacher_fit") is True, "source-dev subjects not excluded from OOF fits")
    require(evidence.get("source_dev_subjects_are_not_router_train") is True, "source-dev evidence entered router training data")

    teacher_records = {record["teacher_id"]: record for record in evidence["teacher_records"]}
    expected_teacher_ids = {f"oof_fold_{fold['fold_id']}" for fold in split["oof_folds"]} | {"source_dev_teacher"}
    require(set(teacher_records) == expected_teacher_ids, "teacher set differs from frozen 3+1 protocol")
    oof_heldout_ids: list[int] = []
    for fold in split["oof_folds"]:
        teacher_id = f"oof_fold_{fold['fold_id']}"
        record = teacher_records[teacher_id]
        train_expected = set(map(int, fold["teacher_train_subject_ids"]))
        eval_expected = set(map(int, fold["heldout_subject_ids"]))
        require(len(train_expected) == 8 and len(eval_expected) == 4, f"{teacher_id} source counts are wrong")
        require(train_expected.isdisjoint(eval_expected), f"{teacher_id} train/eval subjects overlap")
        require(not (train_expected & d_ids) and outer_target not in train_expected | eval_expected,
                f"{teacher_id} includes D or outer target")
        require(set(map(int, record["actual_fit_subject_ids"])) == train_expected,
                f"{teacher_id} recorded fit subjects differ from split")
        require(set(map(int, record["evaluation_subject_ids"])) == eval_expected,
                f"{teacher_id} recorded held-out subjects differ from split")
        oof_heldout_ids.extend(sorted(eval_expected))
    require(len(oof_heldout_ids) == 12 and set(oof_heldout_ids) == r_ids and len(set(oof_heldout_ids)) == 12,
            "OOF held-out folds do not partition R exactly once")
    dev_record = teacher_records["source_dev_teacher"]
    require(set(map(int, dev_record["actual_fit_subject_ids"])) == r_ids and
            set(map(int, dev_record["evaluation_subject_ids"])) == d_ids,
            "source-dev teacher must fit on R and evaluate only D")

    for record in teacher_records.values():
        sample_manifest_path = ROOT / record["actual_sample_set_manifest_path"]
        checkpoint_path = ROOT / record["checkpoint_path"]
        require(sample_manifest_path.is_file() and sha256_file(sample_manifest_path) == record["actual_sample_set_manifest_sha256"],
                f"{record['teacher_id']} exact sample-set manifest hash mismatch")
        require(checkpoint_path.is_file() and sha256_file(checkpoint_path) == record["checkpoint_sha256"],
                f"{record['teacher_id']} checkpoint hash mismatch")
        with np.load(sample_manifest_path, allow_pickle=False) as sample_sets:
            train_sid = sample_sets["train_subject_ids"].astype(int)
            eval_sid = sample_sets["evaluation_subject_ids"].astype(int)
            train_samples = sample_sets["train_sample_ids"].astype(str)
            eval_samples = sample_sets["evaluation_sample_ids"].astype(str)
            require(len(train_samples) == len(train_sid) == len(sample_sets["train_labels"]),
                    f"{record['teacher_id']} train sample arrays have inconsistent lengths")
            require(len(eval_samples) == len(eval_sid) == len(sample_sets["evaluation_labels"]),
                    f"{record['teacher_id']} evaluation sample arrays have inconsistent lengths")
            require(len(set(train_samples)) == len(train_samples) and len(set(eval_samples)) == len(eval_samples),
                    f"{record['teacher_id']} has duplicate sample IDs")
            require(set(train_sid).isdisjoint(set(eval_sid)), f"{record['teacher_id']} actual train/eval subjects overlap")
            require(set(train_samples).isdisjoint(set(eval_samples)), f"{record['teacher_id']} actual train/eval sample IDs overlap")
            require(set(train_sid) == set(map(int, record["actual_fit_subject_ids"])),
                    f"{record['teacher_id']} actual loaded training samples do not match recorded fit subjects")
            require(set(eval_sid) == set(map(int, record["evaluation_subject_ids"])),
                    f"{record['teacher_id']} actual evaluation samples do not match recorded subjects")

    router_results = {}
    for method in ("C2-current", "C2-history"):
        method_dir = fold_root / method
        manifest = run_manifests[method]
        metrics = read_json(method_dir / "metrics.json")
        require(metrics.get("status") == "PASS", f"{method} metrics status is not PASS")
        require(metrics.get("teacher_checkpoints_unchanged_after_router") is True and
                metrics.get("teacher_parameters_updated") is False,
                f"{method} altered teacher parameters")
        require(metrics.get("router_epochs") == 100 and metrics.get("router_batch_size") == 256,
                f"{method} router budget differs from frozen config")
        require(metrics["source_dev_reload_metrics"] == metrics["source_dev_selected_metrics"],
                f"{method} source-dev checkpoint reload metrics differ")
        before = manifest["teacher_checkpoint_sha256s_before_router"]
        after = {key: sha256_file(ROOT / teacher_records[key]["checkpoint_path"]) for key in teacher_records}
        require(before == after, f"{method} teacher checkpoint SHA changed during router training")

        sample_path = method_dir / "target_best" / "sample_outputs.npz"
        sample_metrics = read_json(method_dir / "target_best" / "metrics.json")
        with np.load(sample_path, allow_pickle=False) as samples:
            labels = samples["labels"].astype(int)
            predictions = samples["predictions"].astype(int)
            reference_predictions = samples["reference_logits"].argmax(axis=-1)
            require(np.array_equal(predictions, samples["probabilities"].argmax(axis=-1)),
                    f"{method} saved prediction classes differ from probabilities")
            corrected = int(np.sum((reference_predictions != labels) & (predictions == labels)))
            harmed = int(np.sum((reference_predictions == labels) & (predictions != labels)))
            require(corrected == sample_metrics["corrected_errors"] and harmed == sample_metrics["harmed_correct"],
                    f"{method} final-fusion per-sample correction/harm counts do not recompute")
        router_results[method] = {
            "target_report_accuracy": metrics["target_best_metrics_report_only"]["accuracy"],
            "source_dev_selected_epoch": metrics["source_dev_selected_epoch"],
            "teacher_hashes_unchanged": True,
            "corrected_errors": corrected,
            "harmed_correct": harmed,
        }

    return {
        "status": "PASS",
        "report": str(report_path.relative_to(ROOT)),
        "config_hash": report["config_hash"],
        "implementation_hash": report["implementation_hash"],
        "parent_cells_verified": 90,
        "pilot_methods_verified": EXPECTED_METHODS,
        "B0_parent_reproduction": True,
        "D_subjects": sorted(d_ids),
        "R_subjects": sorted(r_ids),
        "teacher_fits_verified": 4,
        "router_results": router_results,
        "gpu_name": report["gpu_name"],
        "peak_cuda_memory_allocated_bytes": report["peak_cuda_memory_allocated_bytes"],
        "peak_cuda_memory_reserved_bytes": report["peak_cuda_memory_reserved_bytes"],
        "elapsed_seconds": report["elapsed_seconds"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", default="artifacts/seediv_c2_pilot_report.json")
    parser.add_argument("--output", default="artifacts/seediv_c2_pilot_independent_audit.json")
    args = parser.parse_args()
    report_path = Path(args.report)
    if not report_path.is_absolute():
        report_path = ROOT / report_path
    result = audit(report_path)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
