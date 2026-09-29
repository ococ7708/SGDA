"""Read-only preflight for the local SEED-family feature directory layout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from data_utils.constants.path_mapper import path_mapper


def inspect_seediv(root: Path, sessions: list[int]) -> dict:
    feature_root = root / "eeg_feature_smooth"
    report = {"dataset": "seediv", "root": str(root), "label_file": str(feature_root / "label.mat"), "sessions": {}}
    missing = []
    if not (feature_root / "label.mat").is_file():
        missing.append("eeg_feature_smooth/label.mat")
    for session in sessions:
        session_dir = feature_root / str(session)
        by_subject = {}
        for subject in range(1, 16):
            expected_name = _SEEDIV_FILES[session - 1][subject - 1]
            expected_path = session_dir / expected_name
            by_subject[str(subject)] = {"expected": expected_name, "exists": expected_path.is_file()}
            if not expected_path.is_file():
                missing.append(f"session {session}: missing expected {expected_path}")
        report["sessions"][str(session)] = {"directory": str(session_dir), "subjects": by_subject}
    report["missing_or_ambiguous"] = missing
    return report


def inspect_seed(root: Path) -> dict:
    feature_root = root / "ExtractedFeatures"
    files = sorted(p.name for p in feature_root.glob("*.mat")) if feature_root.is_dir() else []
    mats = [name for name in files if name.lower() != "label.mat"]
    expected = [f"{subject}_{date}.mat" for date_row in _SEED_DATES for subject, date in enumerate(date_row, start=1)]
    missing_expected = sorted(set(expected) - set(mats))
    missing = []
    if not (feature_root / "label.mat").is_file():
        missing.append("ExtractedFeatures/label.mat")
    if len(mats) != 45:
        missing.append(f"expected 45 subject/session MAT files excluding label.mat, found {len(mats)}")
    if missing_expected:
        missing.append(f"missing expected SEED files: {missing_expected}")
    return {"dataset": "seed", "root": str(root), "feature_directory": str(feature_root),
            "label_file": str(feature_root / "label.mat"), "mat_count_excluding_label": len(mats),
            "sample_files": mats[:5], "missing_or_ambiguous": missing}


def inspect_seedv(root: Path) -> dict:
    missing = []
    subjects = {}
    for subject in range(1, 17):
        data_path, label_path = root / f"{subject}_data.npy", root / f"{subject}_label.npy"
        subjects[str(subject)] = {"data": data_path.is_file(), "label": label_path.is_file()}
        if not data_path.is_file():
            missing.append(data_path.name)
        if not label_path.is_file():
            missing.append(label_path.name)
    return {"dataset": "seedv", "root": str(root), "subjects": subjects, "missing_or_ambiguous": missing}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=("seed", "seediv", "seedv"))
    parser.add_argument("--sessions", nargs="+", type=int)
    args = parser.parse_args()
    if args.dataset == "seediv":
        sessions = args.sessions or [1, 2, 3]
        if any(session not in (1, 2, 3) for session in sessions):
            parser.error("SEED-IV session IDs must be 1, 2, or 3")
        result = inspect_seediv(Path(path_mapper["seediv_de_lds"]), sessions)
    elif args.dataset == "seed":
        result = inspect_seed(Path(path_mapper["seed_de_lds"]))
    else:
        result = inspect_seedv(Path(path_mapper["seedv_de_lds"]))
    result["status"] = "PASS" if not result["missing_or_ambiguous"] else "FAIL"
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


_SEEDIV_FILES = [
    ["1_20160518.mat", "2_20150915.mat", "3_20150919.mat", "4_20151111.mat", "5_20160406.mat", "6_20150507.mat", "7_20150715.mat", "8_20151103.mat", "9_20151028.mat", "10_20151014.mat", "11_20150916.mat", "12_20150725.mat", "13_20151115.mat", "14_20151205.mat", "15_20150508.mat"],
    ["1_20161125.mat", "2_20150920.mat", "3_20151018.mat", "4_20151118.mat", "5_20160413.mat", "6_20150511.mat", "7_20150717.mat", "8_20151110.mat", "9_20151119.mat", "10_20151021.mat", "11_20150921.mat", "12_20150804.mat", "13_20151125.mat", "14_20151208.mat", "15_20150514.mat"],
    ["1_20161126.mat", "2_20151012.mat", "3_20151101.mat", "4_20151123.mat", "5_20160420.mat", "6_20150512.mat", "7_20150721.mat", "8_20151117.mat", "9_20151209.mat", "10_20151023.mat", "11_20151011.mat", "12_20150807.mat", "13_20140610.mat", "14_20140627.mat", "15_20131105.mat"],
]
_SEED_DATES = [
    ["20131027", "20140404", "20140603", "20140621", "20140411", "20130712", "20131027", "20140511", "20140620", "20131130", "20140618", "20131127", "20140527", "20140601", "20130709"],
    ["20131030", "20140413", "20140611", "20140702", "20140418", "20131016", "20131030", "20140514", "20140627", "20131204", "20140625", "20131201", "20140603", "20140615", "20131016"],
    ["20131107", "20140419", "20140629", "20140705", "20140506", "20131113", "20131106", "20140521", "20140704", "20131211", "20140630", "20131207", "20140610", "20140627", "20131105"],
]


if __name__ == "__main__":
    main()
