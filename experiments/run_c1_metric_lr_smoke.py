"""Run engineering-only C1 metric-LR smoke checks for all SEED-IV sessions."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/seediv_c1_e2_metric_lr_A.json")
    parser.add_argument("--variant", choices=("E2",), default="E2")
    parser.add_argument("--target", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    # One fixed target/seed is only an engineering smoke; the eventual formal
    # experiment is configured separately and is deliberately not launched here.
    for session in (1, 2, 3):
        cmd = [
            sys.executable, "experiments/seediv_c1_end_to_end.py",
            "--config", args.config, "--variant", args.variant,
            "--session", str(session), "--target", str(args.target),
            "--seed", str(args.seed), "--device", args.device, "--smoke",
        ]
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
    print("PASS: C1 smoke completed for SEED-IV Sessions 1, 2 and 3; no full run was launched.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
