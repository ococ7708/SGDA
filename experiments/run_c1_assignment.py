"""Run or print one teammate's registered C1 matrix, with safe resume-by-COMPLETE."""
import argparse, json, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(a):
    config = json.loads((ROOT / a.config).read_text(encoding="utf-8"))
    variants = a.variants.split(",") if a.variants else config["variants"]
    seeds = [int(x) for x in (a.seeds.split(",") if a.seeds else config["seeds"])]
    targets = [int(x) for x in (a.targets.split(",") if a.targets else range(1, 16))]
    from utils.seediv_c1_c2_protocol import canonical_hash
    family = "seediv_c1_smoke" if a.smoke else "seediv_c1"
    base = ROOT / "results" / family / canonical_hash(config)
    for variant in variants:
        for seed in seeds:
            for target in targets:
                complete = base / variant / f"seed{seed}" / f"session{a.session}_target{target:02d}" / "COMPLETE"
                cmd = [sys.executable, str(ROOT / "experiments/seediv_c1_end_to_end.py"), "--config", a.config,
                    "--variant", variant, "--session", str(a.session), "--target", str(target), "--seed", str(seed), "--device", a.device, "--resume"]
                if a.smoke: cmd.append("--smoke")
                print(" ".join(cmd))
                if a.execute and not complete.exists(): subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config", default="configs/seediv_c1_full45.json")
    p.add_argument("--session", type=int, choices=(1,2,3), required=True); p.add_argument("--targets"); p.add_argument("--variants"); p.add_argument("--seeds")
    p.add_argument("--device", default="cuda:0"); p.add_argument("--smoke", action="store_true"); p.add_argument("--execute", action="store_true")
    main(p.parse_args())
