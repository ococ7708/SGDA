"""Aggregate only COMPLETE C1 folds; fail loudly on missing registered cells."""
import argparse, csv, json, sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from utils.seediv_c1_c2_protocol import canonical_hash, load_config


def main(config_path, allow_partial):
    cfg=load_config(config_path); root=ROOT/"results"/"seediv_c1"/canonical_hash(cfg); rows=[]; missing=[]
    for variant in cfg["variants"]:
        for seed in cfg["seeds"]:
            for session in (1,2,3):
                for target in range(1,16):
                    fold=root/variant/f"seed{seed}"/f"session{session}_target{target:02d}"
                    if not (fold/"COMPLETE").exists(): missing.append(str(fold.relative_to(root))); continue
                    m=json.loads((fold/"metrics.json").read_text(encoding="utf-8")); rows.append({"variant":variant,"seed":seed,"session":session,"target":target,**m})
    if missing and not allow_partial: raise SystemExit(f"INCOMPLETE: {len(missing)} cells missing; first={missing[:5]}")
    root.mkdir(parents=True,exist_ok=True)
    if rows:
        with (root/"fold_results.csv").open("w",newline="",encoding="utf-8-sig") as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
    summary={v:{"n":len(x),"mean_best_accuracy":float(np.mean([r['best_accuracy'] for r in x])),"std_ddof1":float(np.std([r['best_accuracy'] for r in x],ddof=1)) if len(x)>1 else 0.0}
        for v in cfg["variants"] if (x:=[r for r in rows if r["variant"]==v])}
    (root/"summary.json").write_text(json.dumps({"summary":summary,"missing":missing},ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2)); print(f"complete={len(rows)} missing={len(missing)}")


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/seediv_c1_full45.json"); p.add_argument("--allow-partial",action="store_true"); a=p.parse_args(); main(a.config,a.allow_partial)
