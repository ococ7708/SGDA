"""End-to-end C1 E0--E5 SEED-IV runner (taskbook 06).

The historical SGDA program is intentionally not imported or modified.  This
runner uses the frozen E3 representation protocol already audited in this repo.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

from config.setting import Setting
from data_utils.constants.path_mapper import path_mapper
from data_utils.load_data import get_data
from data_utils.text_to_vector import label_to_vector
from models.context_affective_metric import MultiSourceAffectiveMetric, prototype_basis
from models.geosem_stda import GeoSemSTDA, compute_source_domain_centroids
from utils.mix_utils import flatten_trials, setup_seed, zscore_subject_wise
from utils.seediv_c1_c2_protocol import canonical_hash, load_config


def _balanced_indices(labels, per_class, seed):
    labels=np.asarray(labels).reshape(-1); rng=np.random.default_rng(seed); chosen=[]
    for cls in sorted(np.unique(labels)):
        ids=np.flatnonzero(labels==cls); chosen.extend(rng.choice(ids,min(per_class,len(ids)),replace=False).tolist())
    return np.asarray(sorted(chosen),dtype=np.int64)


def _class_weights(labels, classes, device):
    counts=np.bincount(np.asarray(labels,dtype=np.int64),minlength=classes)
    return torch.tensor(len(labels)/(classes*np.maximum(counts,1)),dtype=torch.float32,device=device)


def _load_seediv(args):
    os.environ.setdefault("SGDA_SEEDIV_LOAD_WORKERS",str(args.data_load_workers))
    setting=Setting(dataset="seediv_de_lds",dataset_path=path_mapper["seediv_de_lds"],pass_band=[.3,50],
        extract_bands=None,time_window=1,overlap=0,sample_length=3,stride=1,seed=args.seed,feature_type="de_lds",
        only_seg=False,experiment_mode="subject-independent",sessions=[1,2,3],onehot=False)
    data,labels,channels,bands,classes=get_data(setting); data,labels=flatten_trials(data,labels)
    return zscore_subject_wise(data),labels,channels,bands,classes


def _load_reviewed_va(path, class_names, device):
    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    if not payload.get("source") or not payload.get("scale"): raise ValueError("reviewed V-A provenance is required")
    values=[]
    for name in class_names:
        row=payload.get("classes",{}).get(name,{})
        if row.get("valence") is None or row.get("arousal") is None: raise ValueError(f"missing V-A for {name}")
        values.append([row["valence"],row["arousal"]])
    return payload,torch.tensor(values,dtype=torch.float32,device=device)


def _diagnostics(y_true,y_pred,logits,classes):
    y_true=np.asarray(y_true); y_pred=np.asarray(y_pred); index=torch.as_tensor(y_true,dtype=torch.long)
    true=logits.gather(1,index[:,None]).squeeze(1); other=logits.clone(); other.scatter_(1,index[:,None],torch.finfo(logits.dtype).min)
    return {"accuracy":float((y_true==y_pred).mean()),"macro_f1":float(f1_score(y_true,y_pred,average="macro")),
        "balanced_accuracy":float(balanced_accuracy_score(y_true,y_pred)),
        "per_class_recall":recall_score(y_true,y_pred,labels=list(range(classes)),average=None,zero_division=0).tolist(),
        "confusion_matrix":confusion_matrix(y_true,y_pred,labels=list(range(classes))).tolist(),
        "predicted_class_ratio":[float((y_pred==c).mean()) for c in range(classes)],
        "classification_margin_mean":float((true-other.max(1).values).mean())}


def semantic_basis(prototypes, va):
    """Auditable least-squares V-A -> CLIP direction map, then QR orthogonalize."""
    x = va - va.mean(0, keepdim=True)
    y = F.normalize(prototypes, dim=-1) - F.normalize(prototypes, dim=-1).mean(0, keepdim=True)
    directions = (torch.linalg.pinv(x) @ y).T
    return torch.linalg.qr(directions, mode="reduced").Q[:, :2]


def make_loader(x, y, batch, shuffle, smoke, seed):
    x, y = np.asarray(x, np.float32), np.asarray(y, np.int64).reshape(-1)
    if smoke:
        keep = _balanced_indices(y, 2, seed); x, y = x[keep], y[keep]
    r = torch.zeros((len(x), 1, 1), dtype=torch.float32)
    return DataLoader(TensorDataset(torch.from_numpy(x), r, torch.from_numpy(y)), batch_size=batch, shuffle=shuffle), y


@torch.no_grad()
def evaluate(model, mechanism, loader, centroids, device, fusion_tau=0.07):
    model.eval(); mechanism.eval(); ys, preds, logits_all = [], [], []
    for x, r, y in loader:
        x, r = x.to(device), r.to(device)
        _, z_all, _, h, _, _ = model([], [], x, r, return_features=True)
        stack = torch.stack(z_all)
        weights = F.softmax(-torch.norm(stack - centroids[:, None], dim=-1) / fusion_tau, dim=0)
        output = mechanism.fused_output(h, z_all, weights)
        ys.append(y.numpy()); preds.append(output.logits.argmax(-1).cpu().numpy()); logits_all.append(output.logits.cpu())
    yt, yp, logits = np.concatenate(ys), np.concatenate(preds), torch.cat(logits_all)
    return _diagnostics(yt, yp, logits, 4)


def run(args):
    config = load_config(args.config)
    if args.variant not in config["variants"]:
        raise ValueError(f"{args.variant} is not registered in {args.config}")
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data, labels, channels, bands, classes = _load_seediv(args)
    if classes != 4: raise ValueError("SEED-IV must have four classes")
    session, target = args.session - 1, args.target - 1
    source_ids = [x for x in range(15) if x != target]
    text_dim, vectors = label_to_vector("seediv_de_lds", "clip", device=device)
    prototypes = torch.tensor(np.asarray([vectors[i] for i in sorted(vectors)], np.float32), device=device)
    _, va = _load_reviewed_va(args.va_config, ["neutral", "sad", "fear", "happy"], device)
    b_sem, b_proto = semantic_basis(prototypes, va), prototype_basis(prototypes)
    basis = b_proto if args.variant == "E5" else b_sem
    loaders, weights = [], []
    for sid in source_ids:
        loader, y = make_loader(data[session][sid], labels[session][sid], args.batch_size, True, args.smoke, args.seed + sid)
        loaders.append(loader); weights.append(_class_weights(y, 4, device))
    target_loader, _ = make_loader(data[session][target], labels[session][target], args.batch_size, False, args.smoke, args.seed)
    model = GeoSemSTDA(len(source_ids), channels, bands, 128, 64, 32, text_dim, 4,
        graph_heads=4, topk=6, dropout=.3, sample_length=3, representation_mode="cast_level1",
        classifier_type="clip", num_classes=4, cast_variant="e3_strong_de_channel_graph").to(device)
    mechanism = MultiSourceAffectiveMetric(len(source_ids), variant=args.variant, context_dim=128,
        prototypes=prototypes, basis=basis, rank=2, gamma=.5, bound_eps=1e-6, tau=.07,
        lambda_regularization=1e-4).to(device)
    parameters = list(model.parameters()) + list(mechanism.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr, weight_decay=1e-4)
    result_family = "seediv_c1_smoke" if args.smoke else "seediv_c1"
    out = ROOT / "results" / result_family / canonical_hash(config) / args.variant / f"seed{args.seed}" / f"session{args.session}_target{args.target:02d}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config": config, "variant": args.variant, "seed": args.seed, "session": args.session,
        "target": args.target, "source_ids": [x + 1 for x in source_ids], "device": str(device),
        "B_sem": {"va_config": args.va_config, "method": "centered least-squares V-A to CLIP; QR"},
        "B_used": "B_proto" if args.variant == "E5" else "B_sem", "smoke": args.smoke,
        "original_sgda_modified": False}
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    iters = [iter(x) for x in loaders]; steps = min(len(x) for x in loaders); best = None
    start_epoch, history = 1, []
    last_path = out / "last.pt"
    if args.resume and last_path.exists():
        state = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model"]); mechanism.load_state_dict(state["mechanism"])
        optimizer.load_state_dict(state["optimizer"]); torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all([rng.cpu() for rng in state["cuda_rng"]])
        np.random.set_state(state["numpy_rng"]); start_epoch=state["epoch"]+1
        best, history = state.get("best"), state.get("history", [])
    epochs = min(args.epochs, 2) if args.smoke else args.epochs
    for epoch in range(start_epoch, epochs + 1):
        model.train(); mechanism.train()
        for _ in range(steps):
            batches = []
            for i, iterator in enumerate(iters):
                try: batches.append(next(iterator))
                except StopIteration: iters[i] = iter(loaders[i]); batches.append(next(iters[i]))
            xs = [b[0].to(device) for b in batches]; rs = [b[1].to(device) for b in batches]; ys = [b[2].to(device) for b in batches]
            optimizer.zero_grad(); z, _, h, _, _, _ = model(xs, rs, return_features=True)
            outputs = mechanism.source_outputs(h, z)
            loss = torch.stack([F.cross_entropy(o.logits, y, weight=w) + o.regularization for o, y, w in zip(outputs, ys, weights)]).mean()
            loss.backward(); optimizer.step()
        centroids = compute_source_domain_centroids(model, loaders, device)
        metrics = evaluate(model, mechanism, target_loader, centroids, device)
        record = {**metrics, "best_epoch": epoch, "best_accuracy": metrics["accuracy"]}
        history.append({"epoch": epoch, **metrics})
        if best is None or record["accuracy"] > best["accuracy"]:
            best = record
            torch.save({"model": model.state_dict(), "mechanism": mechanism.state_dict(), "optimizer": optimizer.state_dict(),
                "epoch": epoch, "metrics": best, "manifest": manifest, "centroids": centroids.cpu(),
                "torch_rng": torch.get_rng_state()}, out / "target_best.pt")
        torch.save({"model":model.state_dict(),"mechanism":mechanism.state_dict(),"optimizer":optimizer.state_dict(),
            "epoch":epoch,"best":best,"history":history,"manifest":manifest,"centroids":centroids.cpu(),
            "torch_rng":torch.get_rng_state(),"cuda_rng":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng":np.random.get_state()}, last_path)
        print(f"{args.variant} session={args.session} target=S{args.target} epoch={epoch} acc={metrics['accuracy']:.4f} best={best['accuracy']:.4f}")
    (out / "metrics.json").write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "epoch_metrics.json").write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "COMPLETE").write_text("PASS\n", encoding="ascii")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/seediv_c1_full45.json"); p.add_argument("--variant", choices=[f"E{i}" for i in range(6)], required=True)
    p.add_argument("--session", type=int, choices=(1,2,3), required=True); p.add_argument("--target", type=int, choices=range(1,16), required=True)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int, default=200); p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--device", default="cuda:0"); p.add_argument("--data-load-workers", type=int, default=0)
    p.add_argument("--va-config", default="configs/seediv_va_coordinates.json"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    run(p.parse_args())
