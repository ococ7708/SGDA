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
from utils.seediv_c1_c2_protocol import canonical_hash, load_config, resolve_c1_config


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
        extract_bands=None,time_window=1,overlap=0,sample_length=args.sample_length,stride=1,seed=args.seed,feature_type="de_lds",
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


def _build_optimizer(model, mechanism, effective_config):
    """Keep the backbone LR and scale only the C1 metric-mechanism group."""
    base_lr = float(effective_config["learning_rate"])
    multiplier = float(effective_config.get("metric_lr_multiplier", 1.0))
    if multiplier <= 0:
        raise ValueError("metric_lr_multiplier must be positive")
    return torch.optim.Adam([
        {"params": list(model.parameters()), "lr": base_lr, "group_name": "backbone"},
        {"params": list(mechanism.parameters()), "lr": base_lr * multiplier, "group_name": "mechanism"},
    ], lr=base_lr, weight_decay=float(effective_config["weight_decay"]))


def make_loader(x, y, batch, shuffle, smoke, seed):
    x, y = np.asarray(x, np.float32), np.asarray(y, np.int64).reshape(-1)
    if smoke:
        keep = _balanced_indices(y, 2, seed); x, y = x[keep], y[keep]
    r = torch.zeros((len(x), 1, 1), dtype=torch.float32)
    return DataLoader(TensorDataset(torch.from_numpy(x), r, torch.from_numpy(y)), batch_size=batch, shuffle=shuffle), y


@torch.no_grad()
def evaluate(model, mechanism, loader, centroids, device, fusion_tau=0.07,
             fusion_mode="branch_logits", num_classes=4):
    model.eval(); mechanism.eval(); ys, preds, logits_all = [], [], []
    diagnostics = {"centered_logit_rms": [], "prediction_flip_rate": [],
                   "metric_H_fro_mean": [], "metric_H_sample_variance": [],
                   "metric_H_spectral_norm_mean": [], "metric_H_spectral_norm_max": [],
                   "metric_H_spectral_ratio_mean": [], "metric_H_spectral_ratio_max": [],
                   "source_weight_entropy": []}
    base_preds = []
    for x, r, y in loader:
        x, r = x.to(device), r.to(device)
        _, z_all, _, h, _, _ = model([], [], x, r, return_features=True)
        stack = torch.stack(z_all)
        weights = F.softmax(-torch.norm(stack - centroids[:, None], dim=-1) / fusion_tau, dim=0)
        output = mechanism.fused_output(h, z_all, weights, fusion_mode=fusion_mode)
        base_branch_logits = torch.stack([
            F.normalize(embedding, dim=-1) @ head.prototypes.T / head.tau
            for head, embedding in zip(mechanism._branch_heads(), z_all)
        ])
        base_fused = (weights[..., None] * base_branch_logits).sum(0)
        for key in diagnostics:
            if key == "source_weight_entropy":
                value = -(weights * weights.clamp_min(1e-12).log()).sum(0).mean()
            else:
                value = output.diagnostics[key]
            diagnostics[key].append(float(value.detach().cpu()))
        ys.append(y.numpy()); preds.append(output.logits.argmax(-1).cpu().numpy())
        base_preds.append(base_fused.argmax(-1).cpu().numpy()); logits_all.append(output.logits.cpu())
    yt, yp, logits = np.concatenate(ys), np.concatenate(preds), torch.cat(logits_all)
    result = _diagnostics(yt, yp, logits, num_classes)
    base_pred = np.concatenate(base_preds)
    corrected = (base_pred != yt) & (yp == yt)
    harmed = (base_pred == yt) & (yp != yt)
    result.update({
        "final_fused_prediction_flip_rate": float((yp != base_pred).mean()),
        "final_fused_corrected_errors": int(corrected.sum()),
        "final_fused_harmed_correct": int(harmed.sum()),
        "final_fused_net_corrections": int(corrected.sum() - harmed.sum()),
        "final_fused_net_correction_rate": float((corrected.sum() - harmed.sum()) / max(len(yt), 1)),
        "baseline_cosine_accuracy": float((base_pred == yt).mean()),
    })
    for key, values in diagnostics.items():
        result[key] = float(np.max(values) if key.endswith("_max") else np.mean(values))
    return result


def run(args):
    file_config = load_config(args.config)
    effective_config = resolve_c1_config(file_config, {
        "epochs": args.epochs, "batch_size": args.batch_size, "learning_rate": args.lr,
        "metric_lr_multiplier": args.metric_lr_multiplier,
        "tau": args.tau, "rank": args.rank, "gamma": args.gamma,
        "metric_epsilon": args.metric_epsilon, "lambda_H": args.lambda_h,
        "fusion_tau": args.fusion_tau, "fusion_mode": args.fusion_mode,
        "va_config": args.va_config,
    }, smoke=args.smoke)
    if args.variant not in effective_config["variants"]:
        raise ValueError(f"{args.variant} is not registered in {args.config}")
    args.epochs = int(effective_config["epochs"])
    args.batch_size = int(effective_config["batch_size"])
    args.lr = float(effective_config["learning_rate"])
    args.metric_lr_multiplier = float(effective_config["metric_lr_multiplier"])
    args.va_config = effective_config["va_config"]
    args.sample_length = int(effective_config["sample_length"])
    setup_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    data, labels, channels, bands, classes = _load_seediv(args)
    if classes != int(effective_config["num_classes"]): raise ValueError("unexpected SEED-IV class count")
    session, target = args.session - 1, args.target - 1
    source_ids = [x for x in range(15) if x != target]
    text_dim, vectors = label_to_vector("seediv_de_lds", "clip", device=device)
    prototypes = torch.tensor(np.asarray([vectors[i] for i in sorted(vectors)], np.float32), device=device)
    _, va = _load_reviewed_va(args.va_config, ["neutral", "sad", "fear", "happy"], device)
    b_sem, b_proto = semantic_basis(prototypes, va), prototype_basis(prototypes)
    basis = b_proto if args.variant == "E5" else b_sem
    loaders, centroid_loaders, weights = [], [], []
    for sid in source_ids:
        loader, y = make_loader(data[session][sid], labels[session][sid], args.batch_size, True, args.smoke, args.seed + sid)
        loaders.append(loader)
        centroid_loaders.append(DataLoader(loader.dataset, batch_size=args.batch_size, shuffle=False))
        weights.append(_class_weights(y, int(effective_config["num_classes"]), device))
    target_loader, _ = make_loader(data[session][target], labels[session][target], args.batch_size, False, args.smoke, args.seed)
    model = GeoSemSTDA(len(source_ids), channels, bands, int(effective_config["st_dim"]),
        int(effective_config["graph_dim"]), int(effective_config["adapter_bottleneck"]), text_dim,
        int(effective_config["graph_heads"]), graph_heads=int(effective_config["graph_heads"]),
        topk=int(effective_config["topk"]), dropout=float(effective_config["dropout"]),
        sample_length=int(effective_config["sample_length"]),
        representation_mode=effective_config["representation_mode"], classifier_type=effective_config["classifier_type"],
        num_classes=int(effective_config["num_classes"]),
        cast_variant=effective_config["backbone"]).to(device)
    mechanism = MultiSourceAffectiveMetric(len(source_ids), head_sharing=effective_config["head_sharing"],
        variant=args.variant, context_dim=int(effective_config["st_dim"]),
        prototypes=prototypes, basis=basis, rank=int(effective_config["rank"]),
        hidden_dim=int(effective_config["metric_hidden_dim"]),
        gamma=float(effective_config["gamma"]), bound_eps=float(effective_config["metric_epsilon"]),
        tau=float(effective_config["tau"]), lambda_regularization=float(effective_config["lambda_H"])).to(device)
    optimizer = _build_optimizer(model, mechanism, effective_config)
    result_family = "seediv_c1_smoke" if args.smoke else "seediv_c1"
    effective_hash = canonical_hash(effective_config)
    out = ROOT / "results" / result_family / effective_hash / args.variant / f"seed{args.seed}" / f"session{args.session}_target{args.target:02d}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"config_file": str(args.config), "file_config": file_config,
        "effective_config": effective_config, "effective_config_hash": effective_hash,
        "variant": args.variant, "seed": args.seed, "session": args.session,
        "target": args.target, "source_ids": [x + 1 for x in source_ids], "device": str(device),
        "B_sem": {"va_config": args.va_config, "method": "centered least-squares V-A to CLIP; QR"},
        "B_used": "B_proto" if args.variant == "E5" else "B_sem", "smoke": args.smoke,
        "optimizer_param_groups": [
            {"name": group["group_name"], "lr": float(group["lr"]),
             "weight_decay": float(optimizer.defaults["weight_decay"])}
            for group in optimizer.param_groups
        ],
        "base_learning_rate": args.lr,
        "metric_lr_multiplier": args.metric_lr_multiplier,
        "metric_learning_rate": args.lr * args.metric_lr_multiplier,
        "original_sgda_modified": False}
    (out / "run_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    steps = min(len(x) for x in loaders); best = None
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
    # Create shuffled iterators only after RNG restoration so resumed runs use
    # the same sampler state as uninterrupted runs.
    iters = [iter(x) for x in loaders]
    epochs = args.epochs
    for epoch in range(start_epoch, epochs + 1):
        model.train(); mechanism.train()
        gradient_norms, metric_head_gradient_norms = [], []
        train_ce_values, train_regularization_values = [], []
        train_h_spectral_values, train_h_ratio_values = [], []
        for _ in range(steps):
            batches = []
            for i, iterator in enumerate(iters):
                try: batches.append(next(iterator))
                except StopIteration: iters[i] = iter(loaders[i]); batches.append(next(iters[i]))
            xs = [b[0].to(device) for b in batches]; rs = [b[1].to(device) for b in batches]; ys = [b[2].to(device) for b in batches]
            optimizer.zero_grad(); z, _, h, _, _, _ = model(xs, rs, return_features=True)
            if effective_config["training_path"] == "branch_supervision":
                outputs = mechanism.source_outputs(h, z)
                batch_outputs = outputs
                ce_losses = [F.cross_entropy(o.logits, y, weight=w) for o, y, w in zip(outputs, ys, weights)]
                regularization_losses = [o.regularization for o in outputs]
            elif effective_config["training_path"] == "fused_uniform_direct":
                if effective_config["head_sharing"] != "shared":
                    raise ValueError("fused_uniform_direct requires head_sharing=shared")
                ce_losses, regularization_losses, batch_outputs = [], [], []
                for source_h, y, class_weight in zip(h, ys, weights):
                    branch_z = model.project_all_adapters(source_h)
                    uniform = source_h.new_full((len(source_ids), len(source_h)), 1.0 / len(source_ids))
                    output = mechanism.fused_output(source_h, branch_z, uniform, "shared_fused_embedding")
                    batch_outputs.append(output)
                    ce_losses.append(F.cross_entropy(output.logits, y, weight=class_weight))
                    regularization_losses.append(output.regularization)
            else:
                raise ValueError(f"unsupported training_path: {effective_config['training_path']}")
            train_ce_values.append(float(torch.stack(ce_losses).mean().detach().cpu()))
            train_regularization_values.append(float(torch.stack(regularization_losses).mean().detach().cpu()))
            spectral_rows = [o.diagnostics["H_spectral_norm"].reshape(-1) for o in batch_outputs
                if "H_spectral_norm" in o.diagnostics]
            ratio_rows = [o.diagnostics["H_spectral_ratio"].reshape(-1) for o in batch_outputs
                if "H_spectral_ratio" in o.diagnostics]
            train_h_spectral_values.append(float(torch.cat(spectral_rows).mean().detach().cpu()) if spectral_rows else 0.0)
            train_h_ratio_values.append(float(torch.cat(ratio_rows).mean().detach().cpu()) if ratio_rows else 0.0)
            loss = torch.stack([ce + reg for ce, reg in zip(ce_losses, regularization_losses)]).mean()
            loss.backward()
            grad_sq = sum(float(parameter.grad.detach().square().sum().cpu())
                          for parameter in mechanism.parameters() if parameter.grad is not None)
            gradient_norms.append(grad_sq ** 0.5)
            metric_grad_sq = sum(float(parameter.grad.detach().square().sum().cpu())
                for name, parameter in mechanism.named_parameters()
                if "metric_generator" in name and parameter.grad is not None)
            metric_head_gradient_norms.append(metric_grad_sq ** 0.5)
            optimizer.step()
        # no_grad alone does not disable Dropout.  Compute centroids and target
        # features under the same deterministic evaluation-mode distribution.
        model.eval(); mechanism.eval()
        with torch.no_grad():
            centroids = compute_source_domain_centroids(model, centroid_loaders, device)
        metrics = evaluate(model, mechanism, target_loader, centroids, device,
            fusion_tau=float(effective_config["fusion_tau"]),
            fusion_mode=effective_config["fusion_mode"],
            num_classes=int(effective_config["num_classes"]))
        metrics["mechanism_gradient_norm_mean"] = float(np.mean(gradient_norms)) if gradient_norms else 0.0
        metrics["mechanism_gradient_norm_max"] = float(np.max(gradient_norms)) if gradient_norms else 0.0
        metrics["metric_head_gradient_norm_mean"] = float(np.mean(metric_head_gradient_norms)) if metric_head_gradient_norms else 0.0
        metrics["metric_head_gradient_norm_max"] = float(np.max(metric_head_gradient_norms)) if metric_head_gradient_norms else 0.0
        metrics["training_ce_mean"] = float(np.mean(train_ce_values)) if train_ce_values else 0.0
        metrics["training_regularization_mean"] = float(np.mean(train_regularization_values)) if train_regularization_values else 0.0
        metrics["training_objective_mean"] = metrics["training_ce_mean"] + metrics["training_regularization_mean"]
        metrics["training_H_spectral_norm_mean"] = float(np.mean(train_h_spectral_values)) if train_h_spectral_values else 0.0
        metrics["training_H_spectral_norm_max"] = float(np.max(train_h_spectral_values)) if train_h_spectral_values else 0.0
        metrics["training_H_spectral_ratio_gamma_mean"] = float(np.mean(train_h_ratio_values)) if train_h_ratio_values else 0.0
        metrics["backbone_learning_rate"] = args.lr
        metrics["metric_learning_rate"] = args.lr * args.metric_lr_multiplier
        metrics["metric_lr_multiplier"] = args.metric_lr_multiplier
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
    p.add_argument("--seed", type=int, default=42); p.add_argument("--epochs", type=int); p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float); p.add_argument("--tau", type=float); p.add_argument("--rank", type=int); p.add_argument("--gamma", type=float)
    p.add_argument("--metric-lr-multiplier", type=float)
    p.add_argument("--metric-epsilon", type=float); p.add_argument("--lambda-h", type=float); p.add_argument("--fusion-tau", type=float)
    p.add_argument("--fusion-mode", choices=("branch_logits", "legacy_fused_embedding"))
    p.add_argument("--device", default="cuda:0"); p.add_argument("--data-load-workers", type=int, default=0)
    p.add_argument("--va-config"); p.add_argument("--smoke", action="store_true")
    p.add_argument("--resume", action="store_true")
    run(p.parse_args())
