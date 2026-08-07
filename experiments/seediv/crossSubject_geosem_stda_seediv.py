"""
GeoSem-STDA / ReSGCA on SEED-IV, subject-independent LOSO.

This script mirrors the DEAP GeoSem-STDA protocol but adapts it to SEED-IV:
  - target subject labels are used only for evaluation;
  - each non-target subject is one source domain;
  - SEED-IV has 4 semantic emotion classes and 62 EEG channels;
  - ReSGCA is available through --mmd_type resgca.
"""

import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from config.setting import preset_setting, set_setting_by_args
from data_utils.constants.path_mapper import path_mapper
from data_utils.load_data import get_data
from data_utils.text_to_vector import label_to_vector
from experiments.deap.crossSubject_geosem_stda_deap import (
    _compute_lambda_value,
    _compute_mmd_loss,
    _compute_sca_mu,
)
from models.geosem_stda import (
    GeoSemSTDA,
    compute_source_class_centroids,
    log_euclidean_reference,
    predict_class_aware,
    prototype_contrastive_loss,
    tangent_deviation,
)
from utils.args import get_args_parser
from utils.mix_utils import flatten_trials, setup_seed, zscore_subject_wise


def _append_csv_row(path, fieldnames, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _format_float(value):
    return str(value).replace(".", "p").replace("-", "m")


def _parse_subject_ids(subject_ids):
    if subject_ids is None:
        return None
    parsed = []
    for item in subject_ids:
        for part in str(item).split(","):
            part = part.strip()
            if part:
                parsed.append(int(part) - 1)
    if not parsed:
        return None
    if any(idx < 0 for idx in parsed):
        raise ValueError("subject ids are 1-based and must be >= 1")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate subject id in --target_subject_ids: {subject_ids}")
    return parsed


def _select_target_indices(n_subjects, target_subject_ids=None, random_target_count=None, target_seed=42):
    target_ids = _parse_subject_ids(target_subject_ids)
    if target_ids is not None:
        if any(idx >= n_subjects for idx in target_ids):
            raise ValueError(f"target subject id out of range 1..{n_subjects}: {target_subject_ids}")
        return target_ids
    if random_target_count is not None:
        if not (1 <= random_target_count <= n_subjects):
            raise ValueError(f"random_target_count must be in [1, {n_subjects}]")
        rng = np.random.default_rng(target_seed)
        return sorted(rng.choice(n_subjects, size=random_target_count, replace=False).tolist())
    return list(range(n_subjects))


def _compute_class_weights(labels, num_classes, device):
    y = np.asarray(labels).reshape(-1)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _make_model(args, n_sources, channels, num_freq_bands, text_dim):
    return GeoSemSTDA(
        n_sources=n_sources,
        num_electrodes=channels,
        num_freq_bands=num_freq_bands,
        st_dim=args.st_dim,
        graph_dim=args.graph_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        text_dim=text_dim,
        heads=args.heads,
        topk=args.topk,
        dropout=args.dropout,
    ).to(args.device)


@torch.no_grad()
def _stream_log_reference(session_data, source_ids, device, shrinkage, eps, batch_size):
    log_sum = None
    total = 0
    for sid in source_ids:
        x = torch.tensor(np.asarray(session_data[sid]), dtype=torch.float32, device=device)
        for start in range(0, x.size(0), batch_size):
            log_part = log_euclidean_reference(
                [x[start:start + batch_size]],
                shrinkage=shrinkage,
                eps=eps,
            )
            n_part = x[start:start + batch_size].size(0)
            log_sum = log_part * n_part if log_sum is None else log_sum + log_part * n_part
            total += n_part
    return log_sum / max(total, 1)


@torch.no_grad()
def build_geometry_for_fold(session_data, source_ids, target_id, device, shrinkage, eps, batch_size):
    log_ref = _stream_log_reference(session_data, source_ids, device, shrinkage, eps, batch_size)
    r_by_subject = {}
    for sid in source_ids + [target_id]:
        x = torch.tensor(np.asarray(session_data[sid]), dtype=torch.float32, device=device)
        parts = []
        for start in range(0, x.size(0), batch_size):
            parts.append(tangent_deviation(
                x[start:start + batch_size],
                log_ref,
                shrinkage=shrinkage,
                eps=eps,
            ).cpu())
        r_by_subject[sid] = torch.cat(parts, dim=0)
    return r_by_subject


def _train_selection_warmup(model, source_loaders, target_loader, text_prototypes, class_weights, num_classes, args):
    if args.reliability_warmup_epochs <= 0:
        return
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = min(len(loader) for loader in source_loaders)
    total_steps = max(args.reliability_warmup_epochs * steps_per_epoch, 1)
    source_iters = [iter(loader) for loader in source_loaders]
    target_iter = iter(target_loader)
    step = 0

    print(f"Reliability warm-up: {args.reliability_warmup_epochs} epochs, {steps_per_epoch} steps/epoch")
    for epoch in range(args.reliability_warmup_epochs):
        model.train()
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            step += 1
            x_src_list, r_src_list, y_src_list = [], [], []
            for src_idx, src_iter in enumerate(source_iters):
                try:
                    xb, rb, yb = next(src_iter)
                except StopIteration:
                    source_iters[src_idx] = iter(source_loaders[src_idx])
                    xb, rb, yb = next(source_iters[src_idx])
                x_src_list.append(xb.to(args.device))
                r_src_list.append(rb.to(args.device))
                y_src_list.append(yb.to(args.device))

            try:
                x_tb, r_tb, _ = next(target_iter)
            except StopIteration:
                target_iter = iter(target_loader)
                x_tb, r_tb, _ = next(target_iter)

            optimizer.zero_grad()
            z_src_all, z_tgt_all = model(
                x_src_list,
                r_src_list,
                x_tb.to(args.device),
                r_tb.to(args.device),
            )
            loss_proto = prototype_contrastive_loss(
                z_src_all,
                y_src_list,
                text_prototypes,
                tau=args.proto_tau,
                class_weights_list=class_weights,
            )
            loss_mmd = _compute_mmd_loss(
                z_src_all,
                z_tgt_all,
                y_src_list,
                text_prototypes,
                num_classes,
                args,
                step=step,
                total_steps=total_steps,
                r_src_list=r_src_list,
                r_tgt=r_tb.to(args.device),
            )
            loss = loss_proto + _compute_lambda_value(step, total_steps, args) * loss_mmd
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  warm-up epoch {epoch + 1}: loss={epoch_loss / steps_per_epoch:.4f}")


@torch.no_grad()
def _collect_source_embeddings(model, source_loaders, device):
    source_z, source_y = [], []
    model.eval()
    for source_idx, loader in enumerate(source_loaders):
        z_parts, y_parts = [], []
        for xb, rb, yb in loader:
            h, _ = model.encode(xb.to(device), rb.to(device))
            z = model.prototype_head(model.adapters[source_idx](h))
            z_parts.append(z)
            y_parts.append(yb.to(device))
        source_z.append(torch.cat(z_parts, dim=0))
        source_y.append(torch.cat(y_parts, dim=0))
    return source_z, source_y


@torch.no_grad()
def _collect_target_embeddings(model, target_loader, device):
    target_by_source = None
    model.eval()
    for xb, rb, _ in target_loader:
        _, z_tgt_all = model([], [], xb.to(device), rb.to(device))
        if target_by_source is None:
            target_by_source = [[] for _ in z_tgt_all]
        for idx, z in enumerate(z_tgt_all):
            target_by_source[idx].append(z)
    return [torch.cat(parts, dim=0) for parts in target_by_source]


@torch.no_grad()
def sparse_reliability_source_selection(model, source_loaders, target_loader, source_ids, text_prototypes, num_classes, args):
    source_z, source_y = _collect_source_embeddings(model, source_loaders, args.device)
    target_z = _collect_target_embeddings(model, target_loader, args.device)
    text_prototypes = torch.nn.functional.normalize(text_prototypes, dim=-1)
    scores, details = [], []

    for local_idx, sid in enumerate(source_ids):
        z_s = source_z[local_idx]
        y_s = source_y[local_idx]
        z_t = target_z[local_idx]
        d_marg = ((z_s.mean(dim=0) - z_t.mean(dim=0)) ** 2).sum()
        q_t = torch.softmax(z_t @ text_prototypes.T / args.proto_tau, dim=-1)

        target_centers, source_centers, cond_weights = [], [], []
        for cls in range(num_classes):
            cls_weight = q_t[:, cls].sum().clamp_min(1e-6)
            target_centers.append((q_t[:, cls:cls + 1] * z_t).sum(dim=0) / cls_weight)
            mask = y_s == cls
            source_centers.append(z_s[mask].mean(dim=0) if mask.any() else z_s.mean(dim=0))
            cond_weights.append(cls_weight / q_t.size(0))
        d_cond = (
            torch.stack(cond_weights)
            * ((torch.stack(source_centers) - torch.stack(target_centers)) ** 2).sum(dim=-1)
        ).sum()

        logits_s = z_s @ text_prototypes.T / args.proto_tau
        source_acc_proxy = (logits_s.argmax(dim=-1) == y_s).float().mean()
        score = -args.rel_marg_weight * d_marg - args.rel_cond_weight * d_cond + args.rel_val_weight * source_acc_proxy
        scores.append(score)
        details.append({
            "sid": sid,
            "d_marg": float(d_marg.cpu()),
            "d_cond": float(d_cond.cpu()),
            "src_acc_proxy": float(source_acc_proxy.cpu()),
            "score": float(score.cpu()),
        })

    weights = torch.softmax(torch.stack(scores) / args.source_weight_tau, dim=0)
    order = torch.argsort(weights, descending=True)
    selected_positions = order[:min(args.sparse_k_max, len(order))].tolist()
    selected_ids = [source_ids[pos] for pos in selected_positions]
    selected_weights = weights[selected_positions]
    selected_weights = selected_weights / selected_weights.sum().clamp_min(1e-8)

    print("Sparse reliability source selection:")
    for rank, pos in enumerate(selected_positions, start=1):
        item = details[pos]
        print(
            f"  #{rank} S{item['sid'] + 1}: weight={float(selected_weights[rank - 1].cpu()):.4f}, "
            f"score={item['score']:.4f}, marg={item['d_marg']:.4f}, "
            f"cond={item['d_cond']:.4f}, src_acc={item['src_acc_proxy']:.4f}"
        )
    return selected_ids, selected_weights.detach().cpu().tolist()


def load_seediv(args):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)
    setting.dataset_path = path_mapper["seediv_de_lds"]
    setting.dataset = "seediv_de_lds"
    setting.experiment_mode = "subject-independent"
    setting.onehot = False
    setting.sample_length = args.sample_length
    setting.sessions = [1, 2, 3]
    setting.stride = args.stride

    data, label, channels, num_freq_bands, num_classes = get_data(setting)
    text_dim, class_vectors = label_to_vector(
        dataset=setting.dataset,
        LM=args.LM,
        LabelTextMapper=None,
        device=args.device,
    )
    data, label = flatten_trials(data, label)
    if args.subject_zscore:
        data = zscore_subject_wise(data)
        print("Applied subject-wise z-score normalization.")
    return data, label, class_vectors, channels, num_freq_bands, num_classes, text_dim


def evaluate(model, loader, text_prototypes, source_centroids, device, proto_tau, fusion_tau):
    true_parts, pred_parts = predict_class_aware(
        model,
        loader,
        text_prototypes,
        source_centroids,
        device=device,
        proto_tau=proto_tau,
        fusion_tau=fusion_tau,
    )
    return np.concatenate(true_parts), np.concatenate(pred_parts)


def run(args):
    setup_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.dataset = "seediv"
    args.LM = "clip"
    args.sample_length = 3 if args.sample_length is None else args.sample_length
    args.stride = 1 if args.stride is None else args.stride
    args.epochs = 200 if args.epochs is None else args.epochs
    args.batch_size = 64 if args.batch_size is None else args.batch_size
    args.lr = 1e-3 if args.lr is None else args.lr

    if args.st_dim % args.heads != 0:
        raise ValueError(f"st_dim={args.st_dim} must be divisible by heads={args.heads}")
    if args.mmd_warmup_ratio <= 0.0 or args.mmd_warmup_ratio > 1.0:
        raise ValueError(f"mmd_warmup_ratio must be in (0, 1], got {args.mmd_warmup_ratio}")
    if args.mmd_hold_ratio < args.mmd_warmup_ratio or args.mmd_hold_ratio > 1.0:
        raise ValueError(f"mmd_hold_ratio must be in [{args.mmd_warmup_ratio}, 1], got {args.mmd_hold_ratio}")
    if args.resgca_geo_tau <= 0.0:
        raise ValueError(f"resgca_geo_tau must be positive, got {args.resgca_geo_tau}")
    if args.resgca_geo_weight < 0.0:
        raise ValueError(f"resgca_geo_weight must be >= 0, got {args.resgca_geo_weight}")

    output_dir = os.path.join(project_root, "results", "results_seediv_geosem_stda")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"seediv_ep{args.epochs}_bs{args.batch_size}_lr{_format_float(args.lr)}_"
        f"lmda{_format_float(args.lambda_max)}_{args.mmd_type}_{args.mmd_schedule}_seed{args.seed}_{timestamp}"
    )
    run_dir = os.path.join(output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    epoch_log_path = os.path.join(run_dir, "epoch_log.csv")
    subject_csv = os.path.join(run_dir, "subject_results_seediv_geosem_stda.csv")
    summary_path = os.path.join(output_dir, "summary_seediv_geosem_stda_runs.csv")
    config_path = os.path.join(run_dir, "run_config.json")

    print(f"Device: {args.device}")
    print(f"Run directory: {run_dir}")
    data, label, class_vectors, channels, num_freq_bands, num_classes, text_dim = load_seediv(args)
    if args.topk >= channels:
        raise ValueError(f"topk={args.topk} must be smaller than channels={channels}")
    text_prototypes = torch.tensor(
        np.asarray([class_vectors[i] for i in sorted(class_vectors.keys())], dtype=np.float32),
        device=args.device,
    )
    n_sessions = len(data)
    n_subjects = len(data[0])
    target_indices = _select_target_indices(
        n_subjects,
        target_subject_ids=args.target_subject_ids,
        random_target_count=args.random_target_count,
        target_seed=args.target_seed,
    )
    print(f"Targets evaluated: {[idx + 1 for idx in target_indices]}")
    print(f"SEED-IV dimensions: X=[N,{args.sample_length},{channels},{num_freq_bands}], classes={num_classes}")

    run_config = {
        "run_id": run_id,
        "script": os.path.abspath(__file__),
        "dataset": "seediv_de_lds",
        "evaluated_target_subjects": [idx + 1 for idx in target_indices],
        "source_candidate_count_per_target": n_subjects - 1,
        "sessions": list(range(n_sessions)),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "lambda_max": args.lambda_max,
        "lambda_min": args.lambda_min,
        "mmd_type": args.mmd_type,
        "mmd_schedule": args.mmd_schedule,
        "mmd_confidence_gate": args.mmd_confidence_gate,
        "sca_mu_start": args.sca_mu_start,
        "sca_mu_end": args.sca_mu_end,
        "sca_mu_warmup_ratio": args.sca_mu_warmup_ratio,
        "resgca_geo_tau": args.resgca_geo_tau,
        "resgca_geo_weight": args.resgca_geo_weight,
        "st_dim": args.st_dim,
        "graph_dim": args.graph_dim,
        "adapter_bottleneck": args.adapter_bottleneck,
        "topk": args.topk,
        "channels": channels,
        "num_freq_bands": num_freq_bands,
        "num_classes": num_classes,
        "text_dim": text_dim,
    }
    _write_json(config_path, run_config)

    epoch_fields = [
        "run_id", "session_idx", "target_subject", "epoch",
        "loss", "proto", "align", "lambda", "sca_mu", "acc", "macro_f1", "micro_f1",
        "best_acc", "alpha_mean", "final_source_count",
    ]
    subject_fields = [
        "run_id", "session_idx", "target_subject", "source_candidate_count",
        "final_source_count", "acc", "macro_f1", "micro_f1",
    ]
    subject_records = []
    results_acc, results_macro, results_micro = {}, {}, {}

    for session_idx in range(n_sessions):
        session_acc, session_macro, session_micro = [], [], []
        for target_sub in target_indices:
            setup_seed(args.seed)
            all_source_ids = [sid for sid in range(n_subjects) if sid != target_sub]
            source_ids = all_source_ids
            print(f"\n{'=' * 60}")
            print(
                f"Session {session_idx} | Target subject {target_sub + 1}/{n_subjects} "
                f"| Sources used: {len(source_ids)}/{len(all_source_ids)}"
            )
            print(f"{'=' * 60}")

            r_by_subject = build_geometry_for_fold(
                data[session_idx],
                source_ids,
                target_sub,
                args.device,
                shrinkage=args.shrinkage,
                eps=args.spd_eps,
                batch_size=args.geometry_batch_size,
            )

            source_loaders, source_class_weights = [], []
            for sid in source_ids:
                x = torch.tensor(np.asarray(data[session_idx][sid]), dtype=torch.float32)
                r = r_by_subject[sid].float()
                y = torch.tensor(np.asarray(label[session_idx][sid]).reshape(-1), dtype=torch.long)
                class_weights = _compute_class_weights(label[session_idx][sid], num_classes, args.device)
                source_class_weights.append(class_weights)
                source_loaders.append(DataLoader(
                    TensorDataset(x, r, y),
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0,
                    pin_memory=True,
                ))

            x_tgt = torch.tensor(np.asarray(data[session_idx][target_sub]), dtype=torch.float32)
            r_tgt = r_by_subject[target_sub].float()
            y_tgt = torch.tensor(np.asarray(label[session_idx][target_sub]).reshape(-1), dtype=torch.long)
            target_train_loader = DataLoader(
                TensorDataset(x_tgt, r_tgt, y_tgt),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )
            target_eval_loader = DataLoader(
                TensorDataset(x_tgt, r_tgt, y_tgt),
                batch_size=args.batch_size,
                shuffle=False,
                drop_last=False,
                num_workers=0,
                pin_memory=True,
            )

            source_loss_weights = None
            if args.source_selection == "sparse_reliability" and len(source_loaders) > 1:
                warmup_model = _make_model(args, len(source_loaders), channels, num_freq_bands, text_dim)
                _train_selection_warmup(
                    warmup_model,
                    source_loaders,
                    target_train_loader,
                    text_prototypes,
                    source_class_weights,
                    num_classes,
                    args,
                )
                selected_ids, selected_weights = sparse_reliability_source_selection(
                    warmup_model,
                    source_loaders,
                    target_eval_loader,
                    source_ids,
                    text_prototypes,
                    num_classes,
                    args,
                )
                selected_set = set(selected_ids)
                keep_indices = [idx for idx, sid in enumerate(source_ids) if sid in selected_set]
                source_ids = [source_ids[idx] for idx in keep_indices]
                source_loaders = [source_loaders[idx] for idx in keep_indices]
                source_class_weights = [source_class_weights[idx] for idx in keep_indices]
                weight_by_id = {sid: weight for sid, weight in zip(selected_ids, selected_weights)}
                source_loss_weights = [weight_by_id[sid] for sid in source_ids]
                print(f"Sparse selected sources used for final training: {[sid + 1 for sid in source_ids]}")

            model = _make_model(args, len(source_loaders), channels, num_freq_bands, text_dim)
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
            steps_per_epoch = min(len(loader) for loader in source_loaders)
            total_steps = max(args.epochs * steps_per_epoch, 1)
            source_iters = [iter(loader) for loader in source_loaders]
            target_iter = iter(target_train_loader)
            global_step = 0
            best_acc, best_macro, best_micro = 0.0, 0.0, 0.0

            for epoch in range(args.epochs):
                model.train()
                epoch_loss, epoch_proto, epoch_align = 0.0, 0.0, 0.0
                alpha_values = []
                sca_mu_val = np.nan

                for _ in range(steps_per_epoch):
                    global_step += 1
                    x_src_list, r_src_list, y_src_list = [], [], []
                    for src_idx, src_iter in enumerate(source_iters):
                        try:
                            xb, rb, yb = next(src_iter)
                        except StopIteration:
                            source_iters[src_idx] = iter(source_loaders[src_idx])
                            xb, rb, yb = next(source_iters[src_idx])
                        x_src_list.append(xb.to(args.device))
                        r_src_list.append(rb.to(args.device))
                        y_src_list.append(yb.to(args.device))

                    try:
                        x_tb, r_tb, _ = next(target_iter)
                    except StopIteration:
                        target_iter = iter(target_train_loader)
                        x_tb, r_tb, _ = next(target_iter)

                    optimizer.zero_grad()
                    z_src_all, z_tgt_all, _, _, alpha_src, alpha_tgt = model(
                        x_src_list,
                        r_src_list,
                        x_tb.to(args.device),
                        r_tb.to(args.device),
                        return_features=True,
                    )
                    loss_proto = prototype_contrastive_loss(
                        z_src_all,
                        y_src_list,
                        text_prototypes,
                        tau=args.proto_tau,
                        class_weights_list=source_class_weights,
                        source_weights=source_loss_weights,
                    )
                    loss_align = _compute_mmd_loss(
                        z_src_all,
                        z_tgt_all,
                        y_src_list,
                        text_prototypes,
                        num_classes,
                        args,
                        source_weights=source_loss_weights,
                        step=global_step,
                        total_steps=total_steps,
                        r_src_list=r_src_list,
                        r_tgt=r_tb.to(args.device),
                    )
                    lambda_val = _compute_lambda_value(global_step, total_steps, args)
                    sca_mu_val = (
                        _compute_sca_mu(global_step, total_steps, args)
                        if args.mmd_type in ("sca", "resgca")
                        else np.nan
                    )
                    loss = loss_proto + lambda_val * loss_align
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_proto += loss_proto.item()
                    epoch_align += loss_align.item()
                    if alpha_src:
                        alpha_values.append(torch.cat([a.detach().cpu() for a in alpha_src]).mean().item())
                    alpha_values.append(alpha_tgt.detach().cpu().mean().item())

                mean_alpha = float(np.mean(alpha_values)) if alpha_values else 0.0
                should_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch + 1 == args.epochs)
                acc, macro_f1, micro_f1 = np.nan, np.nan, np.nan
                if should_eval:
                    model.eval()
                    source_centroids = compute_source_class_centroids(model, source_loaders, args.device, num_classes)
                    y_true, y_pred = evaluate(
                        model,
                        target_eval_loader,
                        text_prototypes,
                        source_centroids,
                        args.device,
                        proto_tau=args.proto_tau,
                        fusion_tau=args.fusion_tau,
                    )
                    acc = float((y_true == y_pred).mean())
                    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
                    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
                    if acc > best_acc:
                        best_acc, best_macro, best_micro = acc, macro_f1, micro_f1
                        print(f"  >> New best acc: {best_acc:.4f} (epoch {epoch + 1})")

                if (epoch + 1) % args.log_interval == 0:
                    acc_text = f"{acc:.4f}" if should_eval else "skip"
                    sca_mu_text = f" mu={sca_mu_val:.3f}" if args.mmd_type in ("sca", "resgca") else ""
                    print(
                        f"Ep {epoch + 1:3d} | loss={epoch_loss / steps_per_epoch:.4f} "
                        f"proto={epoch_proto / steps_per_epoch:.4f} "
                        f"align={epoch_align / steps_per_epoch:.6f} "
                        f"lambda={lambda_val:.4f}{sca_mu_text} acc={acc_text} "
                        f"best={best_acc:.4f} alpha={mean_alpha:.3f}"
                    )

                _append_csv_row(epoch_log_path, epoch_fields, {
                    "run_id": run_id,
                    "session_idx": session_idx,
                    "target_subject": target_sub + 1,
                    "epoch": epoch + 1,
                    "loss": epoch_loss / steps_per_epoch,
                    "proto": epoch_proto / steps_per_epoch,
                    "align": epoch_align / steps_per_epoch,
                    "lambda": lambda_val,
                    "sca_mu": sca_mu_val,
                    "acc": acc,
                    "macro_f1": macro_f1,
                    "micro_f1": micro_f1,
                    "best_acc": best_acc,
                    "alpha_mean": mean_alpha,
                    "final_source_count": len(source_loaders),
                })

            print(f"Final target {target_sub + 1}: acc={best_acc * 100:.2f}%")
            session_acc.append(best_acc)
            session_macro.append(best_macro)
            session_micro.append(best_micro)
            subject_records.append({
                "run_id": run_id,
                "session_idx": session_idx,
                "target_subject": target_sub + 1,
                "source_candidate_count": n_subjects - 1,
                "final_source_count": len(source_ids),
                "acc": best_acc,
                "macro_f1": best_macro,
                "micro_f1": best_micro,
            })

        results_acc[session_idx] = session_acc
        results_macro[session_idx] = session_macro
        results_micro[session_idx] = session_micro

    with open(subject_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=subject_fields)
        writer.writeheader()
        writer.writerows(subject_records)

    all_acc = np.asarray([row["acc"] for row in subject_records], dtype=np.float32)
    all_macro = np.asarray([row["macro_f1"] for row in subject_records], dtype=np.float32)
    all_micro = np.asarray([row["micro_f1"] for row in subject_records], dtype=np.float32)
    summary = {
        "run_id": run_id,
        "created_at": timestamp,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "lambda_max": args.lambda_max,
        "lambda_min": args.lambda_min,
        "mmd_type": args.mmd_type,
        "mmd_schedule": args.mmd_schedule,
        "resgca_geo_tau": args.resgca_geo_tau,
        "resgca_geo_weight": args.resgca_geo_weight,
        "acc_mean": float(all_acc.mean()),
        "acc_std": float(all_acc.std(ddof=1)) if all_acc.size > 1 else 0.0,
        "macro_f1_mean": float(all_macro.mean()),
        "macro_f1_std": float(all_macro.std(ddof=1)) if all_macro.size > 1 else 0.0,
        "micro_f1_mean": float(all_micro.mean()),
        "micro_f1_std": float(all_micro.std(ddof=1)) if all_micro.size > 1 else 0.0,
        "subject_results_csv": subject_csv,
        "epoch_log_csv": epoch_log_path,
        "run_config_json": config_path,
    }
    _append_csv_row(summary_path, list(summary.keys()), summary)
    print(f"\nOverall Acc: {all_acc.mean() * 100:.2f}% +/- {all_acc.std() * 100:.2f}%")
    print(f"Results saved to {subject_csv}")
    print(f"Summary appended to {summary_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    parser.set_defaults(epochs=200, batch_size=64, lr=1e-3, sample_length=3, stride=1)
    parser.add_argument("--lambda_max", type=float, default=0.2)
    parser.add_argument("--lambda_min", type=float, default=0.05)
    parser.add_argument("--mmd_type", type=str, default="resgca", choices=["marginal", "class_aware", "sca", "resgca"])
    parser.add_argument("--mmd_schedule", type=str, default="warmup_cosine_decay",
                        choices=["monotonic", "warmup_hold", "warmup_decay", "warmup_cosine_decay"])
    parser.add_argument("--mmd_warmup_ratio", type=float, default=0.2)
    parser.add_argument("--mmd_hold_ratio", type=float, default=0.5)
    parser.add_argument("--mmd_confidence_gate", type=str, default="entropy",
                        choices=["none", "soft", "threshold", "entropy"])
    parser.add_argument("--mmd_confidence_threshold", type=float, default=0.6)
    parser.add_argument("--sca_mu_start", type=float, default=0.0)
    parser.add_argument("--sca_mu_end", type=float, default=1.0)
    parser.add_argument("--sca_mu_warmup_ratio", type=float, default=0.5)
    parser.add_argument("--resgca_geo_tau", type=float, default=1.0)
    parser.add_argument("--resgca_geo_weight", type=float, default=1.0)
    parser.add_argument("--proto_tau", type=float, default=0.07)
    parser.add_argument("--fusion_tau", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--spd_eps", type=float, default=1e-5)
    parser.add_argument("--geometry_batch_size", type=int, default=128)
    parser.add_argument("--st_dim", type=int, default=128)
    parser.add_argument("--graph_dim", type=int, default=64)
    parser.add_argument("--adapter_bottleneck", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--source_selection", type=str, default="sparse_reliability",
                        choices=["none", "sparse_reliability"])
    parser.add_argument("--reliability_warmup_epochs", type=int, default=5)
    parser.add_argument("--sparse_k_max", type=int, default=6)
    parser.add_argument("--source_weight_tau", type=float, default=0.5)
    parser.add_argument("--rel_marg_weight", type=float, default=1.0)
    parser.add_argument("--rel_cond_weight", type=float, default=1.0)
    parser.add_argument("--rel_val_weight", type=float, default=0.2)
    parser.add_argument("--subject_zscore", action="store_true", default=True)
    parser.add_argument("--no_subject_zscore", action="store_false", dest="subject_zscore")
    parser.add_argument("--target_subject_ids", type=str, nargs="+", default=None)
    parser.add_argument("--random_target_count", type=int, default=None)
    parser.add_argument("--target_seed", type=int, default=42)
    run(parser.parse_args())
