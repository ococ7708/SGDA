"""
GeoSem-STDA on DEAP, subject-independent LOSO.

This script keeps the existing DEAP/SGDA protocol:
  - target subject labels are used only for evaluation;
  - each non-target subject is one source domain;
  - text prototypes are frozen vectors from data_utils.text_to_vector.

The training objective is intentionally simple:
    L_total = L_proto + lambda(p) * L_mmd
"""

import csv
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
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
from models.geosem_stda import (
    GeoSemSTDA,
    compute_source_class_centroids,
    lambda_warmup,
    log_euclidean_reference,
    multisource_class_aware_mmd,
    multisource_mmd,
    multisource_resgca,
    multisource_semantic_conditional_alignment,
    predict_class_aware,
    prototype_contrastive_loss,
    tangent_deviation,
)
from utils.args import get_args_parser
from utils.log_utils import save_csubs_results_csv
from utils.mix_utils import flatten_trials, setup_seed, zscore_subject_wise


def _format_float(value):
    return str(value).replace(".", "p").replace("-", "m")


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


def _optional_int(args, name):
    return getattr(args, name, None)


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
        raise ValueError("subject_ids are 1-based and must be >= 1")
    if len(set(parsed)) != len(parsed):
        raise ValueError(f"Duplicate subject id in --subject_ids: {subject_ids}")
    return parsed


def _select_target_indices(n_subjects, target_subject_ids=None, random_target_count=None, target_seed=42):
    target_indices = _parse_subject_ids(target_subject_ids)
    if target_indices is not None and random_target_count is not None:
        raise ValueError("Use either --target_subject_ids or --random_target_count, not both")

    if random_target_count is not None:
        if random_target_count <= 0 or random_target_count > n_subjects:
            raise ValueError(f"random_target_count must be in [1, {n_subjects}], got {random_target_count}")
        rng = np.random.default_rng(target_seed)
        target_indices = sorted(rng.choice(np.arange(n_subjects), size=random_target_count, replace=False).tolist())
        print(f"Random target subset (seed={target_seed}): {[idx + 1 for idx in target_indices]}")

    if target_indices is None:
        return list(range(n_subjects))

    invalid = [idx + 1 for idx in target_indices if idx >= n_subjects]
    if invalid:
        raise ValueError(f"Invalid target subject ids {invalid}; dataset has {n_subjects} subjects")
    return target_indices


def _limit_subjects_and_trials(
    data,
    label,
    max_subjects=None,
    max_trials=None,
    subject_ids=None,
    random_subject_count=None,
    subject_seed=42,
):
    subject_ids = _parse_subject_ids(subject_ids)
    if subject_ids is not None and random_subject_count is not None:
        raise ValueError("Use either --subject_ids or --random_subject_count, not both")

    if random_subject_count is not None:
        available = len(data[0])
        if random_subject_count <= 0 or random_subject_count > available:
            raise ValueError(f"random_subject_count must be in [1, {available}], got {random_subject_count}")
        rng = np.random.default_rng(subject_seed)
        subject_ids = sorted(rng.choice(np.arange(available), size=random_subject_count, replace=False).tolist())
        print(f"Random subject subset (seed={subject_seed}): {[idx + 1 for idx in subject_ids]}")

    if subject_ids is not None:
        max_subject_count = len(data[0])
        invalid = [idx + 1 for idx in subject_ids if idx >= max_subject_count]
        if invalid:
            raise ValueError(f"Invalid subject ids {invalid}; dataset has {max_subject_count} subjects")
        data = [[session[idx] for idx in subject_ids] for session in data]
        label = [[session[idx] for idx in subject_ids] for session in label]

    if max_subjects is not None:
        data = [session[:max_subjects] for session in data]
        label = [session[:max_subjects] for session in label]

    if max_trials is not None:
        data = [[subject[:max_trials] for subject in session] for session in data]
        label = [[subject[:max_trials] for subject in session] for session in label]

    return data, label


def _limit_samples(data, label, max_samples=None, strategy="stratified", seed=42):
    if max_samples is None:
        return data, label
    rng = np.random.default_rng(seed)
    out_data, out_label = [], []
    for session_data, session_label in zip(data, label):
        out_session_data, out_session_label = [], []
        for subject_data, subject_label in zip(session_data, session_label):
            y = np.asarray(subject_label).reshape(-1)
            keep = min(max_samples, len(subject_data))
            if strategy == "head":
                idx = np.arange(keep)
            elif strategy == "stratified":
                parts = []
                classes = np.unique(y)
                base = keep // len(classes)
                remainder = keep % len(classes)
                for class_pos, cls in enumerate(classes):
                    cls_idx = np.flatnonzero(y == cls)
                    n_take = min(len(cls_idx), base + (1 if class_pos < remainder else 0))
                    if n_take > 0:
                        parts.append(rng.choice(cls_idx, size=n_take, replace=False))
                idx = np.concatenate(parts) if parts else np.arange(keep)
                if len(idx) < keep:
                    missing = keep - len(idx)
                    rest = np.setdiff1d(np.arange(len(y)), idx, assume_unique=False)
                    if len(rest) > 0:
                        idx = np.concatenate([idx, rng.choice(rest, size=min(missing, len(rest)), replace=False)])
                idx = np.sort(idx)
            else:
                raise ValueError(f"Unsupported sample subset strategy: {strategy}")
            out_session_data.append(np.asarray(subject_data)[idx])
            out_session_label.append(y[idx])
        out_data.append(out_session_data)
        out_label.append(out_session_label)
    return out_data, out_label


def _validate_deap_dimensions(data, args, channels, num_freq_bands, text_dim, text_prototypes):
    sample = np.asarray(data[0][0][0])
    if sample.ndim != 3:
        raise ValueError(f"Expected each DEAP sample to be [L,C,F], got shape {sample.shape}")

    steps, sample_channels, sample_bands = sample.shape
    if steps != args.sample_length:
        raise ValueError(
            f"DEAP sample_length mismatch: loader returned L={steps}, "
            f"but args.sample_length={args.sample_length}. Use --sample_length {steps} "
            f"or regenerate the segmented data."
        )
    if sample_channels != channels:
        raise ValueError(f"Channel mismatch: sample has C={sample_channels}, metadata channels={channels}")
    if sample_bands != num_freq_bands:
        raise ValueError(f"Band mismatch: sample has F={sample_bands}, metadata num_freq_bands={num_freq_bands}")
    if text_prototypes.shape[-1] != text_dim:
        raise ValueError(
            f"Text prototype mismatch: tensor dim={text_prototypes.shape[-1]}, text_dim={text_dim}"
        )

    print(
        f"Dimension check: X=[N,{steps},{sample_channels},{sample_bands}], "
        f"R=[N,{sample_channels},{sample_channels}], text_dim={text_dim}"
    )


def _compute_class_weights(labels, num_classes, device):
    y = np.asarray(labels).reshape(-1)
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def _subject_stat_vector(samples):
    x = np.asarray(samples, dtype=np.float32)
    flat = x.reshape(-1, x.shape[-2], x.shape[-1])
    mean = flat.mean(axis=0).reshape(-1)
    std = flat.std(axis=0).reshape(-1)
    return np.concatenate([mean, std], axis=0)


def select_top_m_sources(session_data, all_source_ids, target_id, top_m):
    if top_m is None or top_m <= 0 or top_m >= len(all_source_ids):
        return all_source_ids

    target_stat = _subject_stat_vector(session_data[target_id])
    distances = []
    for sid in all_source_ids:
        src_stat = _subject_stat_vector(session_data[sid])
        dist = float(np.mean((src_stat - target_stat) ** 2))
        distances.append((dist, sid))
    distances.sort(key=lambda item: item[0])
    selected = [sid for _, sid in distances[:top_m]]
    pretty = ", ".join([f"S{sid + 1}:{dist:.4g}" for dist, sid in distances[:top_m]])
    print(f"Top-{top_m} source selection: {pretty}")
    return selected


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


def _compute_lambda_value(step, total_steps, args):
    if args.mmd_schedule == "monotonic":
        return lambda_warmup(step, total_steps, args.lambda_max)

    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    warmup_ratio = max(args.mmd_warmup_ratio, 1e-6)
    hold_ratio = max(args.mmd_hold_ratio, warmup_ratio)

    if progress <= warmup_ratio:
        return float(args.lambda_max) * progress / warmup_ratio

    if args.mmd_schedule == "warmup_hold" or progress <= hold_ratio:
        return float(args.lambda_max)

    if args.mmd_schedule == "warmup_decay":
        decay_progress = (progress - hold_ratio) / max(1.0 - hold_ratio, 1e-6)
        return float(args.lambda_max + (args.lambda_min - args.lambda_max) * decay_progress)

    if args.mmd_schedule == "warmup_cosine_decay":
        decay_progress = (progress - hold_ratio) / max(1.0 - hold_ratio, 1e-6)
        cosine = 0.5 * (1.0 + np.cos(np.pi * decay_progress))
        return float(args.lambda_min + (args.lambda_max - args.lambda_min) * cosine)

    raise ValueError(f"Unsupported mmd_schedule: {args.mmd_schedule}")


def _compute_sca_mu(step, total_steps, args):
    progress = min(max(step / max(total_steps, 1), 0.0), 1.0)
    ramp = min(progress / max(args.sca_mu_warmup_ratio, 1e-6), 1.0)
    return float(args.sca_mu_start + (args.sca_mu_end - args.sca_mu_start) * ramp)


def _compute_mmd_loss(
    z_src_all,
    z_tgt_all,
    y_src_list,
    text_prototypes,
    num_classes,
    args,
    source_weights=None,
    step=None,
    total_steps=None,
    r_src_list=None,
    r_tgt=None,
):
    if args.mmd_type == "marginal":
        return multisource_mmd(z_src_all, z_tgt_all, source_weights=source_weights)
    if args.mmd_type == "class_aware":
        return multisource_class_aware_mmd(
            z_src_all,
            z_tgt_all,
            y_src_list,
            text_prototypes,
            tau=args.proto_tau,
            num_classes=num_classes,
            source_weights=source_weights,
            confidence_gate=args.mmd_confidence_gate,
            confidence_threshold=args.mmd_confidence_threshold,
        )
    if args.mmd_type == "sca":
        conditional_mu = _compute_sca_mu(step or 0, total_steps or 1, args)
        return multisource_semantic_conditional_alignment(
            z_src_all,
            z_tgt_all,
            y_src_list,
            text_prototypes,
            tau=args.proto_tau,
            num_classes=num_classes,
            source_weights=source_weights,
            confidence_gate=args.mmd_confidence_gate,
            confidence_threshold=args.mmd_confidence_threshold,
            conditional_mu=conditional_mu,
        )
    if args.mmd_type == "resgca":
        if r_src_list is None or r_tgt is None:
            raise ValueError("r_src_list and r_tgt are required when mmd_type='resgca'")
        conditional_mu = _compute_sca_mu(step or 0, total_steps or 1, args)
        return multisource_resgca(
            z_src_all,
            z_tgt_all,
            y_src_list,
            r_src_list,
            r_tgt,
            text_prototypes,
            tau=args.proto_tau,
            num_classes=num_classes,
            source_weights=source_weights,
            confidence_gate=args.mmd_confidence_gate,
            confidence_threshold=args.mmd_confidence_threshold,
            conditional_mu=conditional_mu,
            geo_tau=args.resgca_geo_tau,
            geo_weight=args.resgca_geo_weight,
        )
    raise ValueError(f"Unsupported mmd_type: {args.mmd_type}")


def _train_selection_warmup(
    model,
    source_loaders,
    target_loader,
    text_prototypes,
    class_weights,
    num_classes,
    args,
):
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
            lambda_val = _compute_lambda_value(step, total_steps, args)
            loss = loss_proto + lambda_val * loss_mmd
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


def sparse_reliability_source_selection(
    model,
    source_loaders,
    target_loader,
    source_ids,
    text_prototypes,
    num_classes,
    args,
):
    source_z, source_y = _collect_source_embeddings(model, source_loaders, args.device)
    target_z = _collect_target_embeddings(model, target_loader, args.device)
    text_prototypes = torch.nn.functional.normalize(text_prototypes.to(args.device), dim=-1)

    scores = []
    details = []
    for source_idx, sid in enumerate(source_ids):
        z_s = source_z[source_idx]
        y_s = source_y[source_idx]
        z_t = target_z[source_idx]

        d_marg = torch.sum((z_s.mean(dim=0) - z_t.mean(dim=0)) ** 2)

        logits_t = z_t @ text_prototypes.T / args.proto_tau
        q_t = torch.softmax(logits_t, dim=-1)
        target_centers = []
        source_centers = []
        cond_weights = []
        for cls in range(num_classes):
            cls_weight = q_t[:, cls].sum().clamp_min(1e-6)
            target_centers.append((q_t[:, cls:cls + 1] * z_t).sum(dim=0) / cls_weight)
            mask = y_s == cls
            if mask.any():
                source_centers.append(z_s[mask].mean(dim=0))
            else:
                source_centers.append(z_s.mean(dim=0))
            cond_weights.append(cls_weight / q_t.size(0))
        target_centers = torch.stack(target_centers, dim=0)
        source_centers = torch.stack(source_centers, dim=0)
        cond_weights = torch.stack(cond_weights)
        d_cond = (cond_weights * ((source_centers - target_centers) ** 2).sum(dim=-1)).sum()

        logits_s = z_s @ text_prototypes.T / args.proto_tau
        source_acc_proxy = (logits_s.argmax(dim=-1) == y_s).float().mean()

        score = (
            -args.rel_marg_weight * d_marg
            -args.rel_cond_weight * d_cond
            +args.rel_val_weight * source_acc_proxy
        )
        scores.append(score)
        details.append({
            "sid": sid,
            "d_marg": float(d_marg.cpu()),
            "d_cond": float(d_cond.cpu()),
            "src_acc_proxy": float(source_acc_proxy.cpu()),
            "score": float(score.cpu()),
        })

    scores = torch.stack(scores)
    weights = torch.softmax(scores / args.source_weight_tau, dim=0)
    order = torch.argsort(weights, descending=True)
    selected_positions = []
    cumulative = 0.0
    for pos in order.tolist():
        selected_positions.append(pos)
        cumulative += float(weights[pos].cpu())
        if cumulative >= args.sparse_rho or len(selected_positions) >= args.sparse_k_max:
            break

    selected_positions = selected_positions[:args.sparse_k_max]
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


def load_deap(args, device):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setting.dataset_path = path_mapper["deap"]
    setting.dataset = "deap"
    setting.experiment_mode = "subject-independent"
    setting.onehot = False
    setting.label_used = ["valence"]
    setting.bounds = [5, 5.0001]
    setting.sessions = [1]
    setting.sample_length = args.sample_length
    setting.stride = args.stride
    setting.only_seg = False

    data, label, channels, num_freq_bands, num_classes = get_data(setting)
    data, label = _limit_subjects_and_trials(
        data,
        label,
        max_subjects=_optional_int(args, "max_subjects"),
        max_trials=_optional_int(args, "max_trials"),
        subject_ids=args.subject_ids,
        random_subject_count=args.random_subject_count,
        subject_seed=args.subject_seed,
    )
    data, label = flatten_trials(data, label)
    if args.subject_zscore:
        data = zscore_subject_wise(data)
        print("Applied subject-wise z-score normalization.")
    data, label = _limit_samples(
        data,
        label,
        max_samples=_optional_int(args, "max_samples_per_subject"),
        strategy=args.sample_subset,
        seed=args.seed,
    )

    text_dim, all_class_vectors = label_to_vector(
        dataset=setting.dataset,
        LM=args.LM,
        LabelTextMapper=None,
        device=device,
    )
    return data, label, all_class_vectors, channels, num_freq_bands, num_classes, text_dim


def build_geometry_for_fold(session_data, source_ids, target_id, device, shrinkage, eps, batch_size):
    """Compute source-only log(G), then R for sources and target."""
    source_tensors = [
        torch.tensor(np.asarray(session_data[sid]), dtype=torch.float32, device=device)
        for sid in source_ids
    ]
    with torch.no_grad():
        log_ref = log_euclidean_reference(source_tensors, shrinkage=shrinkage, eps=eps)

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
    return r_by_subject, log_ref


def _source_weight_tensor(source_weights, n_sources, device):
    if source_weights is None:
        return torch.full((n_sources,), 1.0 / max(n_sources, 1), device=device)
    weights = torch.as_tensor(source_weights, dtype=torch.float32, device=device)
    return weights / weights.sum().clamp_min(1e-8)


def _rsg_partner_indices(labels):
    partners = torch.full_like(labels, fill_value=-1)
    for cls in labels.unique().tolist():
        cls_idx = torch.nonzero(labels == cls, as_tuple=False).flatten()
        if cls_idx.numel() < 2:
            continue
        perm = cls_idx[torch.randperm(cls_idx.numel(), device=labels.device)]
        rolled = torch.roll(perm, shifts=1)
        same = rolled == cls_idx
        if same.any():
            rolled[same] = torch.roll(rolled, shifts=1)[same]
        partners[cls_idx] = rolled
    return partners


def _rsg_structured_cutmix(x, r, labels, args):
    """Reliability-aware semantic-geometric CutMix v1 within a selected source batch."""
    if torch.rand((), device=x.device).item() > args.rsg_prob:
        return None, None

    partners = _rsg_partner_indices(labels)
    valid = partners >= 0
    if valid.sum().item() < 2:
        return None, None

    x_base = x[valid].clone()
    x_pair = x[partners[valid]]
    r_valid = r[valid]
    y_mix = labels[valid]

    bsz, steps, channels, bands = x_base.shape
    time_min = max(1, min(args.rsg_time_min, steps))
    time_max = max(time_min, min(args.rsg_time_max, steps))
    band_width = max(1, min(args.rsg_band_width, bands))
    channel_count = max(1, min(channels, int(round(channels * args.rsg_channel_ratio))))

    for idx in range(bsz):
        t_len = int(torch.randint(time_min, time_max + 1, (1,), device=x.device).item())
        t0 = int(torch.randint(0, steps - t_len + 1, (1,), device=x.device).item())
        b0 = int(torch.randint(0, bands - band_width + 1, (1,), device=x.device).item())
        center = int(torch.randint(0, channels, (1,), device=x.device).item())

        geo_scores = r_valid[idx].abs()[center]
        channel_idx = torch.topk(geo_scores, k=channel_count, largest=True).indices
        x_base[idx, t0:t0 + t_len, channel_idx, b0:b0 + band_width] = (
            x_pair[idx, t0:t0 + t_len, channel_idx, b0:b0 + band_width]
        )

    return x_base, y_mix


def rsg_cutmix_proto_loss(
    model,
    x_src_list,
    r_src_list,
    y_src_list,
    text_prototypes,
    log_reference,
    args,
    class_weights_list=None,
    source_weights=None,
):
    if class_weights_list is None:
        class_weights_list = [None] * len(x_src_list)
    text_prototypes = F.normalize(text_prototypes.to(args.device), dim=-1)
    source_weight_vec = _source_weight_tensor(source_weights, len(x_src_list), args.device)
    losses = []
    weights = []

    for src_idx, (x_src, r_src, y_src, class_weights) in enumerate(
        zip(x_src_list, r_src_list, y_src_list, class_weights_list)
    ):
        x_mix, y_mix = _rsg_structured_cutmix(x_src, r_src, y_src, args)
        if x_mix is None:
            continue
        r_mix = tangent_deviation(
            x_mix,
            log_reference,
            shrinkage=args.shrinkage,
            eps=args.spd_eps,
        )
        h_mix, _ = model.encode(x_mix, r_mix)
        z_mix = model.prototype_head(model.adapters[src_idx](h_mix))
        logits = z_mix @ text_prototypes.T / args.proto_tau
        losses.append(F.cross_entropy(logits, y_mix, weight=class_weights))
        weights.append(source_weight_vec[src_idx])

    if not losses:
        return torch.zeros((), device=args.device)

    losses = torch.stack(losses)
    weights = torch.stack(weights)
    weights = weights / weights.sum().clamp_min(1e-8)
    return (weights * losses).sum()


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
    y_true = np.concatenate(true_parts)
    y_pred = np.concatenate(pred_parts)
    return y_true, y_pred


def run(args):
    setup_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.dataset = "deap"
    args.LM = "clip"

    args.sample_length = 9 if args.sample_length is None else args.sample_length
    args.stride = 3 if args.stride is None else args.stride
    args.epochs = 200 if args.epochs is None else args.epochs
    args.batch_size = 64 if args.batch_size is None else args.batch_size
    args.lr = 1e-3 if args.lr is None else args.lr
    if args.st_dim % args.heads != 0:
        raise ValueError(f"st_dim={args.st_dim} must be divisible by heads={args.heads}")
    if args.source_selection == "none" and args.top_m_sources is not None:
        args.source_selection = "fixed_top_m"
    if args.source_selection == "sparse_reliability" and args.sparse_k_max <= 0:
        raise ValueError("sparse_k_max must be positive when using sparse_reliability")
    if args.mmd_warmup_ratio <= 0.0 or args.mmd_warmup_ratio > 1.0:
        raise ValueError(f"mmd_warmup_ratio must be in (0, 1], got {args.mmd_warmup_ratio}")
    if args.mmd_hold_ratio < args.mmd_warmup_ratio or args.mmd_hold_ratio > 1.0:
        raise ValueError(
            f"mmd_hold_ratio must be in [{args.mmd_warmup_ratio}, 1], got {args.mmd_hold_ratio}"
        )
    if args.lambda_min < 0.0:
        raise ValueError(f"lambda_min must be >= 0, got {args.lambda_min}")
    if args.mmd_confidence_gate == "threshold" and not (0.0 <= args.mmd_confidence_threshold <= 1.0):
        raise ValueError(
            f"mmd_confidence_threshold must be in [0, 1], got {args.mmd_confidence_threshold}"
        )
    if not (0.0 <= args.sca_mu_start <= 1.0 and 0.0 <= args.sca_mu_end <= 1.0):
        raise ValueError("sca_mu_start and sca_mu_end must be in [0, 1]")
    if args.sca_mu_warmup_ratio <= 0.0 or args.sca_mu_warmup_ratio > 1.0:
        raise ValueError(f"sca_mu_warmup_ratio must be in (0, 1], got {args.sca_mu_warmup_ratio}")
    if args.resgca_geo_tau <= 0.0:
        raise ValueError(f"resgca_geo_tau must be positive, got {args.resgca_geo_tau}")
    if args.resgca_geo_weight < 0.0:
        raise ValueError(f"resgca_geo_weight must be >= 0, got {args.resgca_geo_weight}")
    if args.use_rsg_cutmix:
        if not (0.0 <= args.rsg_prob <= 1.0):
            raise ValueError(f"rsg_prob must be in [0, 1], got {args.rsg_prob}")
        if args.rsg_lambda_aug < 0.0:
            raise ValueError(f"rsg_lambda_aug must be >= 0, got {args.rsg_lambda_aug}")
        if args.rsg_time_min <= 0 or args.rsg_time_max <= 0:
            raise ValueError("rsg_time_min and rsg_time_max must be positive")
        if args.rsg_time_min > args.rsg_time_max:
            raise ValueError("rsg_time_min must be <= rsg_time_max")
        if args.rsg_band_width <= 0:
            raise ValueError("rsg_band_width must be positive")
        if not (0.0 < args.rsg_channel_ratio <= 1.0):
            raise ValueError(f"rsg_channel_ratio must be in (0, 1], got {args.rsg_channel_ratio}")
    has_target_subset = args.target_subject_ids is not None or args.random_target_count is not None
    has_dataset_subject_subset = (
        args.subject_ids is not None
        or args.random_subject_count is not None
        or args.max_subjects is not None
    )
    if has_target_subset and has_dataset_subject_subset:
        raise ValueError(
            "Target-only experiments must keep the full source pool. "
            "Do not combine --target_subject_ids/--random_target_count with "
            "--subject_ids, --random_subject_count, or --max_subjects."
        )

    output_dir = os.path.join(project_root, "results", "results_deap_geosem_stda")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"ep{args.epochs}_bs{args.batch_size}_lr{_format_float(args.lr)}_"
        f"lmda{_format_float(args.lambda_max)}_tau{_format_float(args.proto_tau)}_"
        f"topk{args.topk}_seed{args.seed}_{timestamp}"
    )
    if args.mmd_type != "marginal" or args.mmd_schedule != "monotonic":
        run_id = f"{run_id}_{args.mmd_type}_{args.mmd_schedule}"
    if args.use_rsg_cutmix:
        run_id = (
            f"{run_id}_rsgp{_format_float(args.rsg_prob)}_"
            f"augl{_format_float(args.rsg_lambda_aug)}"
        )
    run_dir = os.path.join(output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    epoch_log_path = os.path.join(run_dir, "epoch_log.csv")
    summary_path = os.path.join(output_dir, "summary_geosem_stda_runs.csv")
    config_path = os.path.join(run_dir, "run_config.json")

    print(f"Device: {args.device}")
    print(f"Run directory: {run_dir}")
    print(
        f"GeoSem-STDA config: L={args.sample_length}, topk={args.topk}, "
        f"lambda_max={args.lambda_max}, proto_tau={args.proto_tau}"
    )

    data, label, class_vectors, channels, num_freq_bands, num_classes, text_dim = load_deap(args, args.device)
    if args.topk >= channels:
        raise ValueError(f"topk={args.topk} must be smaller than channels={channels}")
    sorted_ids = sorted(class_vectors.keys())
    text_prototypes = torch.tensor(
        np.asarray([class_vectors[i] for i in sorted_ids], dtype=np.float32),
        device=args.device,
    )
    _validate_deap_dimensions(
        data,
        args,
        channels,
        num_freq_bands,
        text_dim,
        text_prototypes,
    )

    run_config = {
        "run_id": run_id,
        "script": os.path.abspath(__file__),
        "dataset": "deap",
        "label_used": ["valence"],
        "lm": args.LM,
        "sample_length": args.sample_length,
        "stride": args.stride,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "lambda_max": args.lambda_max,
        "lambda_min": args.lambda_min,
        "mmd_type": args.mmd_type,
        "mmd_schedule": args.mmd_schedule,
        "mmd_warmup_ratio": args.mmd_warmup_ratio,
        "mmd_hold_ratio": args.mmd_hold_ratio,
        "mmd_confidence_gate": args.mmd_confidence_gate,
        "mmd_confidence_threshold": args.mmd_confidence_threshold,
        "sca_mu_start": args.sca_mu_start,
        "sca_mu_end": args.sca_mu_end,
        "sca_mu_warmup_ratio": args.sca_mu_warmup_ratio,
        "resgca_geo_tau": args.resgca_geo_tau,
        "resgca_geo_weight": args.resgca_geo_weight,
        "proto_tau": args.proto_tau,
        "fusion_tau": args.fusion_tau,
        "topk": args.topk,
        "shrinkage": args.shrinkage,
        "st_dim": args.st_dim,
        "graph_dim": args.graph_dim,
        "adapter_bottleneck": args.adapter_bottleneck,
        "heads": args.heads,
        "dropout": args.dropout,
        "channels": channels,
        "num_freq_bands": num_freq_bands,
        "num_classes": num_classes,
        "text_dim": text_dim,
        "max_subjects": _optional_int(args, "max_subjects"),
        "subject_ids": args.subject_ids,
        "random_subject_count": args.random_subject_count,
        "subject_seed": args.subject_seed,
        "target_subject_ids": args.target_subject_ids,
        "random_target_count": args.random_target_count,
        "target_seed": args.target_seed,
        "max_trials": _optional_int(args, "max_trials"),
        "max_samples_per_subject": _optional_int(args, "max_samples_per_subject"),
        "sample_subset": args.sample_subset,
        "subject_zscore": args.subject_zscore,
        "top_m_sources": args.top_m_sources,
        "source_selection": args.source_selection,
        "reliability_warmup_epochs": args.reliability_warmup_epochs,
        "sparse_rho": args.sparse_rho,
        "sparse_k_max": args.sparse_k_max,
        "source_weight_tau": args.source_weight_tau,
        "rel_marg_weight": args.rel_marg_weight,
        "rel_cond_weight": args.rel_cond_weight,
        "rel_val_weight": args.rel_val_weight,
        "class_weighted_loss": args.class_weighted_loss,
        "use_rsg_cutmix": args.use_rsg_cutmix,
        "rsg_prob": args.rsg_prob,
        "rsg_lambda_aug": args.rsg_lambda_aug,
        "rsg_time_min": args.rsg_time_min,
        "rsg_time_max": args.rsg_time_max,
        "rsg_band_width": args.rsg_band_width,
        "rsg_channel_ratio": args.rsg_channel_ratio,
        "eval_interval": args.eval_interval,
        "created_at": timestamp,
    }
    _write_json(config_path, run_config)

    n_sessions = len(data)
    n_subjects = len(data[0])
    target_indices = _select_target_indices(
        n_subjects,
        target_subject_ids=args.target_subject_ids,
        random_target_count=args.random_target_count,
        target_seed=args.target_seed,
    )
    print(f"Targets evaluated: {[idx + 1 for idx in target_indices]}")
    run_config["evaluated_target_subjects"] = [idx + 1 for idx in target_indices]
    run_config["source_candidate_count_per_target"] = n_subjects - 1
    _write_json(config_path, run_config)
    results_acc, results_macro, results_micro = {}, {}, {}
    target_records = []
    epoch_fields = [
        "run_id", "session_idx", "target_subject", "epoch",
        "loss", "proto", "mmd", "aug", "lambda", "sca_mu", "acc", "macro_f1", "micro_f1",
        "best_acc", "alpha_mean", "epochs", "batch_size", "lr", "seed",
    ]

    for session_idx in range(n_sessions):
        session_acc, session_macro, session_micro = [], [], []

        for target_sub in target_indices:
            setup_seed(args.seed)
            all_source_ids = [sid for sid in range(n_subjects) if sid != target_sub]
            if args.source_selection == "fixed_top_m":
                source_ids = select_top_m_sources(
                    data[session_idx],
                    all_source_ids,
                    target_sub,
                    args.top_m_sources,
                )
            else:
                source_ids = all_source_ids

            print(f"\n{'=' * 60}")
            print(
                f"Session {session_idx} | Target subject {target_sub + 1}/{n_subjects} "
                f"| Sources used: {len(source_ids)}/{len(all_source_ids)}"
            )
            print(f"{'=' * 60}")

            r_by_subject, log_reference = build_geometry_for_fold(
                data[session_idx],
                source_ids,
                target_sub,
                args.device,
                shrinkage=args.shrinkage,
                eps=args.spd_eps,
                batch_size=args.geometry_batch_size,
            )

            source_loaders = []
            source_class_weights = []
            for sid in source_ids:
                x = torch.tensor(np.asarray(data[session_idx][sid]), dtype=torch.float32)
                r = r_by_subject[sid].float()
                y = torch.tensor(np.asarray(label[session_idx][sid]).reshape(-1), dtype=torch.long)
                class_weights = (
                    _compute_class_weights(label[session_idx][sid], num_classes, args.device)
                    if args.class_weighted_loss else None
                )
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
            total_steps = args.epochs * steps_per_epoch
            source_iters = [iter(loader) for loader in source_loaders]
            target_iter = iter(target_train_loader)
            global_step = 0
            best_acc = 0.0
            best_macro = 0.0
            best_micro = 0.0

            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0.0
                epoch_proto = 0.0
                epoch_mmd = 0.0
                epoch_aug = 0.0
                sca_mu_val = np.nan
                alpha_values = []

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
                    loss_mmd = _compute_mmd_loss(
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
                    loss = loss_proto + lambda_val * loss_mmd
                    loss_aug = torch.zeros((), device=args.device)
                    if args.use_rsg_cutmix:
                        loss_aug = rsg_cutmix_proto_loss(
                            model,
                            x_src_list,
                            r_src_list,
                            y_src_list,
                            text_prototypes,
                            log_reference,
                            args,
                            class_weights_list=source_class_weights,
                            source_weights=source_loss_weights,
                        )
                        loss = loss + args.rsg_lambda_aug * loss_aug
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_proto += loss_proto.item()
                    epoch_mmd += loss_mmd.item()
                    epoch_aug += loss_aug.item()
                    if alpha_src:
                        alpha_values.append(torch.cat([a.detach().cpu() for a in alpha_src]).mean().item())
                    alpha_values.append(alpha_tgt.detach().cpu().mean().item())

                mean_alpha = float(np.mean(alpha_values)) if alpha_values else 0.0
                should_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch + 1 == args.epochs)
                acc = np.nan
                macro_f1 = np.nan
                micro_f1 = np.nan

                if should_eval:
                    model.eval()
                    source_centroids = compute_source_class_centroids(
                        model,
                        source_loaders,
                        args.device,
                        num_classes=num_classes,
                    )
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
                        best_acc = acc
                        best_macro = macro_f1
                        best_micro = micro_f1
                        print(f"  >> New best acc: {best_acc:.4f} (epoch {epoch + 1})")

                if (epoch + 1) % args.log_interval == 0:
                    acc_text = f"{acc:.4f}" if should_eval else "skip"
                    sca_mu_text = f" mu={sca_mu_val:.3f}" if args.mmd_type in ("sca", "resgca") else ""
                    print(
                        f"Ep {epoch + 1:3d} | loss={epoch_loss / steps_per_epoch:.4f} "
                        f"proto={epoch_proto / steps_per_epoch:.4f} "
                        f"mmd={epoch_mmd / steps_per_epoch:.6f} "
                        f"aug={epoch_aug / steps_per_epoch:.4f} "
                        f"lambda={lambda_val:.4f}{sca_mu_text} acc={acc_text} "
                        f"best={best_acc:.4f} alpha={mean_alpha:.3f}"
                    )

                _append_csv_row(
                    epoch_log_path,
                    epoch_fields,
                    {
                        "run_id": run_id,
                        "session_idx": session_idx,
                        "target_subject": target_sub + 1,
                        "epoch": epoch + 1,
                        "loss": epoch_loss / steps_per_epoch,
                        "proto": epoch_proto / steps_per_epoch,
                        "mmd": epoch_mmd / steps_per_epoch,
                        "aug": epoch_aug / steps_per_epoch,
                        "lambda": lambda_val,
                        "sca_mu": sca_mu_val,
                        "acc": acc,
                        "macro_f1": macro_f1,
                        "micro_f1": micro_f1,
                        "best_acc": best_acc,
                        "alpha_mean": mean_alpha,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "seed": args.seed,
                    },
                )

            print(f"Final target {target_sub + 1}: acc={best_acc * 100:.2f}%")
            session_acc.append(best_acc)
            session_macro.append(best_macro)
            session_micro.append(best_micro)
            target_records.append({
                "session_idx": session_idx,
                "result_row_index": len(session_acc) - 1,
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

    subject_csv = os.path.join(
        run_dir,
        f"subject_results_deap_geosem_stda_valence_ep{args.epochs}_bs{args.batch_size}.csv",
    )
    save_csubs_results_csv(results_acc, results_macro, results_micro, subject_csv)
    target_map_csv = os.path.join(run_dir, "target_subject_results.csv")
    if target_records:
        with open(target_map_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(target_records[0].keys()))
            writer.writeheader()
            writer.writerows(target_records)

    all_acc = np.concatenate([np.asarray(results_acc[k]) for k in results_acc])
    all_macro = np.concatenate([np.asarray(results_macro[k]) for k in results_macro])
    all_micro = np.concatenate([np.asarray(results_micro[k]) for k in results_micro])
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
        "mmd_warmup_ratio": args.mmd_warmup_ratio,
        "mmd_hold_ratio": args.mmd_hold_ratio,
        "mmd_confidence_gate": args.mmd_confidence_gate,
        "mmd_confidence_threshold": args.mmd_confidence_threshold,
        "sca_mu_start": args.sca_mu_start,
        "sca_mu_end": args.sca_mu_end,
        "sca_mu_warmup_ratio": args.sca_mu_warmup_ratio,
        "resgca_geo_tau": args.resgca_geo_tau,
        "resgca_geo_weight": args.resgca_geo_weight,
        "proto_tau": args.proto_tau,
        "fusion_tau": args.fusion_tau,
        "topk": args.topk,
        "acc_mean": float(all_acc.mean()),
        "acc_std": float(all_acc.std(ddof=1)) if all_acc.size > 1 else 0.0,
        "macro_f1_mean": float(all_macro.mean()),
        "macro_f1_std": float(all_macro.std(ddof=1)) if all_macro.size > 1 else 0.0,
        "micro_f1_mean": float(all_micro.mean()),
        "micro_f1_std": float(all_micro.std(ddof=1)) if all_micro.size > 1 else 0.0,
        "subject_results_csv": subject_csv,
        "target_subject_results_csv": target_map_csv,
        "epoch_log_csv": epoch_log_path,
        "run_config_json": config_path,
    }
    _append_csv_row(summary_path, list(summary.keys()), summary)

    print(f"\nOverall Acc: {all_acc.mean() * 100:.2f}% +/- {all_acc.std() * 100:.2f}%")
    print(f"Results saved to {subject_csv}")
    print(f"Summary appended to {summary_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    parser.set_defaults(epochs=200, batch_size=64, lr=1e-3, sample_length=9, stride=3)
    parser.add_argument("--lambda_max", type=float, default=0.3)
    parser.add_argument("--lambda_min", type=float, default=0.0)
    parser.add_argument("--mmd_type", type=str, default="marginal", choices=["marginal", "class_aware", "sca", "resgca"])
    parser.add_argument("--mmd_schedule", type=str, default="monotonic",
                        choices=["monotonic", "warmup_hold", "warmup_decay", "warmup_cosine_decay"])
    parser.add_argument("--mmd_warmup_ratio", type=float, default=0.2)
    parser.add_argument("--mmd_hold_ratio", type=float, default=0.5)
    parser.add_argument("--mmd_confidence_gate", type=str, default="none",
                        choices=["none", "soft", "threshold", "entropy"])
    parser.add_argument("--mmd_confidence_threshold", type=float, default=0.6)
    parser.add_argument("--sca_mu_start", type=float, default=0.0,
                        help="initial conditional-alignment ratio for --mmd_type sca")
    parser.add_argument("--sca_mu_end", type=float, default=1.0,
                        help="final conditional-alignment ratio for --mmd_type sca")
    parser.add_argument("--sca_mu_warmup_ratio", type=float, default=0.5,
                        help="fraction of training used to ramp SCA from marginal to conditional")
    parser.add_argument("--resgca_geo_tau", type=float, default=1.0,
                        help="temperature for geometry trust in --mmd_type resgca")
    parser.add_argument("--resgca_geo_weight", type=float, default=1.0,
                        help="strength of geometry trust in --mmd_type resgca; 0 disables geometry gating")
    parser.add_argument("--proto_tau", type=float, default=0.07)
    parser.add_argument("--fusion_tau", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--spd_eps", type=float, default=1e-5)
    parser.add_argument("--geometry_batch_size", type=int, default=256)
    parser.add_argument("--st_dim", type=int, default=128)
    parser.add_argument("--graph_dim", type=int, default=64)
    parser.add_argument("--adapter_bottleneck", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--top_m_sources", type=int, default=None,
                        help="use only the top-M unlabeled-statistically closest source subjects")
    parser.add_argument("--source_selection", type=str, default="none",
                        choices=["none", "fixed_top_m", "sparse_reliability"])
    parser.add_argument("--reliability_warmup_epochs", type=int, default=5)
    parser.add_argument("--sparse_rho", type=float, default=0.85)
    parser.add_argument("--sparse_k_max", type=int, default=6)
    parser.add_argument("--source_weight_tau", type=float, default=0.5)
    parser.add_argument("--rel_marg_weight", type=float, default=1.0)
    parser.add_argument("--rel_cond_weight", type=float, default=1.0)
    parser.add_argument("--rel_val_weight", type=float, default=0.2)
    parser.add_argument("--use_rsg_cutmix", action="store_true",
                        help="enable reliability-aware semantic-geometric structured EEG CutMix")
    parser.add_argument("--rsg_prob", type=float, default=0.3,
                        help="probability of applying RSG-CutMix to a selected source batch")
    parser.add_argument("--rsg_lambda_aug", type=float, default=0.1,
                        help="weight of the RSG-CutMix prototype loss")
    parser.add_argument("--rsg_time_min", type=int, default=2,
                        help="minimum contiguous time windows replaced by RSG-CutMix")
    parser.add_argument("--rsg_time_max", type=int, default=4,
                        help="maximum contiguous time windows replaced by RSG-CutMix")
    parser.add_argument("--rsg_band_width", type=int, default=1,
                        help="number of adjacent frequency bands replaced by RSG-CutMix")
    parser.add_argument("--rsg_channel_ratio", type=float, default=0.25,
                        help="ratio of geometry-neighbor channels replaced by RSG-CutMix")
    parser.add_argument("--class_weighted_loss", action="store_true", default=True,
                        help="use source-wise class weights in prototype CE")
    parser.add_argument("--no_class_weighted_loss", action="store_false", dest="class_weighted_loss")
    parser.add_argument("--subject_zscore", action="store_true", default=True,
                        help="apply subject-wise z-score normalization to DE features")
    parser.add_argument("--no_subject_zscore", action="store_false", dest="subject_zscore")
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--subject_ids", type=str, nargs="+", default=None,
                        help="1-based subject ids to include, e.g. --subject_ids 1 3 5 7 9 or --subject_ids 1,3,5,7,9")
    parser.add_argument("--random_subject_count", type=int, default=None,
                        help="randomly sample this many subjects using --subject_seed")
    parser.add_argument("--subject_seed", type=int, default=42)
    parser.add_argument("--target_subject_ids", type=str, nargs="+", default=None,
                        help="1-based target subject ids to evaluate while keeping the full source pool")
    parser.add_argument("--random_target_count", type=int, default=None,
                        help="randomly sample this many target subjects while keeping the full source pool")
    parser.add_argument("--target_seed", type=int, default=42)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--max_samples_per_subject", type=int, default=None)
    parser.add_argument("--sample_subset", type=str, default="stratified", choices=["stratified", "head"])
    run(parser.parse_args())
