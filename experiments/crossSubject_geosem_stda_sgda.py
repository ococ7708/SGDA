"""
Cross-dataset GeoSem-STDA under the SGDA cross-subject protocol.

Supported datasets:
  - seed
  - seediv
  - seedv
  - dreamer

Protocol kept from the original SGDA scripts:
  - leave-one-subject-out within each evaluated session;
  - each non-target subject is a source domain;
  - target labels are used only for epoch-wise evaluation;
  - the reported subject result is best target accuracy over epochs.
"""

import csv
import hashlib
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import confusion_matrix, f1_score
from torch.utils.data import DataLoader, TensorDataset

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from config.setting import preset_setting, set_setting_by_args
from data_utils.constants.path_mapper import path_mapper
from data_utils.load_data import get_data
from data_utils.load_dreamer import get_data as get_dreamer_data
from data_utils.load_seedv import read_seedv_feature, segment as segment_seedv
from data_utils.text_to_vector import label_to_vector
from experiments.deap.crossSubject_geosem_stda_deap import (
    _compute_lambda_value,
    _compute_mmd_loss,
    _compute_sca_mu,
    load_deap,
)
from experiments.seediv.crossSubject_geosem_stda_seediv import (
    _compute_class_weights,
    _make_model,
    _make_optimizer,
    _parse_session_indices,
    _select_target_indices,
    _train_selection_warmup,
    build_geometry_for_fold,
    evaluate,
    sparse_reliability_source_selection,
)
from models.geosem_stda import (
    GeoSemSTDA,
    compute_source_class_centroids,
    compute_source_domain_centroids,
    linear_classification_loss,
    log_euclidean_reference,
    prototype_contrastive_loss,
)
from utils.args import get_args_parser
from utils.mix_utils import (
    apply_euclidean_alignment,
    flatten_trials,
    global_normalization_after_ea,
    setup_seed,
    zscore_subject_seedv,
    zscore_subject_wise,
)


DATASET_DEFAULTS = {
    "deap": {
        "path_key": "deap",
        "loader_dataset": "deap",
        "text_dataset": "deap",
        "sessions": [1],
        "sample_length": 9,
        "stride": 3,
        "batch_size": 64,
        "lambda_max": 0.02,
        "lambda_min": 0.0,
        "topk": 6,
        "geometry_batch_size": 256,
    },
    "seed": {
        "path_key": "seed_de_lds",
        "loader_dataset": "seed_de_lds",
        "text_dataset": "seed_de_lds",
        "sessions": [1, 2, 3],
        "sample_length": 3,
        "stride": 1,
        "batch_size": 128,
        "lambda_max": 0.2,
        "lambda_min": 0.05,
        "topk": 8,
        "geometry_batch_size": 128,
    },
    "seediv": {
        "path_key": "seediv_de_lds",
        "loader_dataset": "seediv_de_lds",
        "text_dataset": "seediv_de_lds",
        "sessions": [1, 2, 3],
        "sample_length": 3,
        "stride": 1,
        "batch_size": 64,
        "lambda_max": 0.2,
        "lambda_min": 0.05,
        "topk": 8,
        "geometry_batch_size": 128,
    },
    "seedv": {
        "path_key": "seedv_de_lds",
        "loader_dataset": "seedv",
        "text_dataset": "seedv",
        "sessions": [1, 2, 3],
        "sample_length": 3,
        "stride": 1,
        "batch_size": 64,
        "lambda_max": 0.2,
        "lambda_min": 0.05,
        "topk": 8,
        "geometry_batch_size": 128,
    },
    "dreamer": {
        "path_key": "dreamer",
        "loader_dataset": "dreamer",
        "text_dataset": "dreamer",
        "sessions": [1],
        "sample_length": 3,
        "stride": 1,
        "batch_size": 64,
        "lambda_max": 0.05,
        "lambda_min": 0.0,
        "topk": 6,
        "geometry_batch_size": 128,
    },
}


EXPERIMENT_VARIANTS = {
    "standard": None,
    "proto_only": {
        "name": "P0_proto_only_top6",
        "mmd_type": "none",
        "source_selection": "sparse_reliability",
        "lambda_max": 0.0,
        "lambda_min": 0.0,
        "selection_warmup_proto_only": True,
    },
    "conditional": {
        "name": "P1_conditional_top6",
        "mmd_type": "conditional_simple",
        "source_selection": "sparse_reliability",
        "selection_warmup_proto_only": True,
    },
    "resgca_topk": {
        "name": "P2_resgca_top6",
        "mmd_type": "resgca",
        "source_selection": "sparse_reliability",
    },
    "resgca_all": {
        "name": "P3_resgca_all_sources",
        "mmd_type": "resgca",
        "source_selection": "none",
    },
}


RAPID_VARIANTS = {
    "r0_current_clip": {
        "name": "R0_current_geosem_clip",
        "representation_mode": "geosem",
        "classifier_type": "clip",
    },
    "r1_de_residual_clip": {
        "name": "R1_de_residual_clip",
        "representation_mode": "de_residual",
        "classifier_type": "clip",
    },
    "r2_de_gated_clip": {
        "name": "R2_de_gated_clip",
        "representation_mode": "de_gated",
        "classifier_type": "clip",
    },
    "r3_de_gated_linear": {
        "name": "R3_de_gated_linear",
        "representation_mode": "de_gated",
        "classifier_type": "linear",
    },
    "r4_de_only_clip": {
        "name": "R4_de_only_clip",
        "representation_mode": "de_only",
        "classifier_type": "clip",
    },
    "e0_r4_de_only": {
        "name": "E0_R4_de_only",
        "representation_mode": "de_only",
        "classifier_type": "clip",
        "cast_level1": True,
        "cast_variant": None,
    },
    "e1_strong_de": {
        "name": "E1_strong_de",
        "representation_mode": "cast_level1",
        "classifier_type": "clip",
        "cast_level1": True,
        "cast_variant": "e1_strong_de",
    },
    "e2_strong_de_channel": {
        "name": "E2_strong_de_channel",
        "representation_mode": "cast_level1",
        "classifier_type": "clip",
        "cast_level1": True,
        "cast_variant": "e2_strong_de_channel",
    },
    "e3_strong_de_channel_graph": {
        "name": "E3_strong_de_channel_graph",
        "representation_mode": "cast_level1",
        "classifier_type": "clip",
        "cast_level1": True,
        "cast_variant": "e3_strong_de_channel_graph",
    },
    "e4_strong_de_channel_graph_multiscale": {
        "name": "E4_strong_de_channel_graph_multiscale",
        "representation_mode": "cast_level1",
        "classifier_type": "clip",
        "cast_level1": True,
        "cast_variant": "e4_strong_de_channel_graph_multiscale",
    },
    "e5_strong_de_full_st": {
        "name": "E5_strong_de_full_st",
        "representation_mode": "cast_level1",
        "classifier_type": "clip",
        "cast_level1": True,
        "cast_variant": "e5_strong_de_full_st",
    },
}

RAPID_TARGETS = [12, 2, 4]
RAPID_DIFFICULTY = {12: "Easy", 2: "Medium", 4: "Hard"}


PILOT_LEVELS = ["Easy", "Medium-Easy", "Medium", "Medium-Hard", "Hard"]
PILOT_PERCENTILES = [0.10, 0.30, 0.50, 0.70, 0.90]


def _format_float(value):
    return str(value).replace(".", "p").replace("-", "m")


def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


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


def _log(message, log_path=None):
    print(message)
    if log_path is not None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(str(message) + "\n")


def _first_sample(data):
    return np.asarray(data[0][0][0])


def _stream_subject_log_center(subject_samples, device, shrinkage, eps, batch_size):
    x = torch.tensor(np.asarray(subject_samples), dtype=torch.float32, device=device)
    log_sum = None
    total = 0
    with torch.no_grad():
        for start in range(0, x.size(0), batch_size):
            batch = x[start:start + batch_size]
            center = log_euclidean_reference([batch], shrinkage=shrinkage, eps=eps)
            n_part = batch.size(0)
            log_sum = center * n_part if log_sum is None else log_sum + center * n_part
            total += n_part
    if total == 0:
        raise ValueError("Subject has no samples; cannot build pilot geometry descriptor")
    return (log_sum / total).detach().cpu()


def _nearest_unique_positions(n_items, percentiles):
    selected = []
    for pct in percentiles:
        raw = int(round(float(pct) * (n_items - 1)))
        candidates = sorted(range(n_items), key=lambda idx: (abs(idx - raw), idx))
        for idx in candidates:
            if idx not in selected:
                selected.append(idx)
                break
    return selected


def build_pilot_protocol(args, data, channels, num_freq_bands):
    if args.dataset_name != "dreamer":
        raise ValueError("The current pilot protocol is defined for DREAMER only")
    if len(data) != 1:
        raise ValueError(f"DREAMER pilot expects one session, got {len(data)}")

    session_data = data[0]
    n_subjects = len(session_data)
    centers = []
    for sid, subject_samples in enumerate(session_data):
        center = _stream_subject_log_center(
            subject_samples,
            args.device,
            shrinkage=args.shrinkage,
            eps=args.spd_eps,
            batch_size=args.geometry_batch_size,
        )
        centers.append(center)
        _log(f"Pilot geometry center built: S{sid + 1}/{n_subjects}")

    center_tensor = torch.stack(centers, dim=0)
    difficulty = []
    for sid in range(n_subjects):
        dists = []
        for other in range(n_subjects):
            if other == sid:
                continue
            dists.append(torch.norm(center_tensor[sid] - center_tensor[other], p="fro"))
        difficulty.append(float(torch.stack(dists).mean().item()))

    order = sorted(range(n_subjects), key=lambda idx: difficulty[idx])
    selected_positions = _nearest_unique_positions(n_subjects, PILOT_PERCENTILES)
    selected_subjects = [order[pos] for pos in selected_positions]
    difficulty_levels = {
        str(subject_id + 1): level
        for subject_id, level in zip(selected_subjects, PILOT_LEVELS)
    }

    payload = {
        "dataset": "dreamer",
        "task": args.dreamer_labeltype,
        "selection_method": "unlabeled_log_euclidean_subject_difficulty",
        "subject_index_base": 1,
        "n_subjects": n_subjects,
        "n_pilot": len(selected_subjects),
        "channels": channels,
        "num_freq_bands": num_freq_bands,
        "sample_length": args.sample_length,
        "stride": args.stride,
        "shrinkage": args.shrinkage,
        "spd_eps": args.spd_eps,
        "difficulty_scores": {
            str(sid + 1): difficulty[sid]
            for sid in range(n_subjects)
        },
        "difficulty_ranking": [
            {
                "rank": rank + 1,
                "subject_id": sid + 1,
                "difficulty": difficulty[sid],
            }
            for rank, sid in enumerate(order)
        ],
        "sorted_subject_ids": [sid + 1 for sid in order],
        "selected_percentiles": PILOT_PERCENTILES,
        "pilot_subject_ids": [sid + 1 for sid in selected_subjects],
        "difficulty_levels": difficulty_levels,
    }

    os.makedirs(os.path.dirname(args.pilot_config), exist_ok=True)
    _write_json(args.pilot_config, payload)

    _log("\n===== DREAMER PILOT-5 =====")
    _log("Rank  Subject  Difficulty  Level")
    selected_level_by_subject = {sid + 1: difficulty_levels[str(sid + 1)] for sid in selected_subjects}
    for rank, sid in enumerate(order, start=1):
        level = selected_level_by_subject.get(sid + 1, "")
        _log(f"{rank:>4}  S{sid + 1:<7} {difficulty[sid]:.6f}  {level}")
    _log("\nSelected Pilot Subjects:")
    for level, sid in zip(PILOT_LEVELS, selected_subjects):
        _log(f"{level:<12} -> S{sid + 1}")
    _log(f"\nPilot config saved to: {args.pilot_config}")
    return payload


def load_pilot_protocol(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("subject_index_base", 1) != 1:
        raise ValueError("Only 1-based pilot subject IDs are supported")
    if "pilot_subject_ids" not in payload:
        raise ValueError(f"Missing pilot_subject_ids in {path}")
    return payload


def _pilot_subject_metadata(pilot_config):
    if pilot_config is None:
        return {}, {}
    difficulty = {
        int(sid): float(score)
        for sid, score in pilot_config.get("difficulty_scores", {}).items()
    }
    levels = {
        int(sid): level
        for sid, level in pilot_config.get("difficulty_levels", {}).items()
    }
    return difficulty, levels


def _apply_experiment_variant(args):
    spec = EXPERIMENT_VARIANTS.get(args.experiment_variant)
    if spec is None:
        args.experiment_label = "standard"
        return args
    args.experiment_label = spec["name"]
    for key, value in spec.items():
        if key != "name":
            setattr(args, key, value)
    return args


def _apply_rapid_variant(args):
    if not args.rapid_pilot3:
        args.representation_mode = "geosem"
        args.classifier_type = "clip"
        args.cast_level1 = False
        args.cast_variant = None
        return args
    if args.dataset_name != "dreamer":
        raise ValueError("--rapid_pilot3 is defined for DREAMER only")
    spec = RAPID_VARIANTS[args.rapid_variant]
    args.experiment_label = spec["name"]
    args.representation_mode = spec["representation_mode"]
    args.classifier_type = spec["classifier_type"]
    args.cast_level1 = bool(spec.get("cast_level1", False))
    args.cast_variant = spec.get("cast_variant")
    args.mmd_type = "none"
    args.lambda_max = 0.0
    args.lambda_min = 0.0
    args.source_selection = "fixed"
    args.eval_classifier = "senior_feature"
    args.reliability_fusion = False
    if args.epochs is None:
        args.epochs = 2 if args.rapid_smoke_test else 100
    if args.target_subject_ids is None:
        args.target_subject_ids = ["12"] if args.rapid_smoke_test else ["12,2,4"]
    return args


def _load_fixed_sources(path, n_subjects):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if payload.get("dataset") != "dreamer":
        raise ValueError(f"Fixed source config must be for dreamer: {path}")
    targets = payload.get("targets", {})
    parsed = {}
    for target_id in RAPID_TARGETS:
        item = targets.get(str(target_id))
        if item is None:
            raise ValueError(f"Missing target S{target_id} in fixed source config: {path}")
        sources = [int(sid) for sid in item.get("sources", [])]
        if len(sources) != 6 or len(set(sources)) != 6:
            raise ValueError(f"Target S{target_id} must have exactly six unique fixed sources")
        if target_id in sources or any(sid < 1 or sid > n_subjects for sid in sources):
            raise ValueError(f"Invalid fixed source IDs for target S{target_id}: {sources}")
        parsed[target_id - 1] = [sid - 1 for sid in sources]
    return parsed


def _make_model_for_run(args, n_sources, channels, num_freq_bands, text_dim, num_classes):
    if not args.rapid_pilot3:
        return _make_model(args, n_sources, channels, num_freq_bands, text_dim)
    return GeoSemSTDA(
        n_sources=n_sources,
        num_electrodes=channels,
        num_freq_bands=num_freq_bands,
        st_dim=args.st_dim,
        graph_dim=args.graph_dim,
        adapter_bottleneck=args.adapter_bottleneck,
        text_dim=text_dim,
        heads=args.heads,
        graph_heads=args.graph_heads,
        topk=args.topk,
        dropout=args.dropout,
        sample_length=args.sample_length,
        representation_mode=args.representation_mode,
        classifier_type=args.classifier_type,
        num_classes=num_classes,
        beta_initial=0.1,
        cast_variant=args.cast_variant,
    ).to(args.device)


def _make_optimizer_for_run(args, model):
    if not args.rapid_pilot3:
        return _make_optimizer(args, model)
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        raise ValueError("No trainable parameters remain after applying the freeze policy")
    return optim.Adam(trainable, lr=args.lr, weight_decay=args.weight_decay)


def _build_fold_representation_inputs(args, session_data, source_ids, target_sub):
    if args.representation_mode in ("de_only", "cast_level1"):
        placeholder = {
            sid: torch.zeros((len(session_data[sid]), 1, 1), dtype=torch.float32)
            for sid in source_ids + [target_sub]
        }
        return placeholder, True
    geometry = build_geometry_for_fold(
        session_data,
        source_ids,
        target_sub,
        args.device,
        shrinkage=args.shrinkage,
        eps=args.spd_eps,
        batch_size=args.geometry_batch_size,
    )
    return geometry, False


def _count_parameters(model):
    return (
        sum(param.numel() for param in model.parameters()),
        sum(param.numel() for param in model.parameters() if param.requires_grad),
    )


def _module_parameter_count(module):
    if module is None:
        return 0
    return sum(param.numel() for param in module.parameters())


def _parameter_breakdown(model):
    return {
        "direct_de_encoder_params": _module_parameter_count(model.direct_de_encoder),
        "structured_encoder_params": _module_parameter_count(model.encoder),
        "cast_level1_encoder_params": _module_parameter_count(model.cast_encoder),
        "adapters_params": _module_parameter_count(model.adapters),
        "head_prototype_related_params": (
            _module_parameter_count(model.prototype_head)
            + _module_parameter_count(model.linear_head)
        ),
        "fusion_params": _module_parameter_count(model.residual_fusion),
    }


def _balanced_source_indices(labels, samples_per_class, seed, subject_id):
    labels = np.asarray(labels).reshape(-1).astype(np.int64)
    classes = np.unique(labels)
    if classes.size < 2:
        raise ValueError(f"S{subject_id} has fewer than two classes; balanced screening is impossible")
    available = {int(cls): np.flatnonzero(labels == cls) for cls in classes}
    effective = min(int(samples_per_class), min(len(indices) for indices in available.values()))
    if effective <= 0:
        raise ValueError(f"S{subject_id} has an empty class")
    rng = np.random.default_rng(int(seed) * 1009 + int(subject_id))
    chosen = np.concatenate([
        rng.choice(available[int(cls)], size=effective, replace=False) for cls in sorted(classes)
    ])
    chosen.sort()
    digest = hashlib.sha256(chosen.astype(np.int64).tobytes()).hexdigest()
    return chosen, {
        "subject_id": int(subject_id),
        "requested_samples_per_class": int(samples_per_class),
        "effective_samples_per_class": int(effective),
        "classes": [int(cls) for cls in sorted(classes)],
        "selected_count": int(chosen.size),
        "index_sha256": digest,
        "indices": chosen.tolist(),
    }


def _resolve_cast_checkpoint_path(path_template, target_subject):
    if not path_template:
        return None
    resolved = path_template.format(target=target_subject)
    return resolved if os.path.isabs(resolved) else os.path.join(project_root, resolved)


def _load_and_optionally_freeze_cast(model, checkpoint_path, freeze_loaded):
    if checkpoint_path is None:
        if freeze_loaded:
            raise ValueError("--cast_freeze_loaded requires --cast_pretrained_checkpoint")
        return {"checkpoint": None, "loaded_parameter_tensors": 0, "frozen_parameter_tensors": 0}
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"CAST checkpoint not found: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = payload.get("model_state_dict", payload) if isinstance(payload, dict) else payload
    if not isinstance(state, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")
    current = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in current and torch.is_tensor(value) and tuple(value.shape) == tuple(current[key].shape)
    }
    if not compatible:
        raise ValueError(f"No name-and-shape compatible parameters in checkpoint: {checkpoint_path}")
    model.load_state_dict(compatible, strict=False)
    frozen = 0
    if freeze_loaded:
        compatible_names = set(compatible)
        for name, param in model.named_parameters():
            if name in compatible_names:
                param.requires_grad_(False)
                frozen += 1
    return {
        "checkpoint": os.path.abspath(checkpoint_path),
        "loaded_parameter_tensors": len(compatible),
        "frozen_parameter_tensors": frozen,
        "freeze_loaded": bool(freeze_loaded),
    }


def _classification_diagnostics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in cm.ravel()]
    recall0 = float(tn / max(tn + fp, 1))
    recall1 = float(tp / max(tp + fn, 1))
    return {
        "confusion_matrix": cm.tolist(),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "recall_class0": recall0,
        "recall_class1": recall1,
        "balanced_accuracy": 0.5 * (recall0 + recall1),
        "predicted_class0_ratio": float((y_pred == 0).mean()),
        "predicted_class1_ratio": float((y_pred == 1).mean()),
        "true_class0_ratio": float((y_true == 0).mean()),
        "true_class1_ratio": float((y_true == 1).mean()),
    }


def simple_conditional_alignment(
    z_src_all,
    z_tgt_all,
    y_src_list,
    text_prototypes,
    tau=0.07,
    num_classes=2,
    eps=1e-6,
):
    text_prototypes = F.normalize(text_prototypes, dim=-1)
    source_losses = []

    for z_src, z_tgt, y_src in zip(z_src_all, z_tgt_all, y_src_list):
        q_t = F.softmax(z_tgt @ text_prototypes.T / tau, dim=-1).detach()
        class_losses = []

        for cls in range(num_classes):
            src_mask = y_src == cls
            if not src_mask.any():
                continue

            tgt_weight = q_t[:, cls].sum().clamp_min(eps)
            src_center = z_src[src_mask].mean(dim=0)
            tgt_center = (q_t[:, cls:cls + 1] * z_tgt).sum(dim=0) / tgt_weight
            class_losses.append(((src_center - tgt_center) ** 2).sum())

        if class_losses:
            source_losses.append(torch.stack(class_losses).mean())

    if not source_losses:
        return torch.zeros((), device=text_prototypes.device)

    loss = torch.stack(source_losses).mean()
    return torch.nan_to_num(loss, nan=0.0, posinf=1e4, neginf=0.0)


def compute_alignment_loss(
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
    if args.mmd_type == "none":
        return torch.zeros((), device=text_prototypes.device)
    if args.mmd_type == "conditional_simple":
        return simple_conditional_alignment(
            z_src_all,
            z_tgt_all,
            y_src_list,
            text_prototypes,
            tau=args.proto_tau,
            num_classes=num_classes,
        )
    return _compute_mmd_loss(
        z_src_all,
        z_tgt_all,
        y_src_list,
        text_prototypes,
        num_classes,
        args,
        source_weights=source_weights,
        step=step,
        total_steps=total_steps,
        r_src_list=r_src_list,
        r_tgt=r_tgt,
    )


def write_pilot_diagnostic_comparison(pilot_root, pilot_config):
    variant_columns = [
        ("P0_proto_only_top6", "P0 Proto"),
        ("P1_conditional_top6", "P1 Conditional"),
        ("P2_resgca_top6", "P2 ReSGCA Top6"),
        ("P3_resgca_all_sources", "P3 ReSGCA All"),
    ]
    pilot_subjects = [int(sid) for sid in pilot_config["pilot_subject_ids"]]
    levels = {
        int(sid): level
        for sid, level in pilot_config.get("difficulty_levels", {}).items()
    }
    results = {}

    for variant_dir, _ in variant_columns:
        csv_path = os.path.join(pilot_root, variant_dir, "per_subject_results.csv")
        variant_results = {}
        if os.path.exists(csv_path):
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    target = int(row["target_subject"])
                    variant_results[target] = float(row.get("best_acc") or row["acc"])
        results[variant_dir] = variant_results

    comparison_path = os.path.join(pilot_root, "pilot5_diagnostic_comparison.csv")
    fields = ["Target", "Difficulty"] + [col for _, col in variant_columns]
    rows = []
    for target in pilot_subjects:
        row = {"Target": f"S{target}", "Difficulty": levels.get(target, "")}
        for variant_dir, col in variant_columns:
            value = results[variant_dir].get(target)
            row[col] = "" if value is None else round(value * 100.0, 4)
        rows.append(row)

    mean_row = {"Target": "Mean", "Difficulty": "-"}
    std_row = {"Target": "Std", "Difficulty": "-"}
    for variant_dir, col in variant_columns:
        vals = np.asarray([results[variant_dir][target] for target in pilot_subjects if target in results[variant_dir]], dtype=np.float32)
        mean_row[col] = "" if vals.size == 0 else round(float(vals.mean() * 100.0), 4)
        std_row[col] = "" if vals.size < 2 else round(float(vals.std(ddof=1) * 100.0), 4)
    rows.extend([mean_row, std_row])

    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    deltas = {}
    p0 = results["P0_proto_only_top6"]
    for variant_dir, col in variant_columns[1:]:
        paired = [
            (p0[target], results[variant_dir][target])
            for target in pilot_subjects
            if target in p0 and target in results[variant_dir]
        ]
        if paired:
            p0_vals = np.asarray([item[0] for item in paired], dtype=np.float32)
            cur_vals = np.asarray([item[1] for item in paired], dtype=np.float32)
            deltas[col] = {
                "delta_acc_pp_vs_p0": float((cur_vals.mean() - p0_vals.mean()) * 100.0),
                "wins_vs_p0": int((cur_vals > p0_vals).sum()),
                "paired_subjects": int(cur_vals.size),
            }
    if "P1_conditional_top6" in results and "P2_resgca_top6" in results:
        p1 = results["P1_conditional_top6"]
        p2 = results["P2_resgca_top6"]
        paired = [(p1[target], p2[target]) for target in pilot_subjects if target in p1 and target in p2]
        if paired:
            p1_vals = np.asarray([item[0] for item in paired], dtype=np.float32)
            p2_vals = np.asarray([item[1] for item in paired], dtype=np.float32)
            deltas["P2 vs P1"] = {
                "delta_acc_pp": float((p2_vals.mean() - p1_vals.mean()) * 100.0),
                "wins": int((p2_vals > p1_vals).sum()),
                "paired_subjects": int(p2_vals.size),
            }
    if "P2_resgca_top6" in results and "P3_resgca_all_sources" in results:
        p2 = results["P2_resgca_top6"]
        p3 = results["P3_resgca_all_sources"]
        paired = [(p2[target], p3[target]) for target in pilot_subjects if target in p2 and target in p3]
        if paired:
            p2_vals = np.asarray([item[0] for item in paired], dtype=np.float32)
            p3_vals = np.asarray([item[1] for item in paired], dtype=np.float32)
            deltas["P2 vs P3"] = {
                "delta_acc_pp": float((p2_vals.mean() - p3_vals.mean()) * 100.0),
                "wins": int((p2_vals > p3_vals).sum()),
                "paired_subjects": int(p2_vals.size),
            }

    delta_path = os.path.join(pilot_root, "pilot5_diagnostic_deltas.json")
    _write_json(delta_path, deltas)
    return comparison_path, delta_path


def write_rapid_pilot3_summary(output_dir):
    variant_columns = [
        ("R0_current_geosem_clip", "R0"),
        ("R1_de_residual_clip", "R1"),
        ("R2_de_gated_clip", "R2"),
        ("R3_de_gated_linear", "R3"),
        ("R4_de_only_clip", "R4"),
    ]
    results = {}
    details = {}
    for variant_dir, short_name in variant_columns:
        path = os.path.join(output_dir, variant_dir, "per_subject_results.csv")
        per_target = {}
        detail_rows = []
        if os.path.exists(path):
            with open(path, "r", newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    per_target[int(row["target_subject"])] = float(row["best_acc"])
                    detail_rows.append(row)
        results[short_name] = per_target
        details[short_name] = detail_rows

    rows = []
    for target_id in RAPID_TARGETS:
        row = {"Target": f"S{target_id}", "Difficulty": RAPID_DIFFICULTY[target_id]}
        for _, short_name in variant_columns:
            value = results[short_name].get(target_id)
            row[short_name] = "" if value is None else round(value * 100.0, 4)
        rows.append(row)
    for statistic in ("Mean", "Std"):
        row = {"Target": statistic, "Difficulty": ""}
        for _, short_name in variant_columns:
            values = np.asarray(
                [results[short_name][sid] for sid in RAPID_TARGETS if sid in results[short_name]],
                dtype=np.float32,
            )
            if values.size == 0:
                row[short_name] = ""
            elif statistic == "Mean":
                row[short_name] = round(float(values.mean() * 100.0), 4)
            else:
                row[short_name] = round(float(values.std(ddof=1) * 100.0), 4) if values.size > 1 else ""
        rows.append(row)

    comparison_path = os.path.join(output_dir, "rapid_pilot3_comparison.csv")
    with open(comparison_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Target", "Difficulty", "R0", "R1", "R2", "R3", "R4"])
        writer.writeheader()
        writer.writerows(rows)

    comparisons = [("R1", "R0"), ("R2", "R1"), ("R2", "R0"), ("R3", "R2"), ("R4", "R0")]
    deltas = {}
    for current, baseline in comparisons:
        paired = [
            (results[baseline][sid], results[current][sid])
            for sid in RAPID_TARGETS
            if sid in results[baseline] and sid in results[current]
        ]
        if paired:
            base = np.asarray([pair[0] for pair in paired], dtype=np.float32)
            cur = np.asarray([pair[1] for pair in paired], dtype=np.float32)
            deltas[f"{current}-{baseline}"] = {
                "delta_mean_best_acc_pp": float((cur.mean() - base.mean()) * 100.0),
                "wins": int((cur > base).sum()),
                "paired_subjects": int(cur.size),
            }

    r0 = results["R0"]
    promising = {}
    if all(sid in r0 for sid in RAPID_TARGETS):
        r0_values = np.asarray([r0[sid] for sid in RAPID_TARGETS], dtype=np.float32)
        for variant in ("R1", "R2", "R3", "R4"):
            if all(sid in results[variant] for sid in RAPID_TARGETS):
                values = np.asarray([results[variant][sid] for sid in RAPID_TARGETS], dtype=np.float32)
                delta_pp = float((values.mean() - r0_values.mean()) * 100.0)
                wins = int((values > r0_values).sum())
                promising[variant] = "PROMISING" if delta_pp >= 2.0 and wins >= 2 else "NO CLEAR ADVANTAGE"

    aggregate_fields = [
        "variant", "S12_best_acc", "S2_best_acc", "S4_best_acc",
        "mean_best_acc", "std_best_acc", "mean_macro_f1",
        "mean_balanced_accuracy", "total_params",
    ]
    aggregate_rows = []
    for _, short_name in variant_columns:
        variant_rows = details[short_name]
        acc_values = np.asarray(
            [float(row["best_acc"]) for row in variant_rows if row.get("best_acc") not in (None, "")],
            dtype=np.float64,
        )
        macro_values = np.asarray(
            [float(row["macro_f1_at_best"]) for row in variant_rows if row.get("macro_f1_at_best") not in (None, "")],
            dtype=np.float64,
        )
        balanced_values = np.asarray(
            [float(row["balanced_accuracy"]) for row in variant_rows if row.get("balanced_accuracy") not in (None, "")],
            dtype=np.float64,
        )
        total_params_values = [row.get("total_params") for row in variant_rows if row.get("total_params") not in (None, "")]
        aggregate_rows.append({
            "variant": short_name,
            "S12_best_acc": results[short_name].get(12, ""),
            "S2_best_acc": results[short_name].get(2, ""),
            "S4_best_acc": results[short_name].get(4, ""),
            "mean_best_acc": float(acc_values.mean()) if acc_values.size else "",
            "std_best_acc": float(acc_values.std(ddof=1)) if acc_values.size > 1 else "",
            "mean_macro_f1": float(macro_values.mean()) if macro_values.size else "",
            "mean_balanced_accuracy": float(balanced_values.mean()) if balanced_values.size else "",
            "total_params": total_params_values[0] if total_params_values else "",
        })
    aggregate_csv_path = os.path.join(output_dir, "rapid_pilot3_summary.csv")
    with open(aggregate_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    summary = {
        "protocol": "rapid_pilot3",
        "targets": RAPID_TARGETS,
        "metric": "mean_best_accuracy",
        "results": results,
        "deltas": deltas,
        "promotion_status": promising,
        "comparison_csv": comparison_path,
        "aggregate_csv": aggregate_csv_path,
    }
    summary_path = os.path.join(output_dir, "rapid_pilot3_summary.json")
    _write_json(summary_path, summary)
    return comparison_path, summary_path, aggregate_csv_path


def write_cast_level1_summary(output_dir):
    variants = [
        ("E0_R4_de_only", "E0"),
        ("E1_strong_de", "E1"),
        ("E2_strong_de_channel", "E2"),
        ("E3_strong_de_channel_graph", "E3"),
        ("E4_strong_de_channel_graph_multiscale", "E4"),
        ("E5_strong_de_full_st", "E5"),
    ]
    records = {}
    for folder, short_name in variants:
        path = os.path.join(output_dir, folder, "per_subject_results.csv")
        rows = []
        if os.path.exists(path):
            with open(path, "r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        records[short_name] = rows

    fields = [
        "variant", "S12_best_acc", "S2_best_acc", "S4_best_acc",
        "mean_best_acc", "std_best_acc", "mean_macro_f1", "mean_balanced_accuracy",
        "total_params", "trainable_params",
    ]
    aggregate = []
    for _, short_name in variants:
        rows = records[short_name]
        by_target = {int(row["target_subject"]): row for row in rows}
        acc = np.asarray([float(row["best_acc"]) for row in rows], dtype=np.float64)
        macro = np.asarray([float(row["macro_f1_at_best"]) for row in rows], dtype=np.float64)
        balanced = np.asarray([float(row["balanced_accuracy"]) for row in rows], dtype=np.float64)
        aggregate.append({
            "variant": short_name,
            "S12_best_acc": by_target.get(12, {}).get("best_acc", ""),
            "S2_best_acc": by_target.get(2, {}).get("best_acc", ""),
            "S4_best_acc": by_target.get(4, {}).get("best_acc", ""),
            "mean_best_acc": float(acc.mean()) if acc.size else "",
            "std_best_acc": float(acc.std(ddof=1)) if acc.size > 1 else "",
            "mean_macro_f1": float(macro.mean()) if macro.size else "",
            "mean_balanced_accuracy": float(balanced.mean()) if balanced.size else "",
            "total_params": rows[0]["total_params"] if rows else "",
            "trainable_params": rows[0]["trainable_params"] if rows else "",
        })
    csv_path = os.path.join(output_dir, "cast_level1_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(aggregate)
    json_path = os.path.join(output_dir, "cast_level1_summary.json")
    _write_json(json_path, {
        "protocol": "cast_level1_rapid_pilot3",
        "targets": RAPID_TARGETS,
        "formal_result": "screening" not in os.path.basename(output_dir).lower() and "smoke" not in os.path.basename(output_dir).lower(),
        "variants": aggregate,
        "summary_csv": csv_path,
    })
    return json_path, csv_path


def _reshape_seedv_flat_features(data, channels=62, num_freq_bands=5):
    reshaped = []
    for session in data:
        session_out = []
        for subject in session:
            subject_out = []
            for sample in subject:
                x = np.asarray(sample)
                if x.ndim == 2:
                    expected = channels * num_freq_bands
                    if x.shape[-1] != expected:
                        raise ValueError(
                            f"SEED-V flat feature dim should be {expected}, got {x.shape[-1]}"
                        )
                    subject_out.append(x.reshape(x.shape[0], channels, num_freq_bands))
                elif x.ndim == 3:
                    subject_out.append(x)
                else:
                    raise ValueError(f"Unexpected SEED-V sample shape: {x.shape}")
            session_out.append(subject_out)
        reshaped.append(session_out)
    return reshaped


def _build_setting(args, dataset_name):
    cfg = DATASET_DEFAULTS[dataset_name]
    setting = preset_setting[args.setting](args) if args.setting is not None else set_setting_by_args(args)
    setting.dataset_path = path_mapper[cfg["path_key"]]
    setting.dataset = cfg["loader_dataset"]
    setting.experiment_mode = "subject-independent"
    setting.onehot = False
    setting.sessions = cfg["sessions"]
    setting.sample_length = args.sample_length
    setting.stride = args.stride
    setting.only_seg = False
    return setting


def load_dataset(args):
    dataset_name = args.dataset_name
    cfg = DATASET_DEFAULTS[dataset_name]

    if dataset_name == "deap":
        data, label, class_vectors, channels, num_freq_bands, num_classes, text_dim = load_deap(args, args.device)
        return data, label, class_vectors, channels, num_freq_bands, num_classes, text_dim

    if dataset_name in ("seed", "seediv"):
        setting = _build_setting(args, dataset_name)
        data, label, channels, num_freq_bands, num_classes = get_data(setting)
        data, label = flatten_trials(data, label)
        if args.subject_zscore:
            data = zscore_subject_wise(data)
            print("Applied subject-wise z-score normalization.")
        text_dataset = cfg["text_dataset"]

    elif dataset_name == "seedv":
        data, label = read_seedv_feature(path_mapper[cfg["path_key"]], cfg["sessions"])
        data, label = segment_seedv(data, label, args.sample_length, args.stride)
        data, label = flatten_trials(data, label)
        sample = _first_sample(data)
        channels, num_freq_bands, num_classes = 62, 5, 5
        if sample.ndim == 2:
            if args.subject_zscore:
                data = zscore_subject_seedv(data)
                print("Applied SEED-V subject-wise z-score normalization on flat features.")
            data = _reshape_seedv_flat_features(data, channels, num_freq_bands)
        elif sample.ndim == 3:
            if args.subject_zscore:
                data = zscore_subject_wise(data)
                print("Applied subject-wise z-score normalization.")
        else:
            raise ValueError(f"Unexpected SEED-V sample shape: {sample.shape}")
        text_dataset = cfg["text_dataset"]

    elif dataset_name == "dreamer":
        setting = _build_setting(args, dataset_name)
        setting.labeltype = args.dreamer_labeltype
        data, label, channels, num_freq_bands, num_classes = get_dreamer_data(setting)
        data, label = flatten_trials(data, label)
        if args.dreamer_ea:
            data = apply_euclidean_alignment(data)
            data, _ = global_normalization_after_ea(data)
            print("Applied DREAMER Euclidean alignment and global normalization.")
        elif args.subject_zscore:
            data = zscore_subject_wise(data)
            print("Applied subject-wise z-score normalization.")
        text_dataset = cfg["text_dataset"]

    else:
        raise ValueError(f"Unsupported dataset_name: {dataset_name}")

    text_dim, class_vectors = label_to_vector(
        dataset=text_dataset,
        LM=args.LM,
        LabelTextMapper=None,
        device=args.device,
    )
    return data, label, class_vectors, channels, num_freq_bands, num_classes, text_dim


def _apply_dataset_defaults(args):
    cfg = DATASET_DEFAULTS[args.dataset_name]
    args.sample_length = cfg["sample_length"] if args.sample_length is None else args.sample_length
    args.stride = cfg["stride"] if args.stride is None else args.stride
    args.epochs = 200 if args.epochs is None else args.epochs
    args.batch_size = cfg["batch_size"] if args.batch_size is None else args.batch_size
    args.lr = 1e-3 if args.lr is None else args.lr
    args.lambda_max = cfg["lambda_max"] if args.lambda_max is None else args.lambda_max
    args.lambda_min = cfg["lambda_min"] if args.lambda_min is None else args.lambda_min
    args.topk = cfg["topk"] if args.topk is None else args.topk
    args.geometry_batch_size = (
        cfg["geometry_batch_size"] if args.geometry_batch_size is None else args.geometry_batch_size
    )
    return args


def _validate_args(args):
    if args.st_dim % args.heads != 0:
        raise ValueError(f"st_dim={args.st_dim} must be divisible by heads={args.heads}")
    if args.graph_dim % args.graph_heads != 0:
        raise ValueError(f"graph_dim={args.graph_dim} must be divisible by graph_heads={args.graph_heads}")
    if args.mmd_warmup_ratio <= 0.0 or args.mmd_warmup_ratio > 1.0:
        raise ValueError(f"mmd_warmup_ratio must be in (0, 1], got {args.mmd_warmup_ratio}")
    if args.mmd_hold_ratio < args.mmd_warmup_ratio or args.mmd_hold_ratio > 1.0:
        raise ValueError(f"mmd_hold_ratio must be in [{args.mmd_warmup_ratio}, 1], got {args.mmd_hold_ratio}")
    if not (0.0 <= args.mmd_start_ratio < 1.0):
        raise ValueError(f"mmd_start_ratio must be in [0, 1), got {args.mmd_start_ratio}")
    if args.uot_epsilon <= 0.0 or args.uot_tau_s <= 0.0 or args.uot_tau_t <= 0.0:
        raise ValueError("uot_epsilon, uot_tau_s, and uot_tau_t must be positive")
    if args.uot_route_tau <= 0.0:
        raise ValueError(f"uot_route_tau must be positive, got {args.uot_route_tau}")
    if args.uot_n_iter <= 0:
        raise ValueError(f"uot_n_iter must be positive, got {args.uot_n_iter}")
    if args.cast_balanced_screening and not args.cast_level1:
        raise ValueError("--cast_balanced_screening is valid only for E0-E5 CAST variants")
    if args.cast_balanced_screening and args.cast_samples_per_class <= 0:
        raise ValueError("--cast_samples_per_class must be positive")
    if args.cast_freeze_loaded and not args.cast_pretrained_checkpoint:
        raise ValueError("--cast_freeze_loaded requires --cast_pretrained_checkpoint")
    if (args.cast_pretrained_checkpoint or args.cast_save_final_checkpoint) and not args.cast_level1:
        raise ValueError("CAST checkpoint options are valid only for E0-E5 variants")


def run(args):
    setup_seed(args.seed)
    args.dataset_name = args.dataset_name.lower()
    if not os.path.isabs(args.pilot_config):
        args.pilot_config = os.path.join(project_root, args.pilot_config)
    if not os.path.isabs(args.fixed_source_config):
        args.fixed_source_config = os.path.join(project_root, args.fixed_source_config)
    if args.cast_pretrained_checkpoint and "{target" not in args.cast_pretrained_checkpoint:
        if not os.path.isabs(args.cast_pretrained_checkpoint):
            args.cast_pretrained_checkpoint = os.path.join(project_root, args.cast_pretrained_checkpoint)
    args = _apply_rapid_variant(args)
    args = _apply_dataset_defaults(args)
    if not args.rapid_pilot3:
        args = _apply_experiment_variant(args)
    args.device = torch.device(args.device if torch.cuda.is_available() and str(args.device).startswith("cuda") else "cpu")
    args.LM = "clip" if args.LM is None else args.LM
    _validate_args(args)

    print(f"Dataset: {args.dataset_name}")
    print(f"Device: {args.device}")
    print(f"Experiment variant: {args.experiment_label}")

    data, label, class_vectors, channels, num_freq_bands, num_classes, text_dim = load_dataset(args)
    if args.topk >= channels:
        raise ValueError(f"topk={args.topk} must be smaller than channels={channels}")

    sample = _first_sample(data)
    if sample.ndim != 3:
        raise ValueError(f"Expected sample shape [L,C,F], got {sample.shape}")
    if sample.shape != (args.sample_length, channels, num_freq_bands):
        raise ValueError(
            f"Dimension mismatch: sample={sample.shape}, expected "
            f"({args.sample_length}, {channels}, {num_freq_bands})"
        )

    if args.build_pilot:
        build_pilot_protocol(args, data, channels, num_freq_bands)
        return

    text_prototypes = torch.tensor(
        np.asarray([class_vectors[i] for i in sorted(class_vectors.keys())], dtype=np.float32),
        device=args.device,
    )
    n_sessions = len(data)
    session_indices = _parse_session_indices(args.session_ids, n_sessions)
    n_subjects = len(data[0])
    pilot_config = None
    fixed_sources = None
    if args.rapid_pilot3:
        pilot_config = load_pilot_protocol(args.pilot_config)
        target_indices = _select_target_indices(
            n_subjects,
            target_subject_ids=args.target_subject_ids,
            random_target_count=None,
            target_seed=args.target_seed,
        )
        invalid_targets = [idx + 1 for idx in target_indices if idx + 1 not in RAPID_TARGETS]
        if invalid_targets:
            raise ValueError(f"Rapid Pilot-3 targets must be drawn from {RAPID_TARGETS}, got {invalid_targets}")
        fixed_sources = _load_fixed_sources(args.fixed_source_config, n_subjects)
    elif args.pilot_mode:
        pilot_config = load_pilot_protocol(args.pilot_config)
        if pilot_config.get("dataset") != args.dataset_name:
            raise ValueError(
                f"Pilot config dataset={pilot_config.get('dataset')} does not match dataset_name={args.dataset_name}"
            )
        target_indices = [int(sid) - 1 for sid in pilot_config["pilot_subject_ids"]]
        if args.pilot_target_limit is not None:
            if args.pilot_target_limit <= 0:
                raise ValueError("--pilot_target_limit must be positive")
            target_indices = target_indices[:args.pilot_target_limit]
    else:
        target_indices = _select_target_indices(
            n_subjects,
            target_subject_ids=args.target_subject_ids,
            random_target_count=args.random_target_count,
            target_seed=args.target_seed,
        )

    difficulty_scores, difficulty_levels = _pilot_subject_metadata(pilot_config)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"{args.dataset_name}_sgda_ep{args.epochs}_bs{args.batch_size}_lr{_format_float(args.lr)}_"
        f"lmda{_format_float(args.lambda_max)}_{args.mmd_type}_{args.mmd_schedule}_seed{args.seed}_{timestamp}"
    )
    if args.rapid_pilot3:
        if args.cast_level1:
            if args.rapid_smoke_test:
                rapid_output_name = "dreamer_cast_level1_smoke"
            elif args.cast_balanced_screening:
                rapid_output_name = f"dreamer_cast_level1_screening_n{args.cast_samples_per_class}"
            else:
                rapid_output_name = "dreamer_cast_level1"
        else:
            rapid_output_name = "dreamer_rapid_pilot3_smoke" if args.rapid_smoke_test else "dreamer_rapid_pilot3"
        output_dir = os.path.join(project_root, "results", rapid_output_name)
        run_dir = os.path.join(output_dir, args.experiment_label)
        subject_csv = os.path.join(run_dir, "per_subject_results.csv")
        summary_json_path = os.path.join(run_dir, "summary.json")
        summary_path = None
        source_selection_path = os.path.join(run_dir, "source_selection.json")
    elif args.pilot_mode:
        pilot_output_name = args.pilot_output_name or f"{args.dataset_name}_pilot{len(pilot_config['pilot_subject_ids'])}"
        pilot_root = os.path.join(project_root, "results", pilot_output_name)
        output_dir = pilot_root
        run_dir = os.path.join(output_dir, args.experiment_label)
        subject_csv = os.path.join(run_dir, "per_subject_results.csv")
        summary_json_path = os.path.join(run_dir, "summary.json")
        summary_path = os.path.join(output_dir, "pilot5_diagnostic_summary.csv")
        source_selection_path = os.path.join(run_dir, "source_selection.json")
    else:
        output_dir = os.path.join(project_root, "results", f"results_{args.dataset_name}_geosem_stda")
        run_dir = os.path.join(output_dir, "runs", run_id)
        subject_csv = os.path.join(run_dir, f"subject_results_{args.dataset_name}_geosem_stda.csv")
        summary_json_path = None
        summary_path = os.path.join(output_dir, f"summary_{args.dataset_name}_geosem_stda_runs.csv")
        source_selection_path = os.path.join(run_dir, "source_selection.json")
    os.makedirs(run_dir, exist_ok=True)
    epoch_log_path = os.path.join(run_dir, "epoch_log.csv")
    config_path = os.path.join(run_dir, "run_config.json")
    training_log_path = os.path.join(run_dir, "training_log.txt")
    if args.rapid_pilot3:
        # A named Rapid variant is one atomic run; do not append stale epochs on rerun.
        for stale_path in (epoch_log_path, training_log_path):
            if os.path.exists(stale_path):
                os.remove(stale_path)

    _log(f"Run directory: {run_dir}", training_log_path)

    _log(f"Sessions evaluated: {[idx + 1 for idx in session_indices]}", training_log_path)
    _log(f"Targets evaluated: {[idx + 1 for idx in target_indices]}", training_log_path)
    _log(
        f"Dimensions: X=[N,{args.sample_length},{channels},{num_freq_bands}], "
        f"classes={num_classes}, text_dim={text_dim}",
        training_log_path,
    )
    if args.rapid_variant == "r4_de_only_clip" and args.rapid_pilot3:
        _log("[R4] Pure DE-only representation enabled", training_log_path)
        _log("[R4] Structured GeoSem branch disabled", training_log_path)
        _log("[R4] CLIP prototype head enabled", training_log_path)
        _log("[R4] Alignment losses disabled", training_log_path)
        _log("[R4] Senior feature fusion unchanged", training_log_path)
        _log("[R4] SPD/Log-Euclidean geometry construction disabled", training_log_path)
        _log("[R4] Source adapters unchanged", training_log_path)
    if args.cast_level1:
        _log(f"[CAST-L1] Variant: {args.rapid_variant}", training_log_path)
        _log("[CAST-L1] No SPD, Log-Euclidean, alignment, ReSGCA, or UOT path", training_log_path)
        _log("[CAST-L1] Target data remain full-size and evaluation-only", training_log_path)
        if args.cast_balanced_screening:
            _log(
                f"[CAST-L1] Screening uses at most {args.cast_samples_per_class} samples/class/source",
                training_log_path,
            )

    run_config = {
        "run_id": run_id,
        "script": os.path.abspath(__file__),
        "dataset_name": args.dataset_name,
        "experiment_variant": args.experiment_variant,
        "experiment_label": args.experiment_label,
        "protocol": "rapid_pilot3" if args.rapid_pilot3 else ("pilot5" if args.pilot_mode else "full"),
        "rapid_pilot3": args.rapid_pilot3,
        "rapid_variant": args.rapid_variant if args.rapid_pilot3 else None,
        "rapid_smoke_test": args.rapid_smoke_test if args.rapid_pilot3 else False,
        "cast_level1": args.cast_level1,
        "cast_variant": args.cast_variant,
        "cast_balanced_screening": args.cast_balanced_screening,
        "cast_samples_per_class": args.cast_samples_per_class if args.cast_balanced_screening else None,
        "cast_pretrained_checkpoint": args.cast_pretrained_checkpoint,
        "cast_freeze_loaded": args.cast_freeze_loaded,
        "cast_save_final_checkpoint": args.cast_save_final_checkpoint,
        "representation_mode": args.representation_mode,
        "classifier_type": args.classifier_type,
        "fixed_source_config": args.fixed_source_config if args.rapid_pilot3 else None,
        "pilot_mode": args.pilot_mode,
        "pilot_config": args.pilot_config if args.pilot_mode else None,
        "evaluated_sessions": [idx + 1 for idx in session_indices],
        "evaluated_target_subjects": [idx + 1 for idx in target_indices],
        "target_labels_used_for_training": False,
        "target_labels_used_for_epoch_selection": True,
        "protocol_caveat": (
            "Best epoch is selected on target accuracy to remain comparable with the existing Rapid Pilot-3; "
            "this is not a strict held-out model-selection protocol."
        ),
        "source_candidate_count_per_target": n_subjects - 1,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sample_length": args.sample_length,
        "stride": args.stride,
        "lr": args.lr,
        "backbone_lr": args.backbone_lr,
        "use_param_groups": args.use_param_groups,
        "selection_warmup_proto_only": args.selection_warmup_proto_only,
        "seed": args.seed,
        "lambda_max": args.lambda_max,
        "lambda_min": args.lambda_min,
        "mmd_type": args.mmd_type,
        "mmd_schedule": args.mmd_schedule,
        "mmd_start_ratio": args.mmd_start_ratio,
        "mmd_confidence_gate": args.mmd_confidence_gate,
        "resgca_geo_tau": args.resgca_geo_tau,
        "resgca_geo_weight": args.resgca_geo_weight,
        "uot_epsilon": args.uot_epsilon,
        "uot_tau_s": args.uot_tau_s,
        "uot_tau_t": args.uot_tau_t,
        "uot_route_tau": args.uot_route_tau,
        "uot_n_iter": args.uot_n_iter,
        "hut_geo_cost_weight": args.hut_geo_cost_weight,
        "hut_agreement_tau": args.hut_agreement_tau,
        "hut_use_agreement_mass": args.hut_use_agreement_mass,
        "hut_use_geometry_cost": args.hut_use_geometry_cost,
        "eval_classifier": args.eval_classifier,
        "reliability_fusion": args.reliability_fusion,
        "centroid_blend": args.centroid_blend,
        "source_selection": args.source_selection,
        "sparse_k_max": args.sparse_k_max,
        "channels": channels,
        "num_freq_bands": num_freq_bands,
        "num_classes": num_classes,
        "text_dim": text_dim,
        "total_params": None,
        "trainable_params": None,
        "parameter_breakdown": None,
        "checkpoint_loads": [],
    }
    _write_json(config_path, run_config)

    epoch_fields = [
        "run_id", "dataset_name", "session_idx", "target_subject", "epoch",
        "loss", "proto", "align", "lambda", "sca_mu", "acc", "macro_f1", "micro_f1",
        "best_acc", "best_epoch", "alpha_mean", "final_source_count",
        "train_loss", "classification_loss", "align_loss", "target_acc",
        "target_macro_f1", "best_acc_so_far", "beta",
    ]
    subject_fields = [
        "run_id", "experiment", "dataset_name", "session_idx", "target_subject",
        "difficulty_level", "difficulty_score", "source_candidate_count",
        "candidate_source_ids", "final_source_count", "selected_source_ids",
        "best_epoch", "acc", "best_acc", "macro_f1", "macro_f1_at_best", "micro_f1", "seed",
        "epochs", "classifier_type", "representation_mode", "cast_variant", "screening_mode", "beta_initial",
        "beta_at_best", "beta_final", "fixed_beta", "total_params", "trainable_params",
        "balanced_accuracy", "recall_class0", "recall_class1",
        "TN", "FP", "FN", "TP",
        "predicted_class0_ratio", "predicted_class1_ratio",
        "true_class0_ratio", "true_class1_ratio",
        "final_acc", "final_macro_f1",
    ]
    subject_records = []
    source_selection_records = []
    screening_manifest_records = []

    for session_idx in session_indices:
        for target_sub in target_indices:
            setup_seed(args.seed)
            candidate_source_ids = [sid + 1 for sid in range(n_subjects) if sid != target_sub]
            all_source_count = len(candidate_source_ids)
            source_ids = (
                list(fixed_sources[target_sub])
                if args.rapid_pilot3
                else [sid for sid in range(n_subjects) if sid != target_sub]
            )
            _log(f"\n{'=' * 64}", training_log_path)
            _log(
                f"{args.dataset_name.upper()} Session {session_idx + 1}/{n_sessions} | "
                f"Target subject {target_sub + 1}/{n_subjects} | "
                f"Candidate sources: {len(source_ids)}",
                training_log_path,
            )
            _log(f"{'=' * 64}", training_log_path)

            # DE-only receives a shape-compatible placeholder and never enters the
            # covariance/SPD/Log-Euclidean geometry builder.
            r_by_subject, geometry_bypassed = _build_fold_representation_inputs(
                args,
                data[session_idx],
                source_ids,
                target_sub,
            )
            if args.representation_mode in ("de_only", "cast_level1") and not geometry_bypassed:
                raise AssertionError("DE-only/CAST unexpectedly invoked structured geometry construction")

            source_loaders, source_class_weights = [], []
            for sid in source_ids:
                x_array = np.asarray(data[session_idx][sid])
                y_array = np.asarray(label[session_idx][sid]).reshape(-1)
                r = r_by_subject[sid].float()
                if args.cast_balanced_screening:
                    indices, manifest = _balanced_source_indices(
                        y_array, args.cast_samples_per_class, args.seed, sid + 1
                    )
                    x_array = x_array[indices]
                    y_array = y_array[indices]
                    r = r[torch.as_tensor(indices, dtype=torch.long)]
                    manifest.update({
                        "session_idx": session_idx + 1,
                        "target_subject": target_sub + 1,
                        "variant": args.rapid_variant,
                    })
                    screening_manifest_records.append(manifest)
                x = torch.tensor(x_array, dtype=torch.float32)
                y = torch.tensor(y_array, dtype=torch.long)
                source_class_weights.append(_compute_class_weights(y_array, num_classes, args.device))
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
            target_train_loader = None
            if not args.rapid_pilot3:
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
            selected_weights = None
            selection_details = [
                {
                    "sid": sid,
                    "score": None,
                    "d_marg": None,
                    "d_cond": None,
                    "src_acc_proxy": None,
                }
                for sid in source_ids
            ]
            if args.rapid_pilot3:
                selected_ids = list(source_ids)
                selected_weights = [1.0 / len(source_ids)] * len(source_ids)
                source_loss_weights = None
                _log(f"Frozen P0 sources used for Rapid Pilot-3: {[sid + 1 for sid in source_ids]}", training_log_path)
            elif args.source_selection == "sparse_reliability" and len(source_loaders) > 1:
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
                selected_ids, selected_weights, selection_details = sparse_reliability_source_selection(
                    warmup_model,
                    source_loaders,
                    target_eval_loader,
                    source_ids,
                    text_prototypes,
                    num_classes,
                    args,
                    return_details=True,
                )
                selected_set = set(selected_ids)
                keep_indices = [idx for idx, sid in enumerate(source_ids) if sid in selected_set]
                source_ids = [source_ids[idx] for idx in keep_indices]
                source_loaders = [source_loaders[idx] for idx in keep_indices]
                source_class_weights = [source_class_weights[idx] for idx in keep_indices]
                weight_by_id = {sid: weight for sid, weight in zip(selected_ids, selected_weights)}
                source_loss_weights = [weight_by_id[sid] for sid in source_ids]
                _log(f"Sparse selected sources used for final training: {[sid + 1 for sid in source_ids]}", training_log_path)
            else:
                selected_ids = list(source_ids)
                selected_weights = [1.0 / max(len(source_ids), 1)] * len(source_ids)
                _log(f"Selected sources used for final training: {[sid + 1 for sid in source_ids]}", training_log_path)

            record_source_weights = source_loss_weights if source_loss_weights is not None else selected_weights
            source_selection_records.append({
                "experiment": args.experiment_label,
                "dataset_name": args.dataset_name,
                "session_idx": session_idx + 1,
                "target_subject": target_sub + 1,
                "candidate_sources": candidate_source_ids,
                "selected_sources": [sid + 1 for sid in source_ids],
                "source_reliability": {
                    str(sid + 1): float(weight)
                    for sid, weight in zip(source_ids, record_source_weights)
                },
                "source_selection_details": [
                    {
                        "subject_id": int(item["sid"]) + 1,
                        "score": item["score"],
                        "marginal_distance": item["d_marg"],
                        "conditional_distance": item["d_cond"],
                        "source_acc_proxy": item["src_acc_proxy"],
                    }
                    for item in selection_details
                ],
            })

            model = _make_model_for_run(
                args,
                len(source_loaders),
                channels,
                num_freq_bands,
                text_dim,
                num_classes,
            )
            checkpoint_path = _resolve_cast_checkpoint_path(
                args.cast_pretrained_checkpoint, target_sub + 1
            )
            checkpoint_info = _load_and_optionally_freeze_cast(
                model, checkpoint_path, args.cast_freeze_loaded
            )
            checkpoint_info["target_subject"] = target_sub + 1
            run_config["checkpoint_loads"].append(checkpoint_info)
            optimizer = _make_optimizer_for_run(args, model)
            total_params, trainable_params = _count_parameters(model)
            parameter_breakdown = _parameter_breakdown(model)
            if args.representation_mode == "de_only":
                if model.encoder is not None or parameter_breakdown["structured_encoder_params"] != 0:
                    raise AssertionError("R4 must have no structured encoder instance or parameters")
            if args.cast_level1 and model.encoder is not None:
                raise AssertionError("CAST Level-1 must not instantiate the geometry-dependent GeoSem encoder")
            if run_config["total_params"] is None:
                run_config["total_params"] = total_params
                run_config["trainable_params"] = trainable_params
                run_config["parameter_breakdown"] = parameter_breakdown
            _write_json(config_path, run_config)
            _log(
                f"Parameters: total={total_params:,}, trainable={trainable_params:,}",
                training_log_path,
            )
            _log(
                "Parameter breakdown: "
                f"direct={parameter_breakdown['direct_de_encoder_params']:,}, "
                f"structured={parameter_breakdown['structured_encoder_params']:,}, "
                f"cast_level1={parameter_breakdown['cast_level1_encoder_params']:,}, "
                f"adapters={parameter_breakdown['adapters_params']:,}, "
                f"head={parameter_breakdown['head_prototype_related_params']:,}, "
                f"fusion={parameter_breakdown['fusion_params']:,}",
                training_log_path,
            )
            steps_per_epoch = min(len(loader) for loader in source_loaders)
            total_steps = max(args.epochs * steps_per_epoch, 1)
            source_iters = [iter(loader) for loader in source_loaders]
            target_iter = iter(target_train_loader) if target_train_loader is not None else None
            global_step = 0
            best_acc, best_macro, best_micro, best_epoch = -1.0, 0.0, 0.0, 0
            beta_initial = model.current_beta() if model.current_beta() is not None else np.nan
            beta_at_best = np.nan
            best_diagnostics = None
            final_acc, final_macro = np.nan, np.nan
            smoke_gradient_checked = False

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

                    optimizer.zero_grad()
                    if args.rapid_pilot3:
                        z_src_all, _, _, _, alpha_src, _ = model(
                            x_src_list,
                            r_src_list,
                            return_features=True,
                        )
                        if args.classifier_type == "linear":
                            loss_proto = linear_classification_loss(
                                z_src_all,
                                y_src_list,
                                model.linear_head,
                                class_weights_list=source_class_weights,
                                source_weights=None,
                            )
                        else:
                            loss_proto = prototype_contrastive_loss(
                                z_src_all,
                                y_src_list,
                                text_prototypes,
                                tau=args.proto_tau,
                                class_weights_list=source_class_weights,
                                source_weights=None,
                            )
                        loss_align = torch.zeros((), device=args.device)
                        lambda_val = 0.0
                        alpha_tgt = None
                    else:
                        try:
                            x_tb, r_tb, _ = next(target_iter)
                        except StopIteration:
                            target_iter = iter(target_train_loader)
                            x_tb, r_tb, _ = next(target_iter)
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
                        loss_align = compute_alignment_loss(
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
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss at step {global_step}: {loss.item()}")
                    loss.backward()
                    if args.rapid_smoke_test and not smoke_gradient_checked:
                        shapes = model.last_representation_shapes
                        if args.representation_mode == "de_only":
                            expected = (x_src_list[-1].size(0), args.st_dim)
                            if shapes != {"h_de": expected, "h_struct": None, "h_fused": expected}:
                                raise AssertionError(f"R4 DE-only representation shape mismatch: {shapes}, expected={expected}")
                            if model.encoder is not None:
                                raise AssertionError("R4 must not instantiate the structured GeoSem encoder")
                        elif args.representation_mode != "geosem":
                            expected = (x_src_list[-1].size(0), args.st_dim)
                            if shapes["h_de"] != expected or (
                                args.representation_mode != "cast_level1" and shapes["h_struct"] != expected
                            ):
                                raise AssertionError(f"Rapid representation shape mismatch: {shapes}, expected={expected}")
                        if args.representation_mode == "cast_level1":
                            if shapes["h_fused"] != expected:
                                raise AssertionError(f"CAST output shape mismatch: {shapes}, expected={expected}")
                            cast_grads = [
                                p.grad for p in model.cast_encoder.parameters()
                                if p.requires_grad and p.grad is not None
                            ]
                            if not cast_grads or not all(torch.isfinite(g).all() for g in cast_grads):
                                raise AssertionError("CAST trainable gradients are missing or non-finite")
                        if args.representation_mode == "de_gated":
                            grad = model.residual_fusion.beta_logit.grad
                            if grad is None or not torch.isfinite(grad).all():
                                raise AssertionError("beta_logit.grad is missing or non-finite")
                            _log(f"Smoke gradient: beta_logit.grad={grad.item():.8f}", training_log_path)
                        if args.classifier_type == "linear":
                            grad = model.linear_head.linear.weight.grad
                            if grad is None or not torch.isfinite(grad).all() or grad.abs().sum() == 0:
                                raise AssertionError("LinearClassifier gradient is missing, zero, or non-finite")
                            _log(
                                f"Smoke gradient: linear_head.grad_norm={grad.norm().item():.8f}; CLIP unused in loss/prediction",
                                training_log_path,
                            )
                        _log(f"Smoke representation shapes: {shapes}", training_log_path)
                        smoke_gradient_checked = True
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_proto += loss_proto.item()
                    epoch_align += loss_align.item()
                    if alpha_src:
                        alpha_values.append(torch.cat([a.detach().cpu() for a in alpha_src]).mean().item())
                    if alpha_tgt is not None:
                        alpha_values.append(alpha_tgt.detach().cpu().mean().item())

                mean_alpha = float(np.mean(alpha_values)) if alpha_values else 0.0
                should_eval = ((epoch + 1) % args.eval_interval == 0) or (epoch + 1 == args.epochs)
                acc, macro_f1, micro_f1 = np.nan, np.nan, np.nan

                if should_eval:
                    model.eval()
                    source_centroids = (
                        None
                        if args.rapid_pilot3
                        else compute_source_class_centroids(
                            model,
                            source_loaders,
                            args.device,
                            num_classes,
                        )
                    )
                    source_domain_centroids = (
                        compute_source_domain_centroids(model, source_loaders, args.device)
                        if args.eval_classifier == "senior_feature"
                        else None
                    )
                    y_true, y_pred = evaluate(
                        model,
                        target_eval_loader,
                        text_prototypes,
                        source_centroids,
                        args.device,
                        proto_tau=args.proto_tau,
                        fusion_tau=args.fusion_tau,
                        eval_classifier=args.eval_classifier,
                        centroid_blend=args.centroid_blend,
                        source_domain_centroids=source_domain_centroids,
                        source_reliability_weights=source_loss_weights,
                        reliability_fusion=args.reliability_fusion,
                    )
                    acc = float((y_true == y_pred).mean())
                    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))
                    micro_f1 = float(f1_score(y_true, y_pred, average="micro"))
                    final_acc, final_macro = acc, macro_f1
                    if acc > best_acc:
                        best_acc, best_macro, best_micro = acc, macro_f1, micro_f1
                        best_epoch = epoch + 1
                        if args.representation_mode == "de_only" or args.cast_level1:
                            best_diagnostics = _classification_diagnostics(y_true, y_pred)
                        beta_at_best = (
                            model.current_beta()
                            if args.representation_mode == "de_gated" or args.cast_variant == "e5_strong_de_full_st"
                            else np.nan
                        )
                        _log(f"  >> New best acc: {best_acc:.4f} (epoch {best_epoch})", training_log_path)

                if (epoch + 1) % args.log_interval == 0:
                    acc_text = f"{acc:.4f}" if should_eval else "skip"
                    sca_mu_text = f" mu={sca_mu_val:.3f}" if args.mmd_type in ("sca", "resgca") else ""
                    beta_value = model.current_beta()
                    beta_value = np.nan if beta_value is None else beta_value
                    beta_text = f" beta={beta_value:.6f}" if np.isfinite(beta_value) else ""
                    _log(
                        f"Ep {epoch + 1:3d} | loss={epoch_loss / steps_per_epoch:.4f} "
                        f"classification={epoch_proto / steps_per_epoch:.4f} "
                        f"align={epoch_align / steps_per_epoch:.6f} "
                        f"lambda={lambda_val:.4f}{sca_mu_text} acc={acc_text} "
                        f"best={best_acc:.4f} alpha={mean_alpha:.3f}{beta_text}",
                        training_log_path,
                    )

                beta_value = model.current_beta()
                beta_value = np.nan if beta_value is None else beta_value
                _append_csv_row(epoch_log_path, epoch_fields, {
                    "run_id": run_id,
                    "dataset_name": args.dataset_name,
                    "session_idx": session_idx + 1,
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
                    "best_epoch": best_epoch,
                    "alpha_mean": mean_alpha,
                    "final_source_count": len(source_loaders),
                    "train_loss": epoch_loss / steps_per_epoch,
                    "classification_loss": epoch_proto / steps_per_epoch,
                    "align_loss": epoch_align / steps_per_epoch,
                    "target_acc": acc,
                    "target_macro_f1": macro_f1,
                    "best_acc_so_far": best_acc,
                    "beta": beta_value,
                })

            _log(
                f"Final target {target_sub + 1}: best_acc={best_acc * 100:.2f}% at epoch {best_epoch}; "
                f"final_acc={final_acc * 100:.2f}%, final_macro_f1={final_macro:.4f}",
                training_log_path,
            )
            if args.cast_level1 and args.cast_save_final_checkpoint:
                checkpoint_output_path = os.path.join(
                    run_dir, f"final_checkpoint_subject{target_sub + 1:02d}.pt"
                )
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "variant": args.rapid_variant,
                    "target_subject": target_sub + 1,
                    "epoch": args.epochs,
                    "selection_rule": "fixed_final_epoch_not_target_best",
                }, checkpoint_output_path)
            target_subject_1based = target_sub + 1
            difficulty_score = difficulty_scores.get(target_subject_1based, np.nan)
            difficulty_level = difficulty_levels.get(target_subject_1based, "")
            diagnostics = best_diagnostics or {
                "balanced_accuracy": np.nan, "recall_class0": np.nan, "recall_class1": np.nan,
                "TN": np.nan, "FP": np.nan, "FN": np.nan, "TP": np.nan,
                "predicted_class0_ratio": np.nan, "predicted_class1_ratio": np.nan,
                "true_class0_ratio": np.nan, "true_class1_ratio": np.nan,
            }
            subject_record = {
                "run_id": run_id,
                "experiment": args.experiment_label,
                "dataset_name": args.dataset_name,
                "session_idx": session_idx + 1,
                "target_subject": target_subject_1based,
                "difficulty_level": difficulty_level,
                "difficulty_score": difficulty_score,
                "source_candidate_count": all_source_count,
                "candidate_source_ids": json.dumps(candidate_source_ids),
                "final_source_count": len(source_ids),
                "selected_source_ids": json.dumps([sid + 1 for sid in source_ids]),
                "best_epoch": best_epoch,
                "acc": best_acc,
                "best_acc": best_acc,
                "macro_f1": best_macro,
                "macro_f1_at_best": best_macro,
                "micro_f1": best_micro,
                "seed": args.seed,
                "epochs": args.epochs,
                "classifier_type": args.classifier_type,
                "representation_mode": args.representation_mode,
                "cast_variant": args.cast_variant or ("e0_r4_de_only" if args.cast_level1 else ""),
                "screening_mode": args.cast_balanced_screening,
                "beta_initial": beta_initial,
                "beta_at_best": beta_at_best,
                "beta_final": model.current_beta() if model.current_beta() is not None else np.nan,
                "fixed_beta": 1.0 if args.representation_mode == "de_residual" else np.nan,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "balanced_accuracy": diagnostics["balanced_accuracy"],
                "recall_class0": diagnostics["recall_class0"],
                "recall_class1": diagnostics["recall_class1"],
                "TN": diagnostics["TN"], "FP": diagnostics["FP"],
                "FN": diagnostics["FN"], "TP": diagnostics["TP"],
                "predicted_class0_ratio": diagnostics["predicted_class0_ratio"],
                "predicted_class1_ratio": diagnostics["predicted_class1_ratio"],
                "true_class0_ratio": diagnostics["true_class0_ratio"],
                "true_class1_ratio": diagnostics["true_class1_ratio"],
                "final_acc": final_acc,
                "final_macro_f1": final_macro,
            }
            subject_records.append(subject_record)
            if args.representation_mode == "de_only" or args.cast_level1:
                diagnostics_payload = {
                    "target_subject": target_subject_1based,
                    "variant": args.rapid_variant,
                    "best_epoch": best_epoch,
                    "best_acc": best_acc,
                    "best_macro_f1": best_macro,
                    **best_diagnostics,
                }
                diagnostics_path = os.path.join(
                    output_dir,
                    f"{args.rapid_variant}_subject{target_subject_1based:02d}_diagnostics.json",
                )
                _write_json(diagnostics_path, diagnostics_payload)

    with open(subject_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=subject_fields)
        writer.writeheader()
        writer.writerows(subject_records)

    all_acc = np.asarray([row["acc"] for row in subject_records], dtype=np.float32)
    all_macro = np.asarray([row["macro_f1"] for row in subject_records], dtype=np.float32)
    all_micro = np.asarray([row["micro_f1"] for row in subject_records], dtype=np.float32)
    all_balanced = np.asarray([row["balanced_accuracy"] for row in subject_records], dtype=np.float32)
    summary = {
        "run_id": run_id,
        "created_at": timestamp,
        "experiment": args.experiment_label,
        "experiment_variant": args.experiment_variant,
        "protocol": "rapid_pilot3" if args.rapid_pilot3 else ("pilot5" if args.pilot_mode else "full"),
        "rapid_variant": args.rapid_variant if args.rapid_pilot3 else None,
        "cast_level1": args.cast_level1,
        "cast_variant": args.cast_variant,
        "screening_mode": args.cast_balanced_screening,
        "representation_mode": args.representation_mode,
        "classifier_type": args.classifier_type,
        "dataset_name": args.dataset_name,
        "task": args.dreamer_labeltype if args.dataset_name == "dreamer" else "",
        "pilot_mode": args.pilot_mode,
        "pilot_subjects": [idx + 1 for idx in target_indices] if (args.pilot_mode or args.rapid_pilot3) else [],
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "sample_length": args.sample_length,
        "stride": args.stride,
        "lr": args.lr,
        "seed": args.seed,
        "lambda_max": args.lambda_max,
        "lambda_min": args.lambda_min,
        "mmd_type": args.mmd_type,
        "mmd_schedule": args.mmd_schedule,
        "source_selection": args.source_selection,
        "sparse_k_max": args.sparse_k_max,
        "eval_classifier": args.eval_classifier,
        "reliability_fusion": args.reliability_fusion,
        "acc_mean": float(all_acc.mean()),
        "acc_std": float(all_acc.std(ddof=1)) if all_acc.size > 1 else 0.0,
        "mean_best_acc": float(all_acc.mean()),
        "std_best_acc": float(all_acc.std(ddof=1)) if all_acc.size > 1 else 0.0,
        "macro_f1_mean": float(all_macro.mean()),
        "macro_f1_std": float(all_macro.std(ddof=1)) if all_macro.size > 1 else 0.0,
        "mean_macro_f1": float(all_macro.mean()),
        "mean_balanced_accuracy": (
            float(np.nanmean(all_balanced)) if np.isfinite(all_balanced).any() else None
        ),
        "micro_f1_mean": float(all_micro.mean()),
        "micro_f1_std": float(all_micro.std(ddof=1)) if all_micro.size > 1 else 0.0,
        "per_subject_best_acc": {
            str(row["target_subject"]): row["acc"]
            for row in subject_records
        },
        "mean_best_epoch": float(np.asarray([row["best_epoch"] for row in subject_records], dtype=np.float32).mean()),
        "beta_initial": (
            subject_records[0]["beta_initial"]
            if args.representation_mode == "de_gated" or args.cast_variant == "e5_strong_de_full_st"
            else None
        ),
        "beta_at_best_by_subject": {
            str(row["target_subject"]): row["beta_at_best"]
            for row in subject_records
        } if args.representation_mode == "de_gated" or args.cast_variant == "e5_strong_de_full_st" else {},
        "beta_final_by_subject": {
            str(row["target_subject"]): row["beta_final"]
            for row in subject_records
        } if args.representation_mode == "de_gated" or args.cast_variant == "e5_strong_de_full_st" else {},
        "total_params": run_config["total_params"],
        "trainable_params": run_config["trainable_params"],
        "subject_results_csv": subject_csv,
        "epoch_log_csv": epoch_log_path,
        "run_config_json": config_path,
        "source_selection_json": source_selection_path,
        "target_labels_used_for_training": False,
        "target_labels_used_for_epoch_selection": True,
    }
    _write_json(source_selection_path, source_selection_records)
    if args.cast_balanced_screening:
        _write_json(os.path.join(run_dir, "balanced_source_manifest.json"), screening_manifest_records)

    if args.rapid_pilot3:
        if args.rapid_variant == "r4_de_only_clip":
            r4_summary_path = os.path.join(output_dir, "R4_de_only_clip_summary.csv")
            r4_fields = [
                "target_subject", "variant", "best_acc", "best_macro_f1", "best_epoch",
                "balanced_accuracy", "recall_class0", "recall_class1",
                "final_acc", "final_macro_f1", "total_params",
            ]
            with open(r4_summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=r4_fields)
                writer.writeheader()
                for row in subject_records:
                    writer.writerow({
                        "target_subject": f"S{row['target_subject']}",
                        "variant": args.rapid_variant,
                        "best_acc": row["best_acc"],
                        "best_macro_f1": row["macro_f1_at_best"],
                        "best_epoch": row["best_epoch"],
                        "balanced_accuracy": row["balanced_accuracy"],
                        "recall_class0": row["recall_class0"],
                        "recall_class1": row["recall_class1"],
                        "final_acc": row["final_acc"],
                        "final_macro_f1": row["final_macro_f1"],
                        "total_params": row["total_params"],
                    })
                writer.writerow({
                    "target_subject": "Mean",
                    "variant": args.rapid_variant,
                    "best_acc": float(np.mean([row["best_acc"] for row in subject_records])),
                    "best_macro_f1": float(np.mean([row["macro_f1_at_best"] for row in subject_records])),
                    "best_epoch": "",
                    "balanced_accuracy": float(np.mean([row["balanced_accuracy"] for row in subject_records])),
                    "recall_class0": float(np.mean([row["recall_class0"] for row in subject_records])),
                    "recall_class1": float(np.mean([row["recall_class1"] for row in subject_records])),
                    "final_acc": float(np.mean([row["final_acc"] for row in subject_records])),
                    "final_macro_f1": float(np.mean([row["final_macro_f1"] for row in subject_records])),
                    "total_params": subject_records[0]["total_params"],
                })
            summary["r4_summary_csv"] = r4_summary_path
        _write_json(summary_json_path, summary)
        if args.cast_level1:
            rapid_summary_path, rapid_summary_csv = write_cast_level1_summary(output_dir)
            comparison_path = rapid_summary_csv
        else:
            comparison_path, rapid_summary_path, rapid_summary_csv = write_rapid_pilot3_summary(output_dir)
        _log(f"\nOverall best Acc: {all_acc.mean() * 100:.2f}% +/- {all_acc.std() * 100:.2f}%", training_log_path)
        _log(f"Results saved to {subject_csv}", training_log_path)
        _log(f"Summary saved to {summary_json_path}", training_log_path)
        _log(f"Rapid comparison updated at {comparison_path}", training_log_path)
        _log(f"Rapid aggregate summary updated at {rapid_summary_path}", training_log_path)
        _log(f"Rapid aggregate CSV updated at {rapid_summary_csv}", training_log_path)
    elif args.pilot_mode:
        _write_json(summary_json_path, summary)
        _append_csv_row(summary_path, list(summary.keys()), summary)
        comparison_path, delta_path = write_pilot_diagnostic_comparison(output_dir, pilot_config)
        _log(f"\nOverall best Acc: {all_acc.mean() * 100:.2f}% +/- {all_acc.std() * 100:.2f}%", training_log_path)
        _log(f"Results saved to {subject_csv}", training_log_path)
        _log(f"Summary saved to {summary_json_path}", training_log_path)
        _log(f"Comparison updated at {comparison_path}", training_log_path)
        _log(f"Delta summary updated at {delta_path}", training_log_path)
    else:
        _append_csv_row(summary_path, list(summary.keys()), summary)
        print(f"\nOverall best Acc: {all_acc.mean() * 100:.2f}% +/- {all_acc.std() * 100:.2f}%")
        print(f"Results saved to {subject_csv}")
        print(f"Summary appended to {summary_path}")


if __name__ == "__main__":
    parser = get_args_parser()
    parser.set_defaults(epochs=None, batch_size=None, lr=None, sample_length=None, stride=None, LM="clip")
    parser.add_argument("--dataset_name", type=str, required=True, choices=sorted(DATASET_DEFAULTS))
    parser.add_argument("--build_pilot", action="store_true", default=False,
                        help="Build the fixed DREAMER Pilot-5 protocol and exit.")
    parser.add_argument("--pilot_mode", action="store_true", default=False,
                        help="Use target subjects from --pilot_config.")
    parser.add_argument("--pilot_config", type=str, default="results/pilot_protocol/dreamer_pilot5.json")
    parser.add_argument("--pilot_target_limit", type=int, default=None,
                        help="Limit pilot targets for smoke testing only.")
    parser.add_argument("--pilot_output_name", type=str, default=None,
                        help="Override pilot output folder name, mainly for smoke tests.")
    parser.add_argument("--experiment_variant", type=str, default="standard",
                        choices=sorted(EXPERIMENT_VARIANTS),
                        help="P0-P3 diagnostic variant. standard keeps explicit CLI parameters.")
    parser.add_argument("--rapid_pilot3", action="store_true", default=False,
                        help="Run the fixed-source DREAMER Rapid Pilot-3 protocol.")
    parser.add_argument("--rapid_variant", type=str, default="r0_current_clip",
                        choices=sorted(RAPID_VARIANTS))
    parser.add_argument("--rapid_smoke_test", action="store_true", default=False,
                        help="Use the isolated Rapid smoke output directory; defaults to S12 x 2 epochs.")
    parser.add_argument(
        "--cast_balanced_screening",
        action="store_true",
        default=False,
        help="Use a deterministic class-balanced subset of each source; target evaluation stays full-size.",
    )
    parser.add_argument(
        "--cast_samples_per_class",
        type=int,
        default=128,
        help="Maximum samples per class and source in CAST screening mode.",
    )
    parser.add_argument(
        "--cast_pretrained_checkpoint",
        type=str,
        default=None,
        help="Checkpoint path; may contain {target} for the one-based target subject ID.",
    )
    parser.add_argument(
        "--cast_freeze_loaded",
        action="store_true",
        default=False,
        help="Freeze all name-and-shape matched tensors loaded from the CAST checkpoint.",
    )
    parser.add_argument(
        "--cast_save_final_checkpoint",
        action="store_true",
        default=False,
        help="Save the fixed final-epoch state for later leakage-safe module-isolation runs.",
    )
    parser.add_argument(
        "--fixed_source_config",
        type=str,
        default="results/dreamer_rapid_pilot3/fixed_sources.json",
    )
    parser.add_argument("--lambda_max", type=float, default=None)
    parser.add_argument("--lambda_min", type=float, default=None)
    parser.add_argument("--mmd_type", type=str, default="resgca",
                        choices=["none", "conditional_simple", "marginal", "class_aware", "sca", "resgca", "hut"])
    parser.add_argument("--mmd_schedule", type=str, default="warmup_cosine_decay",
                        choices=["monotonic", "warmup_hold", "warmup_decay", "warmup_cosine_decay"])
    parser.add_argument("--mmd_start_ratio", type=float, default=0.0)
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
    parser.add_argument("--uot_epsilon", type=float, default=0.05)
    parser.add_argument("--uot_tau_s", type=float, default=1.0)
    parser.add_argument("--uot_tau_t", type=float, default=0.5)
    parser.add_argument("--uot_route_tau", type=float, default=0.2)
    parser.add_argument("--uot_n_iter", type=int, default=20)
    parser.add_argument("--hut_geo_cost_weight", type=float, default=0.2)
    parser.add_argument("--hut_agreement_tau", type=float, default=0.5)
    parser.add_argument("--hut_use_agreement_mass", action="store_true", default=True)
    parser.add_argument("--no_hut_agreement_mass", action="store_false", dest="hut_use_agreement_mass")
    parser.add_argument("--hut_use_geometry_cost", action="store_true", default=True)
    parser.add_argument("--no_hut_geometry_cost", action="store_false", dest="hut_use_geometry_cost")
    parser.add_argument("--proto_tau", type=float, default=0.07)
    parser.add_argument("--fusion_tau", type=float, default=0.5)
    parser.add_argument("--eval_classifier", type=str, default="text",
                        choices=["text", "centroid", "hybrid", "senior_feature"])
    parser.add_argument("--reliability_fusion", action="store_true", default=False)
    parser.add_argument("--centroid_blend", type=float, default=0.5)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--shrinkage", type=float, default=0.1)
    parser.add_argument("--spd_eps", type=float, default=1e-5)
    parser.add_argument("--geometry_batch_size", type=int, default=None)
    parser.add_argument("--st_dim", type=int, default=128)
    parser.add_argument("--graph_dim", type=int, default=64)
    parser.add_argument("--adapter_bottleneck", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--graph_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--backbone_lr", type=float, default=3e-4)
    parser.add_argument("--use_param_groups", action="store_true", default=False)
    parser.add_argument("--selection_warmup_proto_only", action="store_true", default=False)
    parser.add_argument("--log_interval", type=int, default=1)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--source_selection", type=str, default="sparse_reliability",
                        choices=["none", "sparse_reliability", "fixed"])
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
    parser.add_argument("--session_ids", type=str, nargs="+", default=None)
    parser.add_argument("--dreamer_labeltype", type=str, default="val", choices=["val", "aro"])
    parser.add_argument("--dreamer_ea", action="store_true", default=True)
    parser.add_argument("--no_dreamer_ea", action="store_false", dest="dreamer_ea")
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--subject_ids", type=str, nargs="+", default=None)
    parser.add_argument("--random_subject_count", type=int, default=None)
    parser.add_argument("--subject_seed", type=int, default=42)
    parser.add_argument("--max_trials", type=int, default=None)
    parser.add_argument("--max_samples_per_subject", type=int, default=None)
    parser.add_argument("--sample_subset", type=str, default="stratified", choices=["stratified", "head"])
    run(parser.parse_args())
