"""Evaluate a C1 E2 checkpoint normally, with H=0, or with shuffled context only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_utils.constants.path_mapper import path_mapper
from data_utils.text_to_vector import label_to_vector
from experiments.seediv_c1_end_to_end import _diagnostics, _load_reviewed_va, semantic_basis
from models.context_affective_metric import MultiSourceAffectiveMetric, prototype_basis
from models.geosem_stda import GeoSemSTDA
from utils.mix_utils import flatten_trials, setup_seed, zscore_subject_wise
from data_utils.load_data import get_data
from config.setting import Setting


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def shuffle_context_within_subject(context: torch.Tensor, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(len(context), generator=generator)
    return context.index_select(0, permutation.to(context.device)), permutation


def zero_metric_generator(mechanism: MultiSourceAffectiveMetric) -> None:
    for head in mechanism._branch_heads():
        if head.metric_generator is None:
            raise ValueError("H=0 mode is defined only for metric-generating variants E1/E2/E5")
        final = head.metric_generator.net[-1]
        torch.nn.init.zeros_(final.weight)
        torch.nn.init.zeros_(final.bias)


def _load_data(session: int, seed: int, sample_length: int):
    setting = Setting(
        dataset="seediv_de_lds", dataset_path=path_mapper["seediv_de_lds"],
        pass_band=[0.3, 50], extract_bands=None, time_window=1, overlap=0,
        sample_length=sample_length, stride=1, seed=seed, feature_type="de_lds",
        only_seg=False, experiment_mode="subject-independent", sessions=[1, 2, 3], onehot=False,
    )
    data, labels, channels, bands, classes = get_data(setting)
    data, labels = flatten_trials(data, labels)
    data = zscore_subject_wise(data)
    return data, labels, channels, bands, classes


@torch.no_grad()
def evaluate(checkpoint_path: Path, mode: str, shuffle_seed: int, device_name: str, batch_size: int) -> dict:
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    manifest = state["manifest"]
    cfg = manifest["effective_config"]
    session, target, seed = int(manifest["session"]), int(manifest["target"]), int(manifest["seed"])
    variant = str(manifest["variant"])
    if variant not in ("E1", "E2", "E5"):
        raise ValueError(f"checkpoint variant {variant} has no context-dependent H metric")
    setup_seed(seed)
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    data, labels, channels, bands, classes = _load_data(session, seed, int(cfg["sample_length"]))
    text_dim, vectors = label_to_vector("seediv_de_lds", "clip", device=device)
    prototypes = torch.tensor(np.asarray([vectors[i] for i in sorted(vectors)], np.float32), device=device)
    va_payload, va = _load_reviewed_va(ROOT / cfg["va_config"], ["neutral", "sad", "fear", "happy"], device)
    basis = prototype_basis(prototypes, int(cfg["rank"])) if variant == "E5" else semantic_basis(prototypes, va)
    source_ids = [int(x) - 1 for x in manifest["source_ids"]]

    model = GeoSemSTDA(
        len(source_ids), channels, bands, int(cfg["st_dim"]), int(cfg["graph_dim"]),
        int(cfg["adapter_bottleneck"]), text_dim, int(cfg["graph_heads"]),
        graph_heads=int(cfg["graph_heads"]), topk=int(cfg["topk"]), dropout=float(cfg["dropout"]),
        sample_length=int(cfg["sample_length"]), representation_mode=cfg["representation_mode"],
        classifier_type=cfg["classifier_type"], num_classes=int(cfg["num_classes"]),
        cast_variant=cfg["backbone"],
    ).to(device)
    mechanism = MultiSourceAffectiveMetric(
        len(source_ids), head_sharing=cfg["head_sharing"], variant=variant,
        context_dim=int(cfg["st_dim"]), prototypes=prototypes, basis=basis,
        rank=int(cfg["rank"]), hidden_dim=int(cfg["metric_hidden_dim"]),
        gamma=float(cfg["gamma"]), bound_eps=float(cfg["metric_epsilon"]),
        tau=float(cfg["tau"]), lambda_regularization=float(cfg["lambda_H"]),
    ).to(device)
    model.load_state_dict(state["model"])
    mechanism.load_state_dict(state["mechanism"])
    model.eval(); mechanism.eval()
    if mode == "zero_h":
        zero_metric_generator(mechanism)

    x = np.asarray(data[session - 1][target - 1], dtype=np.float32)
    y = np.asarray(labels[session - 1][target - 1], dtype=np.int64).reshape(-1)
    loader = DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)), batch_size=batch_size, shuffle=False)
    centroids = state["centroids"].to(device)
    context_bank = []
    if mode == "context_shuffle":
        for xb, _ in loader:
            xb = xb.to(device)
            _, _, _, h, _, _ = model([], [], xb, torch.zeros((len(xb), 1, 1), device=device), return_features=True)
            context_bank.append(h.detach().cpu())
        context_bank = torch.cat(context_bank, dim=0)
        shuffled_context_bank, permutation = shuffle_context_within_subject(context_bank, shuffle_seed)
    else:
        shuffled_context_bank, permutation = None, None

    all_labels, all_pred, all_base, all_logits = [], [], [], []
    h_spectral, h_ratio = [], []
    cursor = 0
    for xb, yb in loader:
        xb = xb.to(device)
        _, z_all, _, h, _, _ = model([], [], xb, torch.zeros((len(xb), 1, 1), device=device), return_features=True)
        stack = torch.stack(z_all)
        weights = F.softmax(-torch.norm(stack - centroids[:, None], dim=-1) / float(cfg["fusion_tau"]), dim=0)
        if mode == "context_shuffle":
            metric_context = shuffled_context_bank[cursor:cursor + len(xb)].to(device)
        else:
            metric_context = h
        output = mechanism.fused_output(metric_context, z_all, weights, fusion_mode=cfg["fusion_mode"])
        base_branch = torch.stack([
            F.normalize(z, dim=-1) @ head.prototypes.T / head.tau
            for head, z in zip(mechanism._branch_heads(), z_all)
        ])
        base_fused = (weights[..., None] * base_branch).sum(0)
        all_labels.append(yb.numpy())
        all_pred.append(output.logits.argmax(-1).cpu().numpy())
        all_base.append(base_fused.argmax(-1).cpu().numpy())
        all_logits.append(output.logits.cpu())
        h_spectral.append(float(output.diagnostics["metric_H_spectral_norm_max"].cpu()))
        h_ratio.append(float(output.diagnostics["metric_H_spectral_ratio_max"].cpu()))
        cursor += len(xb)

    yt, pred, base_pred, logits = np.concatenate(all_labels), np.concatenate(all_pred), np.concatenate(all_base), torch.cat(all_logits)
    result = _diagnostics(yt, pred, logits, classes)
    corrected = (base_pred != yt) & (pred == yt)
    harmed = (base_pred == yt) & (pred != yt)
    result.update({
        "mode": mode,
        "shuffle_seed": int(shuffle_seed) if mode == "context_shuffle" else None,
        "shuffle_permutation_sha256": hashlib.sha256(permutation.numpy().tobytes()).hexdigest() if permutation is not None else None,
        "context_only_shuffle": mode == "context_shuffle",
        "classification_embeddings_unchanged": True,
        "fusion_weights_unchanged": True,
        "final_fused_prediction_flip_rate": float((pred != base_pred).mean()),
        "final_fused_corrected_errors": int(corrected.sum()),
        "final_fused_harmed_correct": int(harmed.sum()),
        "final_fused_net_correction_rate": float((corrected.sum() - harmed.sum()) / max(len(yt), 1)),
        "metric_H_spectral_norm_max": float(max(h_spectral, default=0.0)),
        "metric_H_spectral_ratio_max": float(max(h_ratio, default=0.0)),
        "gamma": float(cfg["gamma"]),
    })
    result.update({
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "effective_config_hash": manifest["effective_config_hash"],
        "session": session, "target": target, "seed": seed, "variant": variant,
        "device": str(device),
        "class_names": ["neutral", "sad", "fear", "happy"],
        "confusion_matrix": result["confusion_matrix"],
        "va_provenance": {"source": va_payload["source"], "scale": va_payload["scale"]},
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--mode", required=True, choices=("normal", "zero_h", "context_shuffle"))
    parser.add_argument("--shuffle-seed", type=int, default=20260928)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--output")
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint)
    if not checkpoint.is_absolute():
        checkpoint = ROOT / checkpoint
    payload = evaluate(checkpoint, args.mode, args.shuffle_seed, args.device, args.batch_size)
    out = Path(args.output) if args.output else checkpoint.parent / f"checkpoint_eval_{args.mode}.json"
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
