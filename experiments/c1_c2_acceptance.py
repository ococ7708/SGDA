"""Task-06 acceptance/smoke utility. This never launches formal EEG training."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from models.context_affective_metric import AffectiveMetricHead, random_basis
from models.pairwise_utility_router import PairwiseUtilityRouter, build_router_features
from utils.pairwise_consistency import utility_labels
from utils.seediv_c1_c2_protocol import DEV_TARGETS, canonical_hash, load_config, oof_plan, source_roles


def run(config_path, output):
    cfg = load_config(config_path)
    torch.manual_seed(42)
    n, k, c, d, context_dim = 12, 3, 4, 32, 16
    prototypes, z, context = torch.randn(c, d), torch.randn(n, d), torch.randn(n, context_dim)
    basis = random_basis(d, 2, 20260924)
    variants = {}
    for variant in ("E0", "E1", "E2", "E3", "E4", "E5"):
        head = AffectiveMetricHead(variant, context_dim, prototypes, basis)
        result = head(context, z)
        loss = torch.nn.functional.cross_entropy(result.logits, torch.arange(n) % c) + result.regularization
        if loss.requires_grad:  # E0 is intentionally parameter-free.
            loss.backward()
        variants[variant] = {"shape": list(result.logits.shape), "finite": bool(torch.isfinite(result.logits).all())}
    reference, experts = torch.randn(n, c), torch.randn(n, k, c)
    pair_features, distances = torch.randn(6, 3), torch.rand(n, k)
    features = build_router_features(reference, experts, pair_features, distances)
    router = PairwiseUtilityRouter(features.size(-1))
    router.standardizer.fit(features)
    prediction = router(features)
    labels = utility_labels(reference, experts, torch.arange(n) % c)
    router.loss(prediction, labels).backward()
    report = {
        "status": "PASS", "scope": "mathematical and synthetic first-fold smoke; no EEG training",
        "config": str(config_path), "config_hash": canonical_hash(cfg), "c1": variants,
        "c2": {"feature_shape": list(features.shape), "utility_shape": list(labels.shape)},
        "development": {str(t): {"roles": source_roles(t), "oof_T": oof_plan(source_roles(t)["T"])} for t in DEV_TARGETS},
    }
    path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/seediv_c1_full45.json")
    parser.add_argument("--output", default="artifacts/c1_c2_acceptance.json")
    args = parser.parse_args()
    run(args.config, args.output)
