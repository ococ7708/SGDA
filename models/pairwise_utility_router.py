"""Shared C2 utility predictor without source-ID embeddings."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from utils.pairwise_consistency import class_pairs


def build_router_features(
    reference_logits: torch.Tensor,
    expert_logits: torch.Tensor,
    prototype_pair_features: torch.Tensor,
    source_distances: torch.Tensor,
) -> torch.Tensor:
    """Build legal per-sample/source/pair features [N,K,E,F]."""
    reference_probability = reference_logits.softmax(-1)
    expert_probability = expert_logits.softmax(-1)
    ref_entropy = -(reference_probability * reference_probability.clamp_min(1e-12).log()).sum(-1)
    expert_entropy = -(expert_probability * expert_probability.clamp_min(1e-12).log()).sum(-1)
    pairs = class_pairs(reference_logits.size(-1))
    feature_rows = []
    for edge, (a, b) in enumerate(pairs):
        ref_pair = torch.stack((reference_probability[:, a], reference_probability[:, b]), -1)
        expert_pair = torch.stack((expert_probability[:, :, a], expert_probability[:, :, b]), -1)
        ref_margin = reference_logits[:, a] - reference_logits[:, b]
        expert_margin = expert_logits[:, :, a] - expert_logits[:, :, b]
        base = torch.cat((
            ref_pair[:, None].expand(-1, expert_logits.size(1), -1), expert_pair,
            ref_entropy[:, None, None].expand(-1, expert_logits.size(1), 1),
            expert_entropy[:, :, None], ref_margin[:, None, None].expand(-1, expert_logits.size(1), 1),
            expert_margin[:, :, None], (expert_margin - ref_margin[:, None])[:, :, None],
            source_distances[:, :, None],
            prototype_pair_features[edge][None, None].expand(reference_logits.size(0), expert_logits.size(1), -1),
        ), dim=-1)
        feature_rows.append(base)
    return torch.stack(feature_rows, dim=2)


class FeatureStandardizer(nn.Module):
    """Router-train-only standardization persisted in the checkpoint."""
    def __init__(self, feature_dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(feature_dim))
        self.register_buffer("scale", torch.ones(feature_dim))
        self.register_buffer("fitted", torch.tensor(False))

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> None:
        flat = features.reshape(-1, features.size(-1))
        self.mean.copy_(flat.mean(0))
        self.scale.copy_(flat.std(0, unbiased=False).clamp_min(1e-6))
        self.fitted.fill_(True)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not bool(self.fitted):
            raise RuntimeError("FeatureStandardizer must be fitted on router training subjects")
        return (features - self.mean) / self.scale


class PairwiseUtilityRouter(nn.Module):
    def __init__(self, feature_dim: int, hidden=(64, 32)):
        super().__init__()
        self.standardizer = FeatureStandardizer(feature_dim)
        layers: list[nn.Module] = []
        current = feature_dim
        for width in hidden:
            layers.extend((nn.Linear(current, width), nn.ReLU()))
            current = width
        layers.append(nn.Linear(current, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        utility = self.network(self.standardizer(features)).squeeze(-1)
        if mask is not None:
            utility = utility.masked_fill(~mask, torch.finfo(utility.dtype).min)
        return utility

    @staticmethod
    def loss(predicted: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is not None:
            predicted, target = predicted[mask], target[mask]
        return F.huber_loss(predicted, target)
