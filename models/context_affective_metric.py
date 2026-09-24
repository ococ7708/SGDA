"""End-to-end Context-Adaptive Affective Metric (C1).

This module is independent from the historical frozen-feature probe.  It keeps
the prototype classifier differentiable all the way to the EEG encoder and
implements the six pre-registered E0--E5 controls from task specification 06.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


def _require_finite(name: str, value: torch.Tensor) -> None:
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")


def canonical_basis(vectors: torch.Tensor, rank: int) -> torch.Tensor:
    """Return deterministic right singular vectors with a sign convention."""
    if vectors.ndim != 2 or rank < 1:
        raise ValueError("vectors must be [n,d] and rank must be positive")
    _, singular_values, vh = torch.linalg.svd(vectors, full_matrices=False)
    if len(singular_values) < rank or singular_values[rank - 1] <= 1e-7:
        raise ValueError(f"basis rank {rank} is not reachable")
    basis = vh[:rank].T.contiguous()
    pivot = basis.abs().argmax(dim=0, keepdim=True)
    signs = torch.sign(basis.gather(0, pivot)).flatten()
    signs = torch.where(signs == 0, torch.ones_like(signs), signs)
    return basis * signs


def prototype_basis(prototypes: torch.Tensor, rank: int = 2) -> torch.Tensor:
    prototypes = F.normalize(prototypes, dim=-1)
    return canonical_basis(prototypes - prototypes.mean(0, keepdim=True), rank)


def random_basis(dimension: int, rank: int, seed: int, *, dtype=torch.float32) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    matrix = torch.randn(dimension, rank, generator=generator, dtype=dtype)
    basis = torch.linalg.qr(matrix, mode="reduced").Q
    pivot = basis.abs().argmax(dim=0, keepdim=True)
    signs = torch.sign(basis.gather(0, pivot)).flatten()
    return basis * torch.where(signs == 0, torch.ones_like(signs), signs)


def bound_symmetric(raw: torch.Tensor, gamma: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """H = gamma*S/(1+sqrt(sum(S^2)+eps^2)); spectral norm is < gamma."""
    symmetric = 0.5 * (raw + raw.transpose(-1, -2))
    denominator = 1.0 + torch.sqrt(symmetric.square().sum(dim=(-2, -1), keepdim=True) + eps**2)
    bounded = float(gamma) * symmetric / denominator
    _require_finite("bounded metric H", bounded)
    return bounded


def low_rank_metric_scores(
    z: torch.Tensor,
    prototypes: torch.Tensor,
    basis: torch.Tensor,
    h_matrix: torch.Tensor,
    tau: float = 0.07,
    clamp_eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Normalized M-cosine scores and class relations without constructing M."""
    z = F.normalize(z, dim=-1)
    prototypes = F.normalize(prototypes, dim=-1)
    zb, pb = z @ basis, prototypes @ basis
    zp = z @ prototypes.T + torch.einsum("nr,nrs,cs->nc", zb, h_matrix, pb)
    zz = z.square().sum(-1) + torch.einsum("nr,nrs,ns->n", zb, h_matrix, zb)
    pp = prototypes.square().sum(-1)[None] + torch.einsum("cr,nrs,cs->nc", pb, h_matrix, pb)
    clamped_zz, clamped_pp = zz.clamp_min(clamp_eps), pp.clamp_min(clamp_eps)
    logits = zp / torch.sqrt(clamped_zz[:, None] * clamped_pp) / float(tau)
    relation_num = prototypes @ prototypes.T + torch.einsum("ar,nrs,bs->nab", pb, h_matrix, pb)
    relation = relation_num / torch.sqrt(clamped_pp[:, :, None] * clamped_pp[:, None, :])
    diagnostics = {
        "z_norm_under_metric": zz,
        "prototype_norm_under_metric": pp,
        "denominator_clamp_count": (zz < clamp_eps).sum() + (pp < clamp_eps).sum(),
    }
    _require_finite("C1 logits", logits)
    _require_finite("C1 relation", relation)
    return logits, relation, diagnostics


class SymmetricMetricGenerator(nn.Module):
    def __init__(self, context_dim: int, rank: int = 2, hidden_dim: int = 64, static: bool = False):
        super().__init__()
        self.rank = int(rank)
        self.static = bool(static)
        n_upper = rank * (rank + 1) // 2
        self.context_norm = nn.LayerNorm(context_dim)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, n_upper)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if self.static:
            context = torch.zeros_like(context)
        values = self.net(self.context_norm(context))
        result = values.new_zeros((len(values), self.rank, self.rank))
        rows, columns = torch.triu_indices(self.rank, self.rank, device=values.device)
        result[:, rows, columns] = values
        result[:, columns, rows] = values
        return result


class DynamicLogitCorrection(nn.Module):
    def __init__(self, context_dim: int, num_classes: int, hidden_dim: int = 64, beta: float = 1.0):
        super().__init__()
        self.beta = float(beta)
        self.context_norm = nn.LayerNorm(context_dim)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_classes)
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, context: torch.Tensor, base_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = torch.tanh(self.net(self.context_norm(context)))
        delta = self.beta * (raw - raw.mean(dim=-1, keepdim=True))
        return base_logits + delta, delta


class CommonPlaneRotation(nn.Module):
    def __init__(self, context_dim: int, hidden_dim: int = 64, angle_cap: float = math.pi / 4):
        super().__init__()
        self.angle_cap = float(angle_cap)
        self.context_norm = nn.LayerNorm(context_dim)
        self.net = nn.Sequential(nn.Linear(context_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(
        self, context: torch.Tensor, z: torch.Tensor, prototypes: torch.Tensor,
        basis: torch.Tensor, tau: float = 0.07,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if basis.shape[1] != 2:
            raise ValueError("E4 requires a two-dimensional orthonormal basis")
        theta = self.angle_cap * torch.tanh(self.net(self.context_norm(context))).squeeze(-1)
        cosine, sine = theta.cos(), theta.sin()
        rotation = torch.stack((cosine, -sine, sine, cosine), dim=-1).reshape(-1, 2, 2)
        pb = prototypes @ basis
        rotated = prototypes[None] + torch.einsum(
            "dr,nrs,cs->ncd", basis, rotation - torch.eye(2, device=z.device, dtype=z.dtype), pb
        )
        logits = torch.einsum("nd,ncd->nc", F.normalize(z, dim=-1), F.normalize(rotated, dim=-1)) / tau
        return logits, rotated, theta


@dataclass
class C1Output:
    logits: torch.Tensor
    regularization: torch.Tensor
    relation: torch.Tensor | None
    diagnostics: dict[str, torch.Tensor]


class AffectiveMetricHead(nn.Module):
    """One branch head for E0--E5; E1/E2 use semantic B and E5 uses B_proto."""

    VALID_VARIANTS = ("E0", "E1", "E2", "E3", "E4", "E5")

    def __init__(
        self, variant: str, context_dim: int, prototypes: torch.Tensor, basis: torch.Tensor,
        *, rank: int = 2, hidden_dim: int = 64, gamma: float = 0.5,
        bound_eps: float = 1e-6, tau: float = 0.07, lambda_regularization: float = 1e-4,
    ):
        super().__init__()
        variant = variant.upper()
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"unknown C1 variant: {variant}")
        self.variant, self.rank = variant, int(rank)
        self.gamma, self.bound_eps, self.tau = float(gamma), float(bound_eps), float(tau)
        self.lambda_regularization = float(lambda_regularization)
        self.register_buffer("prototypes", F.normalize(prototypes.detach().clone(), dim=-1))
        self.register_buffer("basis", basis.detach().clone())
        self.metric_generator = None
        self.logit_head = None
        self.rotation_head = None
        if variant in ("E1", "E2", "E5"):
            self.metric_generator = SymmetricMetricGenerator(
                context_dim, rank=rank, hidden_dim=hidden_dim, static=variant == "E1"
            )
        elif variant == "E3":
            self.logit_head = DynamicLogitCorrection(context_dim, len(prototypes), hidden_dim)
        elif variant == "E4":
            self.rotation_head = CommonPlaneRotation(context_dim, hidden_dim)

    def forward(self, context: torch.Tensor, z: torch.Tensor) -> C1Output:
        z = F.normalize(z, dim=-1)
        base = z @ self.prototypes.T / self.tau
        zero = base.new_zeros(())
        if self.variant == "E0":
            return C1Output(base, zero, None, {
                "centered_logit_rms": zero, "prediction_flip_rate": zero,
            })
        if self.metric_generator is not None:
            raw = self.metric_generator(context)
            bounded = bound_symmetric(raw, self.gamma, self.bound_eps)
            logits, relation, low_rank_diag = low_rank_metric_scores(
                z, self.prototypes, self.basis, bounded, self.tau
            )
            regularization = self.lambda_regularization * bounded.square().sum(dim=(-2, -1)).mean()
            diagnostics = {
                **low_rank_diag,
                "H": bounded,
                "H_fro": torch.linalg.matrix_norm(bounded, ord="fro"),
                "H_sample_variance": bounded.var(dim=0, unbiased=False).mean(),
                "metric_eigenvalues": 1.0 + torch.linalg.eigvalsh(bounded),
            }
        elif self.logit_head is not None:
            logits, delta = self.logit_head(context, base)
            regularization = self.lambda_regularization * delta.square().mean()
            relation = None
            diagnostics = {"delta": delta}
        else:
            logits, rotated, theta = self.rotation_head(context, z, self.prototypes, self.basis, self.tau)
            regularization = self.lambda_regularization * theta.square().mean()
            relation = rotated @ rotated.transpose(-1, -2)
            diagnostics = {"theta": theta, "rotated_prototypes": rotated}
        delta = logits - base
        delta = delta - delta.mean(dim=-1, keepdim=True)
        diagnostics["centered_logit_rms"] = delta.square().mean().sqrt()
        diagnostics["prediction_flip_rate"] = (logits.argmax(-1) != base.argmax(-1)).float().mean()
        return C1Output(logits, regularization, relation, diagnostics)


class MultiSourceAffectiveMetric(nn.Module):
    """Independent per-source heads or one explicitly shared C1 head."""

    def __init__(self, n_sources: int, head_sharing: str = "independent", **head_kwargs):
        super().__init__()
        if head_sharing not in ("independent", "shared"):
            raise ValueError("head_sharing must be independent or shared")
        self.n_sources = int(n_sources)
        self.head_sharing = head_sharing
        count = self.n_sources if head_sharing == "independent" else 1
        self.heads = nn.ModuleList([AffectiveMetricHead(**head_kwargs) for _ in range(count)])

    def _branch_heads(self):
        return list(self.heads) if self.head_sharing == "independent" else [self.heads[0]] * self.n_sources

    def source_outputs(self, contexts: list[torch.Tensor], embeddings: list[torch.Tensor]) -> list[C1Output]:
        if len(contexts) != self.n_sources or len(embeddings) != self.n_sources:
            raise ValueError("source contexts/embeddings must match the number of C1 heads")
        return [head(context, embedding) for head, context, embedding in zip(self._branch_heads(), contexts, embeddings)]

    def fused_output(
        self, context: torch.Tensor, embeddings: list[torch.Tensor], source_weights: torch.Tensor,
        fusion_mode: str = "branch_logits",
    ) -> C1Output:
        """Fuse branch scores by default, matching how every branch head is trained.

        ``legacy_fused_embedding`` is retained only for an explicitly registered
        ablation; it sends the same fused embedding through all branch heads.
        """
        stack = torch.stack(embeddings, dim=0)  # [K,N,D]
        if source_weights.shape != stack.shape[:2]:
            raise ValueError("source_weights must be [K,N]")
        fused_z = F.normalize((source_weights[..., None] * stack).sum(dim=0), dim=-1)
        if fusion_mode == "branch_logits":
            outputs = [head(context, embedding) for head, embedding in zip(self._branch_heads(), embeddings)]
        elif fusion_mode == "shared_fused_embedding":
            if self.head_sharing != "shared":
                raise ValueError("shared_fused_embedding requires head_sharing=shared")
            outputs = [self.heads[0](context, fused_z)]
        elif fusion_mode == "legacy_fused_embedding":
            outputs = [head(context, fused_z) for head in self._branch_heads()]
        else:
            raise ValueError(f"unsupported C1 fusion_mode: {fusion_mode}")
        logits = outputs[0].logits if fusion_mode == "shared_fused_embedding" else (
            source_weights[..., None] * torch.stack([o.logits for o in outputs])
        ).sum(0)
        regularization = torch.stack([o.regularization for o in outputs]).mean()
        relations = [o.relation for o in outputs if o.relation is not None]
        if not relations:
            relation = None
        elif fusion_mode == "shared_fused_embedding":
            relation = relations[0]
        else:
            relation = (source_weights.T[..., None, None] * torch.stack(relations, dim=1)).sum(1)
        diagnostics = {
            "centered_logit_rms": torch.stack([o.diagnostics["centered_logit_rms"] for o in outputs]).mean(),
            "prediction_flip_rate": torch.stack([o.diagnostics["prediction_flip_rate"] for o in outputs]).mean(),
            "fused_embedding": fused_z,
            "fusion_mode": fusion_mode,
        }
        h_fro = [o.diagnostics["H_fro"].mean() for o in outputs if "H_fro" in o.diagnostics]
        h_var = [o.diagnostics["H_sample_variance"] for o in outputs if "H_sample_variance" in o.diagnostics]
        diagnostics["metric_H_fro_mean"] = logits.new_zeros(()) if not h_fro else torch.stack(h_fro).mean()
        diagnostics["metric_H_sample_variance"] = logits.new_zeros(()) if not h_var else torch.stack(h_var).mean()
        return C1Output(logits, regularization, relation, diagnostics)
