"""Pairwise C2 actions, utility labels and consistent multi-class fusion."""

from __future__ import annotations

import itertools

import torch
import torch.nn.functional as F


def class_pairs(num_classes: int) -> tuple[tuple[int, int], ...]:
    return tuple(itertools.combinations(range(num_classes), 2))


def center_logits(logits: torch.Tensor) -> torch.Tensor:
    return logits - logits.mean(dim=-1, keepdim=True)


def pairwise_residuals(
    reference_logits: torch.Tensor, expert_logits: torch.Tensor, clip_limit: float = 2.0,
) -> torch.Tensor:
    """Return r[k,a<b] = clip(d_expert-d_reference) as [N,K,E]."""
    reference = center_logits(reference_logits)
    expert = center_logits(expert_logits)
    pairs = class_pairs(reference.size(-1))
    ref_margin = torch.stack([reference[:, a] - reference[:, b] for a, b in pairs], dim=-1)
    expert_margin = torch.stack([expert[:, :, a] - expert[:, :, b] for a, b in pairs], dim=-1)
    return (expert_margin - ref_margin[:, None]).clamp(-float(clip_limit), float(clip_limit))


def single_action_logits(
    reference_logits: torch.Tensor, residual: torch.Tensor, pair: tuple[int, int], alpha: float = 0.25,
) -> torch.Tensor:
    a, b = pair
    result = center_logits(reference_logits).clone()
    adjustment = float(alpha) * residual / 2.0
    result[:, a] += adjustment
    result[:, b] -= adjustment
    return result


def utility_labels(
    reference_logits: torch.Tensor, expert_logits: torch.Tensor, labels: torch.Tensor,
    *, alpha: float = 0.25, clip_limit: float = 2.0,
) -> torch.Tensor:
    """Full multi-class CE improvement for every source and pair: [N,K,E]."""
    pairs = class_pairs(reference_logits.size(-1))
    residual = pairwise_residuals(reference_logits, expert_logits, clip_limit)
    base_ce = F.cross_entropy(reference_logits, labels, reduction="none")
    values = []
    for source in range(expert_logits.size(1)):
        source_values = []
        for edge, pair in enumerate(pairs):
            revised = single_action_logits(reference_logits, residual[:, source, edge], pair, alpha)
            source_values.append(base_ce - F.cross_entropy(revised, labels, reduction="none"))
        values.append(torch.stack(source_values, dim=-1))
    return torch.stack(values, dim=1)


def apply_best_single_action(
    reference_logits: torch.Tensor, residuals: torch.Tensor, predicted_utility: torch.Tensor,
    alpha: float = 0.25,
) -> tuple[torch.Tensor, torch.Tensor]:
    """U1: execute the maximum positive predicted-utility action, else null."""
    n, _, edges = predicted_utility.shape
    pairs = class_pairs(reference_logits.size(-1))
    if edges != len(pairs):
        raise ValueError("predicted utility has the wrong pair dimension")
    flattened = predicted_utility.reshape(n, -1)
    best_value, best_index = flattened.max(dim=-1)
    source_index, edge_index = best_index // edges, best_index % edges
    output = center_logits(reference_logits).clone()
    active = best_value > 0
    for edge, (a, b) in enumerate(pairs):
        selected = active & (edge_index == edge)
        if selected.any():
            selected_residual = residuals[selected, source_index[selected], edge]
            delta = float(alpha) * selected_residual / 2.0
            output[selected, a] += delta
            output[selected, b] -= delta
    action = torch.where(active, best_index + 1, torch.zeros_like(best_index))
    return output, action


def solve_weighted_pairwise(
    reference_logits: torch.Tensor, target_margins: torch.Tensor, weights: torch.Tensor,
    ridge: float = 1.0,
) -> torch.Tensor:
    """U2 weighted graph-Laplacian solve with exact all-zero no-op."""
    reference = center_logits(reference_logits)
    n, classes = reference.shape
    pairs = class_pairs(classes)
    if target_margins.shape != weights.shape or target_margins.shape != (n, len(pairs)):
        raise ValueError("target_margins and weights must be [N,C*(C-1)/2]")
    output = reference.clone()
    active_rows = weights.sum(-1) > 0
    identity = torch.eye(classes, device=reference.device, dtype=reference.dtype)
    ones = torch.ones(classes, 1, device=reference.device, dtype=reference.dtype)
    for row in torch.nonzero(active_rows, as_tuple=False).flatten().tolist():
        laplacian = float(ridge) * identity
        rhs = float(ridge) * reference[row]
        for edge, (a, b) in enumerate(pairs):
            weight = weights[row, edge]
            vector = torch.zeros(classes, device=reference.device, dtype=reference.dtype)
            vector[a], vector[b] = 1.0, -1.0
            laplacian = laplacian + weight * torch.outer(vector, vector)
            rhs = rhs + weight * target_margins[row, edge] * vector
        kkt = torch.cat((
            torch.cat((laplacian, ones), dim=1),
            torch.cat((ones.T, torch.zeros(1, 1, device=reference.device, dtype=reference.dtype)), dim=1),
        ), dim=0)
        solution = torch.linalg.solve(kkt, torch.cat((rhs, rhs.new_zeros(1))))
        output[row] = solution[:classes]
    return output


def apply_multi_pair_actions(
    reference_logits: torch.Tensor, residuals: torch.Tensor, predicted_utility: torch.Tensor,
    *, alpha: float = 0.25, utility_scale: float = 1.0, ridge: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """U2: best source per pair, positive gates, consistent multiclass logits."""
    best_utility, best_source = predicted_utility.max(dim=1)
    gates = (best_utility / max(float(utility_scale), 1e-4)).clamp(0, 1)
    gates = torch.where(best_utility > 0, gates, torch.zeros_like(gates))
    chosen = residuals.gather(1, best_source[:, None, :]).squeeze(1)
    reference = center_logits(reference_logits)
    pairs = class_pairs(reference.size(-1))
    base_margin = torch.stack([reference[:, a] - reference[:, b] for a, b in pairs], dim=-1)
    target_margin = base_margin + gates * float(alpha) * chosen
    return solve_weighted_pairwise(reference, target_margin, gates, ridge), gates, best_source
