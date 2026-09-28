"""Causal per-trial C2 history and probability smoothing utilities."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from utils.pairwise_consistency import class_pairs


HISTORY_VALUE_NAMES = (
    "past_ema_reference_margin",
    "past_ema_expert_margin",
    "past_ema_residual_margin",
    "past_residual_margin_std",
    "current_minus_past_residual_margin",
    "history_mask",
)


class CausalMarginHistory:
    """Stateful strict-past EMA; safe to carry across inference chunks."""

    def __init__(self, alpha: float = 0.8):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("history alpha must be in [0,1)")
        self.alpha = float(alpha)
        self._states = {}

    def reset(self) -> None:
        self._states.clear()

    def update(self, reference_margin, expert_margin, residual_margin, key, window_start: int,
               *, matched_current: bool = False) -> np.ndarray:
        m0 = np.asarray(reference_margin, dtype=np.float64)
        mk = np.asarray(expert_margin, dtype=np.float64)
        dk = np.asarray(residual_margin, dtype=np.float64)
        if mk.ndim != 2 or dk.shape != mk.shape or m0.shape != (mk.shape[1],):
            raise ValueError("margin state expects m0 [E], mk/dk [K,E]")
        state = self._states.get(tuple(map(int, key)))
        initialized = state is not None
        if state is not None and int(window_start) <= state["last_window_start"]:
            raise ValueError("history windows must arrive in strictly increasing time order")
        if initialized:
            mean_m0, mean_mk, mean_dk, second_dk = state["mean_m0"], state["mean_mk"], state["mean_dk"], state["second_dk"]
            std_dk = np.sqrt(np.maximum(second_dk - mean_dk * mean_dk, 0.0))
            features = [
                np.broadcast_to(mean_m0[None, :], mk.shape),
                mean_mk,
                mean_dk,
                std_dk,
                dk - mean_dk,
            ]
        else:
            mean_m0 = mean_mk = mean_dk = second_dk = None
            features = [np.zeros_like(mk) for _ in range(5)]
        if matched_current:
            features = [
                np.broadcast_to(m0[None, :], mk.shape),
                mk,
                dk,
                np.zeros_like(mk),
                np.zeros_like(mk),
            ]
        row = np.zeros((*mk.shape, len(HISTORY_VALUE_NAMES)), dtype=np.float32)
        for index, value in enumerate(features):
            row[:, :, index] = value
        row[:, :, 5] = float(initialized)

        if initialized:
            alpha = self.alpha
            new_mean_m0 = alpha * mean_m0 + (1.0 - alpha) * m0
            new_mean_mk = alpha * mean_mk + (1.0 - alpha) * mk
            new_mean_dk = alpha * mean_dk + (1.0 - alpha) * dk
            new_second_dk = alpha * second_dk + (1.0 - alpha) * np.square(dk)
        else:
            new_mean_m0, new_mean_mk, new_mean_dk, new_second_dk = m0.copy(), mk.copy(), dk.copy(), np.square(dk)
        self._states[tuple(map(int, key))] = {
            "mean_m0": new_mean_m0,
            "mean_mk": new_mean_mk,
            "mean_dk": new_mean_dk,
            "second_dk": new_second_dk,
            "last_window_start": int(window_start),
        }
        return row


class CausalProbabilityEMA:
    """Stateful probability EMA that is invariant to chunk boundaries."""

    def __init__(self, alpha: float = 0.8):
        if not 0.0 <= alpha < 1.0:
            raise ValueError("EMA alpha must be in [0,1)")
        self.alpha = float(alpha)
        self._states = {}

    def reset(self) -> None:
        self._states.clear()

    def update(self, probability, key, window_start: int) -> np.ndarray:
        current = np.asarray(probability, dtype=np.float64)
        state = self._states.get(tuple(map(int, key)))
        if state is None:
            value = current
        else:
            if int(window_start) <= state["last_window_start"]:
                raise ValueError("probability EMA windows must arrive in strictly increasing time order")
            value = self.alpha * state["probability"] + (1.0 - self.alpha) * current
        value = value / max(float(value.sum()), 1e-15)
        self._states[tuple(map(int, key))] = {"probability": value.copy(), "last_window_start": int(window_start)}
        return value.astype(np.float32)


def _stable_groups(subject_ids, session_ids, trial_ids, window_starts):
    subject_ids = np.asarray(subject_ids, dtype=np.int32)
    session_ids = np.asarray(session_ids, dtype=np.int16)
    trial_ids = np.asarray(trial_ids, dtype=np.int16)
    window_starts = np.asarray(window_starts, dtype=np.int32)
    n = len(subject_ids)
    if any(len(x) != n for x in (session_ids, trial_ids, window_starts)):
        raise ValueError("history metadata columns must have matching lengths")
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, key in enumerate(zip(subject_ids, session_ids, trial_ids)):
        groups[tuple(map(int, key))].append(index)
    ordered_groups = []
    for key in sorted(groups):
        indices = sorted(groups[key], key=lambda i: int(window_starts[i]))
        starts = window_starts[indices]
        if len(np.unique(starts)) != len(starts):
            raise ValueError(f"duplicate window start in history sequence {key}")
        ordered_groups.append((key, indices))
    return ordered_groups


def pair_margins(reference_logits: np.ndarray, expert_logits: np.ndarray):
    """Return uncorrected reference/expert margins as [N,K,E]."""
    ref = np.asarray(reference_logits, dtype=np.float64)
    exp = np.asarray(expert_logits, dtype=np.float64)
    if ref.ndim != 2 or exp.ndim != 3 or exp.shape[0] != ref.shape[0] or exp.shape[2] != ref.shape[1]:
        raise ValueError("expected reference [N,C] and experts [N,K,C]")
    pairs = class_pairs(ref.shape[1])
    m0 = np.stack([ref[:, a] - ref[:, b] for a, b in pairs], axis=-1)
    mk = np.stack([exp[:, :, a] - exp[:, :, b] for a, b in pairs], axis=-1)
    return m0, mk, mk - m0[:, None, :]


def build_history_features(
    reference_logits: np.ndarray,
    expert_logits: np.ndarray,
    subject_ids,
    session_ids,
    trial_ids,
    window_starts,
    *,
    alpha: float = 0.8,
    matched_current: bool = False,
) -> np.ndarray:
    """Build strict-past margin summaries [N,K,E,6].

    The state at t is read before the current evidence is incorporated. EMA
    state is initialized from the first window only after that window's
    feature row is emitted. Returned rows retain the caller's input order.
    """
    m0, mk, dk = pair_margins(reference_logits, expert_logits)
    n, k, edges = dk.shape
    out = np.zeros((n, k, edges, len(HISTORY_VALUE_NAMES)), dtype=np.float32)
    groups = _stable_groups(subject_ids, session_ids, trial_ids, window_starts)
    history = CausalMarginHistory(alpha)
    subject_ids = np.asarray(subject_ids)
    session_ids = np.asarray(session_ids)
    trial_ids = np.asarray(trial_ids)
    window_starts = np.asarray(window_starts)
    for _, indices in groups:
        for index in indices:
            key = (subject_ids[index], session_ids[index], trial_ids[index])
            out[index] = history.update(
                m0[index], mk[index], dk[index], key, int(window_starts[index]),
                matched_current=matched_current,
            )
    return out


def probability_ema(
    probabilities: np.ndarray,
    subject_ids,
    session_ids,
    trial_ids,
    window_starts,
    *,
    alpha: float = 0.8,
) -> np.ndarray:
    """Smooth current probabilities with strictly prior EMA state per trial."""
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 2:
        raise ValueError("probabilities must be [N,C]")
    result = np.empty_like(p)
    smoother = CausalProbabilityEMA(alpha)
    for _, indices in _stable_groups(subject_ids, session_ids, trial_ids, window_starts):
        for index in indices:
            key = (subject_ids[index], session_ids[index], trial_ids[index])
            result[index] = smoother.update(p[index], key, int(window_starts[index]))
    return result.astype(np.float32)


def append_history_features(current_features: torch.Tensor, history_features: np.ndarray | torch.Tensor) -> torch.Tensor:
    history = torch.as_tensor(history_features, dtype=current_features.dtype, device=current_features.device)
    if current_features.shape[:-1] != history.shape[:-1]:
        raise ValueError("current and history features must match over [N,K,E]")
    return torch.cat((current_features, history), dim=-1)
