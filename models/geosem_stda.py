import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _matrix_log_spd(mat, eps=1e-5):
    """Batched Log-Euclidean matrix logarithm for symmetric positive matrices."""
    mat = 0.5 * (mat + mat.transpose(-1, -2))
    eigvals, eigvecs = torch.linalg.eigh(mat)
    eigvals = eigvals.clamp_min(eps)
    log_diag = torch.diag_embed(torch.log(eigvals))
    return eigvecs @ log_diag @ eigvecs.transpose(-1, -2)


def shrinkage_covariance(x, shrinkage=0.1, eps=1e-5):
    """
    Build shrinkage covariance from DE samples.

    Args:
        x: [B, L, C, F]
    Returns:
        cov: [B, C, C]
    """
    if x.dim() != 4:
        raise ValueError(f"Expected x shape [B,L,C,F], got {tuple(x.shape)}")

    bsz, steps, channels, bands = x.shape
    obs = steps * bands
    y = x.permute(0, 2, 1, 3).reshape(bsz, channels, obs)
    y = y - y.mean(dim=-1, keepdim=True)

    denom = max(obs - 1, 1)
    cov = y @ y.transpose(-1, -2) / denom
    trace = cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1, keepdim=True).unsqueeze(-1)
    eye = torch.eye(channels, device=x.device, dtype=x.dtype).expand(bsz, channels, channels)
    iso = trace / channels * eye
    cov = (1.0 - shrinkage) * cov + shrinkage * iso + eps * eye
    return 0.5 * (cov + cov.transpose(-1, -2))


def log_euclidean_reference(source_batches, shrinkage=0.1, eps=1e-5):
    """Compute G from source batches only, then return log(G)."""
    if not source_batches:
        raise ValueError("source_batches must not be empty when computing reference geometry")
    logs = []
    for x in source_batches:
        cov = shrinkage_covariance(x, shrinkage=shrinkage, eps=eps)
        logs.append(_matrix_log_spd(cov, eps=eps))
    return torch.cat(logs, dim=0).mean(dim=0)


def tangent_deviation(x, log_reference, shrinkage=0.1, eps=1e-5):
    cov = shrinkage_covariance(x, shrinkage=shrinkage, eps=eps)
    return _matrix_log_spd(cov, eps=eps) - log_reference.to(device=x.device, dtype=x.dtype)


def geometric_adjacency(r, topk=6, eps=1e-6):
    """
    Convert tangent deviation matrices to row-normalized geometric adjacency.

    Args:
        r: [B, C, C]
    Returns:
        adj: [B, C, C]
    """
    bsz, channels, _ = r.shape
    scores = r.abs()
    scores = scores.masked_fill(
        torch.eye(channels, device=r.device, dtype=torch.bool).unsqueeze(0),
        0.0
    )
    k = min(max(int(topk), 1), channels - 1)
    threshold = torch.topk(scores, k=k, dim=-1).values[..., -1:]
    adj = torch.where(scores >= threshold, scores, torch.zeros_like(scores))
    return 0.5 * (adj + adj.transpose(-1, -2))


def vech(x):
    channels = x.size(-1)
    rows, cols = torch.triu_indices(channels, channels, device=x.device)
    return x[..., rows, cols]


class AttentionAdjacency(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by graph heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)

    def forward(self, h):
        bsz, nodes, _ = h.shape
        q = self.q(h).view(bsz, nodes, self.heads, self.head_dim).transpose(1, 2)
        k = self.k(h).view(bsz, nodes, self.heads, self.head_dim).transpose(1, 2)
        logits = q @ k.transpose(-1, -2)
        logits = logits / math.sqrt(self.head_dim)
        adj = F.softmax(logits, dim=-1).mean(dim=1)
        return 0.5 * (adj + adj.transpose(-1, -2))


class GeometryGate(nn.Module):
    def __init__(self, channels, geo_dim):
        super().__init__()
        self.geo_proj = nn.Sequential(
            nn.Linear(channels * (channels + 1) // 2, geo_dim),
            nn.GELU(),
            nn.LayerNorm(geo_dim),
        )
        self.alpha = nn.Linear(geo_dim, 1)

    def forward(self, r):
        g = self.geo_proj(vech(r))
        alpha = torch.sigmoid(self.alpha(g)).view(-1, 1, 1)
        return g, alpha


class DynamicGraphConv(nn.Module):
    def __init__(self, num_bands, channels, graph_dim=64, topk=6, dropout=0.3, graph_heads=4):
        super().__init__()
        self.channels = channels
        self.topk = topk
        self.node_proj = nn.Linear(num_bands, graph_dim)
        self.learn_adj = AttentionAdjacency(graph_dim, heads=graph_heads)
        self.gate = GeometryGate(channels, graph_dim)
        self.gcn = nn.Linear(graph_dim, graph_dim, bias=False)
        self.pool_score = nn.Linear(graph_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, r):
        bsz, steps, channels, _ = x.shape
        if channels != self.channels:
            raise ValueError(f"Expected {self.channels} channels, got {channels}")

        a_geo = geometric_adjacency(r, topk=self.topk)
        geo_token, alpha = self.gate(r)

        spatial_steps = []
        eye = torch.eye(channels, device=x.device, dtype=x.dtype).unsqueeze(0)
        for t in range(steps):
            h0 = self.node_proj(x[:, t])
            a_learn = self.learn_adj(h0)
            adj = alpha * a_geo + (1.0 - alpha) * a_learn
            adj_hat = adj + eye
            deg = adj_hat.sum(dim=-1).clamp_min(1e-6)
            norm = deg.pow(-0.5).unsqueeze(-1) * adj_hat * deg.pow(-0.5).unsqueeze(-2)
            h = F.gelu(norm @ self.gcn(h0))
            h = self.dropout(h)
            weight = F.softmax(self.pool_score(h).squeeze(-1), dim=-1)
            spatial_steps.append((weight.unsqueeze(-1) * h).sum(dim=1))

        return torch.stack(spatial_steps, dim=1), geo_token, alpha.squeeze(-1).squeeze(-1)


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        num_bands,
        dim,
        heads=4,
        dropout=0.3,
        max_len=128,
        kernels=(3, 5, 7),
    ):
        super().__init__()
        self.max_len = max_len
        self.channel_proj = nn.Linear(num_bands, dim)
        self.channel_score = nn.Linear(dim, 1)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, dim))
        self.scale_convs = nn.ModuleList(
            nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2)
            for k in kernels
        )
        self.scale_score = nn.Linear(dim, 1)
        self.scale_norm = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        node = F.gelu(self.channel_proj(x))
        channel_weight = F.softmax(self.channel_score(node), dim=2)
        h = (channel_weight * node).sum(dim=2)
        steps = h.size(1)
        if steps <= self.max_len:
            pos = self.pos_embed[:, :steps]
        else:
            pos = F.interpolate(
                self.pos_embed.transpose(1, 2),
                size=steps,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        h = h + pos

        conv_in = h.transpose(1, 2)
        scale_stack = torch.stack(
            [F.gelu(conv(conv_in)).transpose(1, 2) for conv in self.scale_convs],
            dim=2,
        )
        scale_weight = F.softmax(self.scale_score(scale_stack).squeeze(-1), dim=2)
        multi_scale = (scale_weight.unsqueeze(-1) * scale_stack).sum(dim=2)
        h = self.scale_norm(h + self.dropout(multi_scale))
        ctx, _ = self.self_attn(h, h, h, need_weights=False)
        h = self.norm1(h + self.dropout(ctx))
        return self.norm2(h + self.ffn(h))


class CrossAttentionBlock(nn.Module):
    def __init__(self, dim=128, heads=4, dropout=0.3):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, temporal, spatial):
        ctx, _ = self.attn(query=temporal, key=spatial, value=spatial, need_weights=False)
        h = self.norm1(temporal + ctx)
        return self.norm2(h + self.ffn(h))


class GeoSemEncoder(nn.Module):
    def __init__(
        self,
        num_electrodes=32,
        num_freq_bands=5,
        graph_dim=64,
        st_dim=128,
        heads=4,
        graph_heads=None,
        topk=6,
        dropout=0.3,
    ):
        super().__init__()
        graph_heads = heads if graph_heads is None else graph_heads
        self.graph = DynamicGraphConv(
            num_freq_bands,
            num_electrodes,
            graph_dim,
            topk,
            dropout,
            graph_heads=graph_heads,
        )
        self.temporal = TemporalEncoder(num_freq_bands, st_dim, heads=heads, dropout=dropout)
        self.spatial_proj = nn.Linear(graph_dim, st_dim)
        self.cross_attn = CrossAttentionBlock(st_dim, heads, dropout)
        self.time_pool = nn.Linear(st_dim, 1)
        self.fuse = nn.Sequential(
            nn.Linear(st_dim + graph_dim, st_dim),
            nn.LayerNorm(st_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x, r):
        spatial, geo_token, alpha = self.graph(x, r)
        temporal = self.temporal(x)
        h_seq = self.cross_attn(temporal, self.spatial_proj(spatial))
        time_weight = F.softmax(self.time_pool(h_seq).squeeze(-1), dim=1)
        h_st = (time_weight.unsqueeze(-1) * h_seq).sum(dim=1)
        return self.fuse(torch.cat([h_st, geo_token], dim=-1)), alpha


class BottleneckAdapter(nn.Module):
    def __init__(self, dim=128, bottleneck_dim=32, dropout=0.3):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.up = nn.Linear(bottleneck_dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, h):
        return h + self.up(self.dropout(F.gelu(self.down(h))))


class PrototypeClassifier(nn.Module):
    def __init__(self, in_dim=128, text_dim=512):
        super().__init__()
        self.proj = nn.Linear(in_dim, text_dim)

    def forward(self, h):
        return F.normalize(self.proj(h), dim=-1)


class GeoSemSTDA(nn.Module):
    def __init__(
        self,
        n_sources,
        num_electrodes=32,
        num_freq_bands=5,
        st_dim=128,
        graph_dim=64,
        adapter_bottleneck=32,
        text_dim=512,
        heads=4,
        graph_heads=None,
        topk=6,
        dropout=0.3,
    ):
        super().__init__()
        self.n_sources = n_sources
        self.encoder = GeoSemEncoder(
            num_electrodes=num_electrodes,
            num_freq_bands=num_freq_bands,
            graph_dim=graph_dim,
            st_dim=st_dim,
            heads=heads,
            graph_heads=graph_heads,
            topk=topk,
            dropout=dropout,
        )
        self.adapters = nn.ModuleList([
            BottleneckAdapter(st_dim, adapter_bottleneck, dropout)
            for _ in range(n_sources)
        ])
        self.prototype_head = PrototypeClassifier(st_dim, text_dim)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x, r):
        return self.encoder(x, r)

    def project_all_adapters(self, h):
        return [self.prototype_head(adapter(h)) for adapter in self.adapters]

    def forward(self, x_src_list, r_src_list, x_tgt, r_tgt, return_features=False):
        z_src_all = []
        h_src_list = []
        alpha_src_list = []

        if x_src_list:
            for i, (x_src, r_src) in enumerate(zip(x_src_list, r_src_list)):
                h_src, alpha_src = self.encode(x_src, r_src)
                h_src_list.append(h_src)
                alpha_src_list.append(alpha_src)
                z_src_all.append(self.prototype_head(self.adapters[i](h_src)))

        h_tgt, alpha_tgt = self.encode(x_tgt, r_tgt)
        z_tgt_all = self.project_all_adapters(h_tgt)

        if return_features:
            return z_src_all, z_tgt_all, h_src_list, h_tgt, alpha_src_list, alpha_tgt
        return z_src_all, z_tgt_all


def _normalize_source_weights(source_weights, n_items, device):
    if source_weights is None:
        return torch.full((n_items,), 1.0 / max(n_items, 1), device=device)
    weights = torch.as_tensor(source_weights, dtype=torch.float32, device=device)
    if weights.numel() != n_items:
        raise ValueError(f"Expected {n_items} source weights, got {weights.numel()}")
    return weights / weights.sum().clamp_min(1e-8)


def prototype_contrastive_loss(
    z_src_all,
    y_src_list,
    text_prototypes,
    tau=0.07,
    class_weights_list=None,
    source_weights=None,
):
    text_prototypes = F.normalize(text_prototypes, dim=-1)
    losses = []
    if class_weights_list is None:
        class_weights_list = [None] * len(z_src_all)
    for z, labels, class_weights in zip(z_src_all, y_src_list, class_weights_list):
        logits = z @ text_prototypes.T / tau
        losses.append(F.cross_entropy(logits, labels, weight=class_weights))
    losses = torch.stack(losses)
    weights = _normalize_source_weights(source_weights, len(losses), losses.device)
    return (weights * losses).sum()


def multisource_mmd(z_src_all, z_tgt_all, source_weights=None):
    losses = []
    for z_src, z_tgt in zip(z_src_all, z_tgt_all):
        delta = z_src.mean(dim=0) - z_tgt.mean(dim=0)
        losses.append((delta * delta).sum())
    losses = torch.stack(losses)
    weights = _normalize_source_weights(source_weights, len(losses), losses.device)
    return (weights * losses).sum()


def _target_soft_weights(q_t, confidence_gate="none", confidence_threshold=0.6, eps=1e-6):
    conf = q_t.max(dim=-1).values.detach()
    if confidence_gate == "none":
        return torch.ones_like(conf)
    if confidence_gate == "soft":
        return conf
    if confidence_gate == "threshold":
        return (conf >= confidence_threshold).float()
    if confidence_gate == "entropy":
        entropy = -(q_t * q_t.clamp_min(eps).log()).sum(dim=-1)
        return (1.0 - entropy / math.log(q_t.size(-1))).clamp(0.0, 1.0).detach()
    raise ValueError(f"Unsupported confidence_gate: {confidence_gate}")


def multisource_class_aware_mmd(
    z_src_all,
    z_tgt_all,
    y_src_list,
    text_prototypes,
    tau=0.07,
    route_tau=None,
    num_classes=2,
    source_weights=None,
    confidence_gate="none",
    confidence_threshold=0.6,
    eps=1e-6,
):
    text_prototypes = F.normalize(text_prototypes, dim=-1)
    route_tau = tau if route_tau is None else route_tau
    losses = []

    for z_src, z_tgt, y_src in zip(z_src_all, z_tgt_all, y_src_list):
        logits_t = z_tgt @ text_prototypes.T / route_tau
        q_t = F.softmax(logits_t, dim=-1).detach()
        q_t = q_t * _target_soft_weights(
            q_t,
            confidence_gate=confidence_gate,
            confidence_threshold=confidence_threshold,
            eps=eps,
        ).unsqueeze(-1)

        class_losses = []
        class_weights = []
        for cls in range(num_classes):
            src_mask = y_src == cls
            if src_mask.any():
                src_center = z_src[src_mask].mean(dim=0)
            else:
                src_center = z_src.mean(dim=0)

            tgt_weight = q_t[:, cls].sum()
            if tgt_weight <= eps:
                continue
            tgt_center = (q_t[:, cls:cls + 1] * z_tgt).sum(dim=0) / tgt_weight.clamp_min(eps)
            class_losses.append(((src_center - tgt_center) ** 2).sum())
            class_weights.append(tgt_weight / q_t.sum().clamp_min(eps))

        if class_losses:
            class_losses = torch.stack(class_losses)
            class_weights = torch.stack(class_weights)
            class_weights = class_weights / class_weights.sum().clamp_min(eps)
            losses.append((class_weights * class_losses).sum())
        else:
            delta = z_src.mean(dim=0) - z_tgt.mean(dim=0)
            losses.append((delta * delta).sum())

    losses = torch.stack(losses)
    weights = _normalize_source_weights(source_weights, len(losses), losses.device)
    return (weights * losses).sum()


def multisource_semantic_conditional_alignment(
    z_src_all,
    z_tgt_all,
    y_src_list,
    text_prototypes,
    tau=0.07,
    route_tau=None,
    num_classes=2,
    source_weights=None,
    confidence_gate="entropy",
    confidence_threshold=0.6,
    conditional_mu=1.0,
    eps=1e-6,
):
    cond = multisource_class_aware_mmd(
        z_src_all,
        z_tgt_all,
        y_src_list,
        text_prototypes,
        tau=tau,
        route_tau=route_tau,
        num_classes=num_classes,
        source_weights=source_weights,
        confidence_gate=confidence_gate,
        confidence_threshold=confidence_threshold,
        eps=eps,
    )
    conditional_mu = float(min(max(conditional_mu, 0.0), 1.0))
    if conditional_mu >= 1.0:
        return cond
    marg = multisource_mmd(z_src_all, z_tgt_all, source_weights=source_weights)
    return (1.0 - conditional_mu) * marg + conditional_mu * cond


def _flatten_geometry(r):
    return r.flatten(start_dim=1)


def multisource_resgca(
    z_src_all,
    z_tgt_all,
    y_src_list,
    r_src_list,
    r_tgt,
    text_prototypes,
    tau=0.07,
    route_tau=None,
    num_classes=2,
    source_weights=None,
    confidence_gate="entropy",
    confidence_threshold=0.6,
    conditional_mu=1.0,
    geo_tau=1.0,
    geo_weight=1.0,
    eps=1e-6,
):
    text_prototypes = F.normalize(text_prototypes, dim=-1)
    route_tau = tau if route_tau is None else route_tau
    r_tgt_flat = _flatten_geometry(r_tgt)
    losses = []

    for z_src, z_tgt, y_src, r_src in zip(z_src_all, z_tgt_all, y_src_list, r_src_list):
        logits_t = z_tgt @ text_prototypes.T / route_tau
        q_t = F.softmax(logits_t, dim=-1).detach()
        uncertainty_weight = _target_soft_weights(
            q_t,
            confidence_gate=confidence_gate,
            confidence_threshold=confidence_threshold,
            eps=eps,
        )
        r_src_flat = _flatten_geometry(r_src)

        class_losses = []
        class_weights = []
        for cls in range(num_classes):
            src_mask = y_src == cls
            if src_mask.any():
                src_center = z_src[src_mask].mean(dim=0)
                src_geo_center = r_src_flat[src_mask].mean(dim=0)
            else:
                src_center = z_src.mean(dim=0)
                src_geo_center = r_src_flat.mean(dim=0)

            if geo_weight > 0.0:
                geo_dist = ((r_tgt_flat - src_geo_center.unsqueeze(0)) ** 2).mean(dim=-1)
                geo_trust = torch.exp(-float(geo_weight) * geo_dist / max(float(geo_tau), eps)).detach()
            else:
                geo_trust = torch.ones(z_tgt.size(0), device=z_tgt.device, dtype=z_tgt.dtype)

            target_weight = q_t[:, cls] * uncertainty_weight * geo_trust
            weight_sum = target_weight.sum()
            if weight_sum <= eps:
                continue

            target_center = (target_weight.unsqueeze(-1) * z_tgt).sum(dim=0) / weight_sum.clamp_min(eps)
            class_losses.append(((src_center - target_center) ** 2).sum())
            class_weights.append(weight_sum)

        if class_losses:
            class_losses = torch.stack(class_losses)
            class_weights = torch.stack(class_weights)
            class_weights = class_weights / class_weights.sum().clamp_min(eps)
            losses.append((class_weights * class_losses).sum())
        else:
            delta = z_src.mean(dim=0) - z_tgt.mean(dim=0)
            losses.append((delta * delta).sum())

    losses = torch.stack(losses)
    source_weight_vec = _normalize_source_weights(source_weights, len(losses), losses.device)
    cond = (source_weight_vec * losses).sum()
    conditional_mu = float(min(max(conditional_mu, 0.0), 1.0))
    if conditional_mu >= 1.0:
        return cond
    marg = multisource_mmd(z_src_all, z_tgt_all, source_weights=source_weights)
    return (1.0 - conditional_mu) * marg + conditional_mu * cond


def _js_divergence(p, q, eps=1e-6):
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    p = p / p.sum(dim=-1, keepdim=True).clamp_min(eps)
    q = q / q.sum(dim=-1, keepdim=True).clamp_min(eps)
    m = 0.5 * (p + q)
    return 0.5 * (p * (p / m.clamp_min(eps)).log()).sum(dim=-1) + \
        0.5 * (q * (q / m.clamp_min(eps)).log()).sum(dim=-1)


@torch.no_grad()
def _uot_sinkhorn_plan(cost, a, b, epsilon=0.05, tau_s=1.0, tau_t=0.5, n_iter=20, eps=1e-8):
    epsilon = max(float(epsilon), eps)
    tau_s = max(float(tau_s), eps)
    tau_t = max(float(tau_t), eps)
    theta_s = tau_s / (tau_s + epsilon)
    theta_t = tau_t / (tau_t + epsilon)

    cost = cost - cost.min()
    kernel = torch.exp(-cost / epsilon).clamp_min(eps)
    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(int(n_iter)):
        u = (a / (kernel @ v).clamp_min(eps)).clamp_min(eps).pow(theta_s)
        v = (b / (kernel.t() @ u).clamp_min(eps)).clamp_min(eps).pow(theta_t)

    return u.unsqueeze(1) * kernel * v.unsqueeze(0)


def multisource_geosem_hut(
    z_src_all,
    z_tgt_all,
    y_src_list,
    r_src_list,
    r_tgt,
    text_prototypes,
    tau=0.07,
    route_tau=None,
    num_classes=2,
    source_weights=None,
    confidence_gate="entropy",
    confidence_threshold=0.6,
    uot_epsilon=0.05,
    uot_tau_s=1.0,
    uot_tau_t=0.5,
    uot_n_iter=20,
    geo_tau=1.0,
    geo_cost_weight=0.2,
    agreement_tau=0.5,
    use_agreement_mass=True,
    use_geometry_cost=True,
    eps=1e-6,
):
    text_prototypes = F.normalize(text_prototypes, dim=-1)
    route_tau = tau if route_tau is None else route_tau
    r_tgt_flat = _flatten_geometry(r_tgt)
    losses = []

    for z_src, z_tgt, y_src, r_src in zip(z_src_all, z_tgt_all, y_src_list, r_src_list):
        logits_t = z_tgt @ text_prototypes.T / route_tau
        q_sem = F.softmax(logits_t, dim=-1).detach()
        conf_weight = _target_soft_weights(
            q_sem,
            confidence_gate=confidence_gate,
            confidence_threshold=confidence_threshold,
            eps=eps,
        )

        r_src_flat = _flatten_geometry(r_src)
        class_geo_centers = []
        for cls in range(num_classes):
            cls_mask = y_src == cls
            class_geo_centers.append(r_src_flat[cls_mask].mean(dim=0) if cls_mask.any() else r_src_flat.mean(dim=0))
        class_geo_centers = torch.stack(class_geo_centers, dim=0)
        geo_class_dist = ((r_tgt_flat.unsqueeze(1) - class_geo_centers.unsqueeze(0)) ** 2).mean(dim=-1)
        q_geo = F.softmax(-geo_class_dist / max(float(geo_tau), eps), dim=-1).detach()
        if use_agreement_mass:
            agreement = torch.exp(-_js_divergence(q_sem, q_geo, eps=eps) / max(float(agreement_tau), eps)).detach()
        else:
            agreement = torch.ones(z_tgt.size(0), device=z_tgt.device, dtype=z_tgt.dtype)

        class_losses = []
        class_masses = []
        for cls in range(num_classes):
            src_mask = y_src == cls
            if not src_mask.any():
                continue

            z_s = z_src[src_mask]
            r_s = r_src_flat[src_mask]
            semantic_cost = (1.0 - z_s @ z_tgt.T).clamp_min(0.0)

            if use_geometry_cost and geo_cost_weight > 0.0:
                geo_cost = ((r_s.unsqueeze(1) - r_tgt_flat.unsqueeze(0)) ** 2).mean(dim=-1)
                geo_cost = geo_cost / geo_cost.detach().mean().clamp_min(eps)
                cost = semantic_cost + float(geo_cost_weight) * geo_cost
            else:
                cost = semantic_cost

            source_mass = torch.ones(z_s.size(0), device=z_s.device, dtype=z_s.dtype)
            target_mass = q_sem[:, cls] * conf_weight * agreement
            if target_mass.sum() <= eps:
                continue

            source_mass = source_mass / source_mass.sum().clamp_min(eps)
            target_mass = target_mass / target_mass.sum().clamp_min(eps)
            gamma = _uot_sinkhorn_plan(
                cost.detach(),
                source_mass.detach(),
                target_mass.detach(),
                epsilon=uot_epsilon,
                tau_s=uot_tau_s,
                tau_t=uot_tau_t,
                n_iter=uot_n_iter,
                eps=eps,
            )
            transported_mass = gamma.sum().clamp_min(eps)
            class_losses.append((gamma * cost).sum() / transported_mass)
            class_masses.append((q_sem[:, cls] * conf_weight * agreement).sum().detach())

        if class_losses:
            class_losses = torch.stack(class_losses)
            class_masses = torch.stack(class_masses)
            class_masses = class_masses / class_masses.sum().clamp_min(eps)
            losses.append((class_masses * class_losses).sum())
        else:
            delta = z_src.mean(dim=0) - z_tgt.mean(dim=0)
            losses.append((delta * delta).sum())

    losses = torch.stack(losses)
    source_weight_vec = _normalize_source_weights(source_weights, len(losses), losses.device)
    return (source_weight_vec * losses).sum()


def lambda_warmup(step, total_steps, lambda_max):
    if total_steps <= 0:
        return float(lambda_max)
    progress = min(max(step / total_steps, 0.0), 1.0)
    return float(lambda_max) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


@torch.no_grad()
def compute_source_domain_centroids(model, source_loaders, device):
    centroids = []
    for source_idx, loader in enumerate(source_loaders):
        feats = []
        for xb, rb, _ in loader:
            xb = xb.to(device)
            rb = rb.to(device)
            h, _ = model.encode(xb, rb)
            z = model.prototype_head(model.adapters[source_idx](h))
            feats.append(z)
        centroids.append(F.normalize(torch.cat(feats, dim=0).mean(dim=0), dim=-1))
    return torch.stack(centroids, dim=0)


@torch.no_grad()
def compute_source_class_centroids(model, source_loaders, device, num_classes):
    centroids = []
    for source_idx, loader in enumerate(source_loaders):
        sums = None
        counts = torch.zeros(num_classes, device=device)
        for xb, rb, yb in loader:
            xb = xb.to(device)
            rb = rb.to(device)
            yb = yb.to(device)
            h, _ = model.encode(xb, rb)
            z = model.prototype_head(model.adapters[source_idx](h))
            if sums is None:
                sums = torch.zeros(num_classes, z.size(-1), device=device)
            sums.index_add_(0, yb, z)
            counts.index_add_(0, yb, torch.ones_like(yb, dtype=z.dtype))
        counts = counts.clamp_min(1.0).unsqueeze(-1)
        centroids.append(F.normalize(sums / counts, dim=-1))
    return torch.stack(centroids, dim=0)


@torch.no_grad()
def predict_class_aware(model, dataloader, text_prototypes, source_class_centroids,
                        device, proto_tau=0.07, fusion_tau=0.5,
                        eval_classifier="text", centroid_blend=0.5,
                        source_domain_centroids=None,
                        source_reliability_weights=None,
                        reliability_fusion=False):
    model.eval()
    text_prototypes = F.normalize(text_prototypes.to(device), dim=-1)
    source_class_centroids = source_class_centroids.to(device)
    eval_classifier = str(eval_classifier).lower()
    centroid_blend = float(min(max(centroid_blend, 0.0), 1.0))
    y_true, y_pred = [], []

    for xb, rb, yb in dataloader:
        xb = xb.to(device)
        rb = rb.to(device)
        _, z_tgt_all = model([], [], xb, rb)
        z_stack = torch.stack(z_tgt_all, dim=0)  # [K, B, D]
        logits = z_stack @ text_prototypes.T / proto_tau
        probs = F.softmax(logits, dim=-1)
        sims_to_centroids = torch.einsum("kbd,kcd->kbc", z_stack, source_class_centroids)
        expected_dist = (probs * (1.0 - sims_to_centroids)).sum(dim=-1)
        weights = F.softmax(-expected_dist / fusion_tau, dim=0).unsqueeze(-1)
        z_fused = F.normalize((weights * z_stack).sum(dim=0), dim=-1)
        text_logits = z_fused @ text_prototypes.T
        centroid_logits = (weights * sims_to_centroids).sum(dim=0)

        if eval_classifier == "text":
            logits_eval = text_logits
        elif eval_classifier == "centroid":
            logits_eval = centroid_logits
        elif eval_classifier == "hybrid":
            logits_eval = (1.0 - centroid_blend) * text_logits + centroid_blend * centroid_logits
        elif eval_classifier == "senior_feature":
            if source_domain_centroids is None:
                raise ValueError("source_domain_centroids is required for senior_feature evaluation")
            source_domain_centroids = source_domain_centroids.to(device)
            domain_dists = torch.norm(
                z_stack - source_domain_centroids.unsqueeze(1),
                p=2,
                dim=-1,
            )
            domain_logits = -domain_dists / fusion_tau
            if reliability_fusion:
                prior = _normalize_source_weights(
                    source_reliability_weights,
                    z_stack.size(0),
                    device,
                ).clamp_min(1e-8)
                domain_logits = domain_logits + prior.log().unsqueeze(-1)
            domain_weights = F.softmax(domain_logits, dim=0).unsqueeze(-1)
            z_senior = F.normalize((domain_weights * z_stack).sum(dim=0), dim=-1)
            logits_eval = z_senior @ text_prototypes.T
        else:
            raise ValueError(f"Unsupported eval_classifier: {eval_classifier}")

        pred = logits_eval.argmax(dim=-1)
        y_true.append(yb.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    return y_true, y_pred
