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
    adj = adj + torch.eye(channels, device=r.device, dtype=r.dtype).unsqueeze(0)
    return adj / adj.sum(dim=-1, keepdim=True).clamp_min(eps)


def vech(x):
    channels = x.size(-1)
    rows, cols = torch.triu_indices(channels, channels, device=x.device)
    return x[..., rows, cols]


class AttentionAdjacency(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)

    def forward(self, h):
        logits = self.q(h) @ self.k(h).transpose(-1, -2)
        logits = logits / math.sqrt(h.size(-1))
        return F.softmax(logits, dim=-1)


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
    def __init__(self, num_bands, channels, graph_dim=64, topk=6, dropout=0.3):
        super().__init__()
        self.channels = channels
        self.topk = topk
        self.node_proj = nn.Linear(num_bands, graph_dim)
        self.learn_adj = AttentionAdjacency(graph_dim)
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
    def __init__(self, num_bands, dim, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_bands, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x.mean(dim=2))


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
        topk=6,
        dropout=0.3,
    ):
        super().__init__()
        self.graph = DynamicGraphConv(num_freq_bands, num_electrodes, graph_dim, topk, dropout)
        self.temporal = TemporalEncoder(num_freq_bands, st_dim, dropout)
        self.spatial_proj = nn.Linear(graph_dim, st_dim)
        self.cross_attn = CrossAttentionBlock(st_dim, heads, dropout)
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
        h_st = h_seq.mean(dim=1)
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


def multisource_class_aware_mmd(
    z_src_all,
    z_tgt_all,
    y_src_list,
    text_prototypes,
    tau=0.07,
    num_classes=2,
    source_weights=None,
    confidence_gate="none",
    confidence_threshold=0.6,
    eps=1e-6,
):
    text_prototypes = F.normalize(text_prototypes, dim=-1)
    losses = []

    for z_src, z_tgt, y_src in zip(z_src_all, z_tgt_all, y_src_list):
        logits_t = z_tgt @ text_prototypes.T / tau
        q_t = F.softmax(logits_t, dim=-1).detach()
        conf = q_t.max(dim=-1).values.detach()

        if confidence_gate == "soft":
            q_t = q_t * conf.unsqueeze(-1)
        elif confidence_gate == "threshold":
            q_t = q_t * (conf >= confidence_threshold).float().unsqueeze(-1)
        elif confidence_gate != "none":
            raise ValueError(f"Unsupported confidence_gate: {confidence_gate}")

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


def lambda_warmup(step, total_steps, lambda_max):
    if total_steps <= 0:
        return float(lambda_max)
    progress = min(max(step / total_steps, 0.0), 1.0)
    return float(lambda_max) * (2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0)


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
                        device, proto_tau=0.07, fusion_tau=0.5):
    model.eval()
    text_prototypes = F.normalize(text_prototypes.to(device), dim=-1)
    source_class_centroids = source_class_centroids.to(device)
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
        pred = (z_fused @ text_prototypes.T).argmax(dim=-1)
        y_true.append(yb.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    return y_true, y_pred
