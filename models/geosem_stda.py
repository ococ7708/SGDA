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


class DirectDEEncoder(nn.Module):
    """Minimal direct path that preserves information from flattened DE input."""

    def __init__(self, input_dim, st_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, st_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x.flatten(start_dim=1))


CAST_LEVEL1_VARIANTS = (
    "e1_strong_de",
    "e2_strong_de_channel",
    "e3_strong_de_channel_graph",
    "e4_strong_de_channel_graph_multiscale",
    "e5_strong_de_full_st",
)


class StrongDEEncoder(nn.Module):
    """R4-compatible direct DE projection plus one residual MLP block."""

    def __init__(self, input_dim, st_dim=128, dropout=0.3):
        super().__init__()
        self.input_proj = DirectDEEncoder(input_dim, st_dim=st_dim, dropout=dropout)
        self.residual_block = nn.Sequential(
            nn.LayerNorm(st_dim),
            nn.Linear(st_dim, st_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(st_dim * 2, st_dim),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(st_dim)

    def forward(self, x):
        h = self.input_proj(x)
        return self.output_norm(h + self.residual_block(h))


class DynamicChannelSelector(nn.Module):
    """Sample-dependent channel weighting without electrode coordinates or SPD features."""

    def __init__(self, num_freq_bands, hidden_dim=32):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(num_freq_bands, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # Aggregate only over time; retain one score per EEG channel.
        logits = self.score(x.mean(dim=1)).squeeze(-1)
        weights = F.softmax(logits, dim=-1)
        # Mean-one scaling keeps the input magnitude comparable with E1.
        scaled = x * (weights * x.size(2)).unsqueeze(1).unsqueeze(-1)
        return scaled, weights


class LearnedSparseSpatialGraph(nn.Module):
    """Learned top-k channel graph built only from DE node features."""

    def __init__(self, num_freq_bands, dim=128, heads=4, topk=6, dropout=0.3):
        super().__init__()
        if dim % heads != 0:
            raise ValueError(f"dim={dim} must be divisible by graph heads={heads}")
        self.heads = heads
        self.head_dim = dim // heads
        self.topk = topk
        self.node_proj = nn.Linear(num_freq_bands, dim)
        self.q = nn.Linear(dim, dim, bias=False)
        self.k = nn.Linear(dim, dim, bias=False)
        self.message = nn.Linear(dim, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        self.pool_score = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        bsz, steps, channels, _ = x.shape
        if channels < 2:
            raise ValueError("LearnedSparseSpatialGraph requires at least two channels")
        h0 = F.gelu(self.node_proj(x))
        q = self.q(h0).view(bsz, steps, channels, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        k = self.k(h0).view(bsz, steps, channels, self.heads, self.head_dim).permute(0, 1, 3, 2, 4)
        logits = (q @ k.transpose(-1, -2)) / math.sqrt(self.head_dim)
        logits = logits.mean(dim=2)
        keep = min(max(int(self.topk), 1), channels)
        top_indices = logits.topk(k=keep, dim=-1).indices
        mask = torch.zeros_like(logits, dtype=torch.bool).scatter_(-1, top_indices, True)
        sparse_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        adjacency = F.softmax(sparse_logits, dim=-1)
        h = adjacency @ self.message(h0)
        h = self.norm(h0 + self.dropout(F.gelu(h)))
        channel_weights = F.softmax(self.pool_score(h).squeeze(-1), dim=-1)
        spatial_tokens = (channel_weights.unsqueeze(-1) * h).sum(dim=2)
        return spatial_tokens, adjacency, channel_weights


class MultiScaleTemporalContext(nn.Module):
    """Temporal kernels 3/5/7, learned scale weights, then one MHSA block."""

    def __init__(self, dim=128, heads=4, dropout=0.3, kernels=(3, 5, 7)):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv1d(dim, dim, kernel_size=k, padding=k // 2) for k in kernels
        )
        self.scale_logits = nn.Parameter(torch.zeros(len(kernels)))
        self.conv_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, spatial_tokens):
        conv_input = spatial_tokens.transpose(1, 2)
        scales = torch.stack(
            [F.gelu(conv(conv_input)).transpose(1, 2) for conv in self.convs], dim=2
        )
        scale_weights = F.softmax(self.scale_logits, dim=0)
        h = (scales * scale_weights.view(1, 1, -1, 1)).sum(dim=2)
        h = self.conv_norm(spatial_tokens + self.dropout(h))
        context, _ = self.attn(h, h, h, need_weights=False)
        return self.attn_norm(h + self.dropout(context)), scale_weights


class CASTLevel1Encoder(nn.Module):
    """Incremental CAST-EEG Level-1 encoder with no geometry/alignment dependency."""

    def __init__(
        self,
        variant,
        sample_length,
        num_electrodes,
        num_freq_bands,
        st_dim=128,
        heads=4,
        graph_heads=4,
        topk=6,
        dropout=0.3,
        beta_initial=0.1,
    ):
        super().__init__()
        if variant not in CAST_LEVEL1_VARIANTS:
            raise ValueError(f"Unsupported CAST Level-1 variant: {variant}")
        self.variant = variant
        self.level = CAST_LEVEL1_VARIANTS.index(variant) + 1
        input_dim = int(sample_length) * int(num_electrodes) * int(num_freq_bands)
        self.strong_de = StrongDEEncoder(input_dim, st_dim=st_dim, dropout=dropout)
        self.channel_selector = (
            DynamicChannelSelector(num_freq_bands) if self.level >= 2 else None
        )
        self.spatial_graph = (
            LearnedSparseSpatialGraph(
                num_freq_bands,
                dim=st_dim,
                heads=graph_heads,
                topk=topk,
                dropout=dropout,
            )
            if self.level >= 3 else None
        )
        self.temporal = (
            MultiScaleTemporalContext(st_dim, heads=heads, dropout=dropout)
            if self.level >= 4 else None
        )
        self.cross_attn = (
            CrossAttentionBlock(st_dim, heads=heads, dropout=dropout)
            if self.level >= 5 else None
        )
        self.simple_fusion_norm = nn.LayerNorm(st_dim) if self.level in (3, 4) else None
        self.gated_fusion_norm = nn.LayerNorm(st_dim) if self.level >= 5 else None
        if self.level >= 5:
            beta_logit = math.log(beta_initial / (1.0 - beta_initial))
            self.beta_logit = nn.Parameter(torch.tensor(beta_logit, dtype=torch.float32))
        else:
            self.register_parameter("beta_logit", None)
        self.last_flow = None

    @property
    def beta(self):
        return torch.sigmoid(self.beta_logit) if self.beta_logit is not None else None

    def forward(self, x):
        x_selected = x
        channel_weights = None
        if self.channel_selector is not None:
            x_selected, channel_weights = self.channel_selector(x)
        h_base = self.strong_de(x_selected)

        spatial_tokens = adjacency = spatial_channel_weights = None
        temporal_tokens = scale_weights = cross_tokens = None
        if self.spatial_graph is not None:
            spatial_tokens, adjacency, spatial_channel_weights = self.spatial_graph(x_selected)
        if self.temporal is not None:
            temporal_tokens, scale_weights = self.temporal(spatial_tokens)

        if self.level <= 2:
            h = h_base
        elif self.level == 3:
            h = self.simple_fusion_norm(h_base + spatial_tokens.mean(dim=1))
        elif self.level == 4:
            h = self.simple_fusion_norm(h_base + temporal_tokens.mean(dim=1))
        else:
            cross_tokens = self.cross_attn(temporal_tokens, spatial_tokens)
            h = self.gated_fusion_norm(h_base + self.beta * cross_tokens.mean(dim=1))

        self.last_flow = {
            "input": tuple(x.shape),
            "channel_weights": None if channel_weights is None else tuple(channel_weights.shape),
            "h_base": tuple(h_base.shape),
            "spatial_tokens": None if spatial_tokens is None else tuple(spatial_tokens.shape),
            "adjacency": None if adjacency is None else tuple(adjacency.shape),
            "spatial_channel_weights": (
                None if spatial_channel_weights is None else tuple(spatial_channel_weights.shape)
            ),
            "temporal_tokens": None if temporal_tokens is None else tuple(temporal_tokens.shape),
            "scale_weights": None if scale_weights is None else tuple(scale_weights.shape),
            "cross_tokens": None if cross_tokens is None else tuple(cross_tokens.shape),
            "output": tuple(h.shape),
        }
        return h


class ResidualFusion(nn.Module):
    """Fuse direct DE and structured representations without changing their width."""

    def __init__(self, dim=128, mode="none", beta_initial=0.1):
        super().__init__()
        if mode not in ("none", "fixed", "gated"):
            raise ValueError(f"Unsupported residual fusion mode: {mode}")
        if not 0.0 < beta_initial < 1.0:
            raise ValueError(f"beta_initial must be in (0, 1), got {beta_initial}")
        self.mode = mode
        self.norm = nn.LayerNorm(dim) if mode != "none" else nn.Identity()
        if mode == "gated":
            beta_logit = math.log(beta_initial / (1.0 - beta_initial))
            self.beta_logit = nn.Parameter(torch.tensor(beta_logit, dtype=torch.float32))
        else:
            self.register_parameter("beta_logit", None)

    @property
    def beta(self):
        if self.mode == "fixed":
            return 1.0
        if self.mode == "gated":
            return torch.sigmoid(self.beta_logit)
        return None

    def forward(self, h_de, h_struct):
        if self.mode == "none":
            return h_struct
        if h_de is None:
            raise ValueError(f"h_de is required for residual fusion mode={self.mode}")
        beta = 1.0 if self.mode == "fixed" else torch.sigmoid(self.beta_logit)
        return self.norm(h_de + beta * h_struct)


class LinearClassifier(nn.Module):
    def __init__(self, text_dim=512, num_classes=2):
        super().__init__()
        self.linear = nn.Linear(text_dim, num_classes)

    def forward(self, z):
        return self.linear(z)


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
        sample_length=3,
        representation_mode="geosem",
        classifier_type="clip",
        num_classes=2,
        beta_initial=0.1,
        cast_variant=None,
    ):
        super().__init__()
        if representation_mode not in ("geosem", "de_residual", "de_gated", "de_only", "cast_level1"):
            raise ValueError(f"Unsupported representation_mode: {representation_mode}")
        if classifier_type not in ("clip", "linear"):
            raise ValueError(f"Unsupported classifier_type: {classifier_type}")
        self.n_sources = n_sources
        self.representation_mode = representation_mode
        self.classifier_type = classifier_type
        self.encoder = None
        if representation_mode not in ("de_only", "cast_level1"):
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
        self.cast_encoder = None
        if representation_mode == "cast_level1":
            self.cast_encoder = CASTLevel1Encoder(
                variant=cast_variant,
                sample_length=sample_length,
                num_electrodes=num_electrodes,
                num_freq_bands=num_freq_bands,
                st_dim=st_dim,
                heads=heads,
                graph_heads=graph_heads if graph_heads is not None else heads,
                topk=topk,
                dropout=dropout,
                beta_initial=beta_initial,
            )
        fusion_mode = {
            "geosem": "none",
            "de_residual": "fixed",
            "de_gated": "gated",
            "de_only": "none",
            "cast_level1": "none",
        }[representation_mode]
        input_dim = int(sample_length) * int(num_electrodes) * int(num_freq_bands)
        self.direct_de_encoder = (
            DirectDEEncoder(input_dim, st_dim=st_dim, dropout=dropout)
            if representation_mode not in ("geosem", "cast_level1")
            else None
        )
        self.residual_fusion = ResidualFusion(
            dim=st_dim,
            mode=fusion_mode,
            beta_initial=beta_initial,
        )
        self.adapters = nn.ModuleList([
            BottleneckAdapter(st_dim, adapter_bottleneck, dropout)
            for _ in range(n_sources)
        ])
        self.prototype_head = PrototypeClassifier(st_dim, text_dim)
        self.linear_head = None
        self.last_representation_shapes = None
        self.apply(self._init_weights)
        if classifier_type == "linear":
            # Initialize the R3-only head after all shared modules so R2 and R3
            # have identical shared parameters under the same random seed.
            self.linear_head = LinearClassifier(text_dim=text_dim, num_classes=num_classes)
            self.linear_head.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(self, x, r):
        if self.representation_mode == "cast_level1":
            h = self.cast_encoder(x)
            flow = self.cast_encoder.last_flow
            self.last_representation_shapes = {
                "h_de": flow["h_base"],
                "h_struct": flow["cross_tokens"] or flow["temporal_tokens"] or flow["spatial_tokens"],
                "h_fused": flow["output"],
                "cast_flow": flow,
            }
            return h, None
        if self.representation_mode == "de_only":
            h_de = self.direct_de_encoder(x)
            self.last_representation_shapes = {
                "h_de": tuple(h_de.shape),
                "h_struct": None,
                "h_fused": tuple(h_de.shape),
            }
            return h_de, None
        h_struct, alpha = self.encoder(x, r)
        h_de = self.direct_de_encoder(x) if self.direct_de_encoder is not None else None
        h = self.residual_fusion(h_de, h_struct)
        self.last_representation_shapes = {
            "h_de": None if h_de is None else tuple(h_de.shape),
            "h_struct": tuple(h_struct.shape),
            "h_fused": tuple(h.shape),
        }
        return h, alpha

    def current_beta(self):
        if self.cast_encoder is not None:
            beta = self.cast_encoder.beta
            return None if beta is None else float(beta.detach().cpu().item())
        beta = self.residual_fusion.beta
        if beta is None:
            return None
        if torch.is_tensor(beta):
            return float(beta.detach().cpu().item())
        return float(beta)

    def project_all_adapters(self, h):
        return [self.prototype_head(adapter(h)) for adapter in self.adapters]

    def forward(self, x_src_list, r_src_list, x_tgt=None, r_tgt=None, return_features=False):
        z_src_all = []
        h_src_list = []
        alpha_src_list = []

        if x_src_list:
            for i, (x_src, r_src) in enumerate(zip(x_src_list, r_src_list)):
                h_src, alpha_src = self.encode(x_src, r_src)
                h_src_list.append(h_src)
                if alpha_src is not None:
                    alpha_src_list.append(alpha_src)
                z_src_all.append(self.prototype_head(self.adapters[i](h_src)))

        if x_tgt is not None:
            if r_tgt is None:
                raise ValueError("r_tgt is required when x_tgt is provided")
            h_tgt, alpha_tgt = self.encode(x_tgt, r_tgt)
            z_tgt_all = self.project_all_adapters(h_tgt)
        else:
            h_tgt, alpha_tgt, z_tgt_all = None, None, []

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


def linear_classification_loss(
    z_src_all,
    y_src_list,
    linear_head,
    class_weights_list=None,
    source_weights=None,
):
    """Shared supervised linear-head loss across all source adapters."""
    if linear_head is None:
        raise ValueError("linear_head is required for linear classification loss")
    losses = []
    if class_weights_list is None:
        class_weights_list = [None] * len(z_src_all)
    for z, labels, class_weights in zip(z_src_all, y_src_list, class_weights_list):
        losses.append(F.cross_entropy(linear_head(z), labels, weight=class_weights))
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
                        reliability_fusion=False,
                        classifier_type=None):
    model.eval()
    classifier_type = model.classifier_type if classifier_type is None else str(classifier_type).lower()
    if classifier_type not in ("clip", "linear"):
        raise ValueError(f"Unsupported classifier_type: {classifier_type}")
    if classifier_type == "clip":
        text_prototypes = F.normalize(text_prototypes.to(device), dim=-1)
    if source_class_centroids is not None:
        source_class_centroids = source_class_centroids.to(device)
    eval_classifier = str(eval_classifier).lower()
    centroid_blend = float(min(max(centroid_blend, 0.0), 1.0))
    y_true, y_pred = [], []

    for xb, rb, yb in dataloader:
        xb = xb.to(device)
        rb = rb.to(device)
        _, z_tgt_all = model([], [], xb, rb)
        z_stack = torch.stack(z_tgt_all, dim=0)  # [K, B, D]
        if eval_classifier == "senior_feature":
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
            if classifier_type == "linear":
                if model.linear_head is None:
                    raise ValueError("Model has no linear_head for linear classification")
                logits_eval = model.linear_head(z_senior)
            else:
                logits_eval = z_senior @ text_prototypes.T
        else:
            if classifier_type == "linear":
                raise ValueError("Linear classification requires classifier-independent senior_feature fusion")
            if source_class_centroids is None:
                raise ValueError(f"source_class_centroids is required for {eval_classifier} evaluation")
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
            else:
                raise ValueError(f"Unsupported eval_classifier: {eval_classifier}")

        pred = logits_eval.argmax(dim=-1)
        y_true.append(yb.cpu().numpy())
        y_pred.append(pred.cpu().numpy())

    return y_true, y_pred
