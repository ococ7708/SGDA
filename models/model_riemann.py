import torch
import torch.nn as nn
import torch.nn.functional as F


class SFE_Riemann(nn.Module):
    """MLP-based Shared Feature Extractor for Riemann tangent space features"""
    def __init__(self, input_dim=528, hidden_dim=768, final_dim=512, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DSFEBranch(nn.Module):
    def __init__(self, in_dim=512, proj_dim=512, bottleneck_dim=128, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(bottleneck_dim, proj_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, dim=-1)
        return x


class SGDA_model(nn.Module):
    def __init__(self, n_sources=14, num_electrodes=528, eeg_dim=512,
                 bottleneck_dim=128, text_dim=512, dropout=0.3):
        super().__init__()
        self.n_sources = n_sources

        self.sfe = SFE_Riemann(
            input_dim=num_electrodes,
            hidden_dim=768,
            final_dim=eeg_dim,
            dropout=dropout
        )

        self.branches = nn.ModuleList([
            DSFEBranch(in_dim=eeg_dim, proj_dim=text_dim,
                       bottleneck_dim=bottleneck_dim, dropout=dropout)
            for _ in range(n_sources)
        ])

        self.apply(self._init_weights)
        self._init_final_layers()

    def _init_weights(self, m):
        if isinstance(m, (nn.BatchNorm1d,)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_final_layers(self):
        for branch in self.branches:
            linears = [m for m in branch.net if isinstance(m, nn.Linear)]
            if linears:
                last_linear = linears[-1]
                nn.init.orthogonal_(last_linear.weight, gain=1.0)
                if last_linear.bias is not None:
                    nn.init.constant_(last_linear.bias, 0)

    def forward(self, x_src_list, x_tgt, return_sfe_features=False):
        z_src_all = []
        src_feat_list = []

        if len(x_src_list) > 0:
            batch_sizes = [x.shape[0] for x in x_src_list]
            x_src_cat = torch.cat(x_src_list, dim=0)
            feat_src_cat = self.sfe(x_src_cat)
            src_feat_list = list(torch.split(feat_src_cat, batch_sizes, dim=0))

            for i in range(self.n_sources):
                z_src = self.branches[i](src_feat_list[i])
                z_src_all.append(z_src)

        tgt_feat = self.sfe(x_tgt)
        z_tgt_all = []
        for i in range(self.n_sources):
            z_tgt = self.branches[i](tgt_feat)
            z_tgt_all.append(z_tgt)

        if return_sfe_features:
            return z_src_all, z_tgt_all, src_feat_list, tgt_feat
        return z_src_all, z_tgt_all
