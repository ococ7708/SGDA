import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. SFE: 浅层共享特征提取 
# ==========================================
class SFE_SimpleLinear(nn.Module):
    def __init__(self, num_electrodes=62, num_freq_bands=5, num_time_steps=3, hidden_dim=512, final_dim=128, dropout=0.3):
        super().__init__()
        
        # 输入维度: 3 * 62 * 5 = 930
        self.input_dim = num_time_steps * num_electrodes * num_freq_bands 
        
        self.net = nn.Sequential(
            # 第一层
            nn.Linear(self.input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            
            # 第二层:
            nn.Linear(hidden_dim, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.LeakyReLU(0.1, inplace=True) 
        )

    def forward(self, x):
        B = x.size(0)
        x_flat = x.view(B, -1) 
        out = self.net(x_flat)
        return out

# ==========================================
# 2. DSFE: 域特定分支 (在低维空间微调)
# ==========================================
class DSFEBranch_OptionA(nn.Module):
    def __init__(self, in_dim=128,bottleneck_dim=64, proj_dim=128, dropout=0.3): 
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(in_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(dropout),
            
            nn.Linear(bottleneck_dim, proj_dim)
        )

    def forward(self, x):
        x = self.net(x)
        x = F.normalize(x, dim=-1) # 投影到单位超球面上
        return x

# ==========================================
# 3. 主模型: 包含 EEG 分支和文本投影分支
# ==========================================
class SGDA_model(nn.Module):
    def __init__(self, n_sources=14, num_electrodes=62,
                 eeg_dim=512, 
                 bottleneck_dim=64,
                 text_dim=512,
                 dropout=0.1):       # 对齐的目标低维空间 (关键参数)
        super().__init__()
        self.n_sources = n_sources

        
        # --- EEG 分支 ---
        # SFE 输出 512/768 维
        self.sfe = SFE_SimpleLinear(
            num_electrodes=num_electrodes, 
            num_freq_bands=5, 
            num_time_steps=3, 
            hidden_dim=768,        # 中间层可以适当减小
            final_dim=eeg_dim   
        )
        
        self.branches = nn.ModuleList([
            DSFEBranch_OptionA(in_dim=eeg_dim,bottleneck_dim=bottleneck_dim, proj_dim=text_dim, dropout=dropout)
            for _ in range(n_sources)
        ])
        # 初始化
        self.apply(self._init_weights)
        self._init_final_layers()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def _init_final_layers(self):
        # 初始化 EEG 分支的最后一层
        for branch in self.branches:
            last_linear = branch.net[-1]
            nn.init.orthogonal_(last_linear.weight, gain=1.0)
            if last_linear.bias is not None: nn.init.constant_(last_linear.bias, 0)
        

    def forward(self, x_src_list, x_tgt):
        """
        x_src_list: List of source EEG tensors
        x_tgt: Target EEG tensor
        text_vectors: [Num_Classes, 512] 包含所有类别(如3类)的原始 CLIP 向量
        """        
        
        # 2. 处理 EEG 源域数据
        z_src_all = []
        if len(x_src_list) > 0:
            batch_sizes = [x.shape[0] for x in x_src_list]
            x_src_cat = torch.cat(x_src_list, dim=0)
            
            feat_src_cat = self.sfe(x_src_cat) # -> [Total_Batch, 128]
            
            src_feat_list = torch.split(feat_src_cat, batch_sizes, dim=0)
            for i in range(self.n_sources):
                z_src = self.branches[i](src_feat_list[i]) # -> [Batch_i, 128]
                z_src_all.append(z_src)

        # 3. 处理 EEG 目标域数据
        tgt_feat = self.sfe(x_tgt) # -> [Batch_t, 128]
        z_tgt_all = []
        for i in range(self.n_sources):
            z_tgt = self.branches[i](tgt_feat) # -> [Batch_t, 128]
            z_tgt_all.append(z_tgt)

        # 返回: 源域特征列表, 目标域特征列表, 投影后的文本原型
        return z_src_all, z_tgt_all