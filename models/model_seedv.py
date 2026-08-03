import torch
import torch.nn as nn
import torch.nn.functional as F

class SFE(nn.Module):
    def __init__(self, num_electrodes=62, num_freq_bands=5, dropout=0.3, final_dim=512):
        super().__init__()

        # ==========================================
        # 1. 卷积层设计
        # 输入图形状: [B, 1, 310, 3] (Height=310, Width=3)
        # ==========================================
        self.dw_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,         # 建议增加到 16 或 24，因为单层需要更多特征图
                # kernel_size=(5, 3):
                #   - Height=5: 一次看 1个电极的所有频带 (正好5个频带)
                #   - Width=3:  一次看完所有时间步 (T=3)
                kernel_size=(5, 3),      
                stride=(1, 1),           # 步长设为 1，精细扫描
                # 效果：将 310 的高度特征图压缩为 155，去除冗余，保留关键模式
                # stride=(2, 1),
                padding=(2, 0),          # H方向padding=2保持高度，W方向padding=0变成1
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU()
        )

        # ==========================================
        # 2. 维度计算
        # ==========================================
        # 卷积后 H 保持 310 (因为 pad=2, k=5)，W 变为 1 (因为 k=3, T=3, pad=0)
        # 输出形状: [B, 16, 310, 1]
        self.flat_dim = 16 * (num_electrodes * num_freq_bands) * 1
        # conv_out_height = (num_electrodes * num_freq_bands + 1) // 2 
        # self.flat_dim = 16 * conv_out_height * 1
        
        # 自动适配中间层
        if self.flat_dim > 2000:
            hidden_dim = 1024
        else:
            hidden_dim = 768

        print(f"Structure: [C*Fre, T] -> Flatten Dim: {self.flat_dim}")

        # ==========================================
        # 3. 全连接层
        # ==========================================
        self.fc = nn.Sequential(
            nn.Linear(self.flat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, final_dim),
            nn.BatchNorm1d(final_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Input x: [B, T, C*Fre] = [B, 3, 310]
        # B, T, C, Fre = x.shape
        
        x = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        
        x = self.dw_conv(x)  # -> [B, 16, 310, 1]
        
        x = x.flatten(1)     # -> [B, 16*310*1]
        x = self.fc(x)
        return x

# ==========================================
# 2. 域特定分支 (DSFE)
# ==========================================
class DSFEBranch(nn.Module):
    def __init__(self, in_dim=512, proj_dim=512,bottleneck_dim=128, dropout=0.3): # 增加 dropout 参数，建议初始设为 0.5
        super().__init__()
        self.net = nn.Sequential(
            # 第一步：压缩去噪 (Encoder)
            nn.Linear(in_dim, bottleneck_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            
            # 第二步：语义重构/对齐 (Decoder/Projector)
            nn.Linear(bottleneck_dim, proj_dim)
        )
    def forward(self, x):
        x = self.net(x)
        # 核心：L2 Normalize，映射到单位超球面，用于余弦相似度计算
        x = F.normalize(x, dim=-1)
        return x

# ==========================================
# 3. 整体 MS-MDA 风格对齐网络 (含初始化)
# ==========================================
class SGDA_model(nn.Module):
    def __init__(self, n_sources=14, num_electrodes=62, eeg_dim=512,bottleneck_dim=128, text_dim=512,dropout=0.3):
        super().__init__()
        self.n_sources = n_sources # 修复：必须保存 n_sources 供 forward 使用
        
        self.sfe = SFE(num_electrodes=num_electrodes, final_dim=eeg_dim)
        
        # 多源分支
        self.branches = nn.ModuleList([
            # 2. 显式传入 bottleneck_dim
            DSFEBranch(in_dim=eeg_dim, proj_dim=text_dim, bottleneck_dim=bottleneck_dim, dropout=dropout) 
            for _ in range(n_sources)
        ])

        # -----------------------------------------------------
        # 初始化逻辑入口
        # -----------------------------------------------------
        self.apply(self._init_weights)       # 1. 应用通用 Kaiming 初始化
        self._init_final_layers()            # 2. 覆盖最后一层为正交初始化

    def _init_weights(self, m):
        """
        通用初始化函数：适用于大多数隐藏层
        """
        if isinstance(m, nn.Conv2d):
            # Kaiming Normal 针对 ELU/ReLU 的非线性特性
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            # 归一化层初始化：保持原样
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            # 默认线性层使用 Kaiming
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _init_final_layers(self):
        """
        特殊初始化：针对直接输出 Embedding 的层
        对于 Metric Learning (Cosine Similarity)，正交初始化优于高斯分布
        """
        for branch in self.branches:
            # 动态遍历 net 容器，找出所有的 Linear 层
            linears = [m for m in branch.net if isinstance(m, nn.Linear)]
            if linears:
                # 取列表中最后一个 Linear 层（即输出层）
                last_linear = linears[-1]
                nn.init.orthogonal_(last_linear.weight, gain=1.0)
            if last_linear.bias is not None:
                nn.init.constant_(last_linear.bias, 0)

    def forward(self, x_src_list, x_tgt):
            """
            x_src_list: List of Tensors or []
            x_tgt: Tensor
            """
            z_src_all = []
            
            # --- 修改点 1: 检查源域列表是否为空 ---
            if len(x_src_list) > 0:
                batch_sizes = [x.shape[0] for x in x_src_list]
                x_src_cat = torch.cat(x_src_list, dim=0)  # 只有非空才 cat 

                # 共享特征提取
                feat_src_cat = self.sfe(x_src_cat) 
                # 按 source 拆分
                src_feat_list = torch.split(feat_src_cat, batch_sizes, dim=0)
                
                # 通过每个分支进行投影
                for i in range(self.n_sources):
                    z_src = self.branches[i](src_feat_list[i])
                    z_src_all.append(z_src)
            # ------------------------------------

            # --- 修改点 2: 目标域处理 (始终执行) ---
            tgt_feat = self.sfe(x_tgt) # [B_t, eeg_dim]
            z_tgt_all = []
            
            for i in range(self.n_sources):
                z_tgt = self.branches[i](tgt_feat)
                z_tgt_all.append(z_tgt)
            # ------------------------------------

            return z_src_all, z_tgt_all