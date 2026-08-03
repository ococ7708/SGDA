import torch
import torch.nn.functional as F
import torch.nn as nn


def align_loss(z, text_embeds, labels):
    """
    z: [B, D]
    text_embeds: [C, D]
    labels: [B] (long)
    """
    selected = text_embeds[labels]            # [B, D]
    cos_sim = (z * selected).sum(-1)   # cosine since both normalized
    cos_loss = (1.0 - cos_sim).mean()  
    return cos_loss


def mmd_linear(f_s, f_t):
    """线性MMD实现"""
    delta = f_s.mean(0) - f_t.mean(0)
    return torch.sum(delta * delta)


def discrepancy_loss(z_tgt_all, text_prototypes,tau=0.07):
    """
    z_tgt_all: list of [B, D] (for each branch)
    text_prototypes: [C, D]
    """
    probs = []
    for z in z_tgt_all:
        sim = torch.matmul(z, text_prototypes.T) / tau  # [B, C]
        p = F.softmax(sim, dim=-1)
        probs.append(p)
    probs = torch.stack(probs, dim=0)  # [N, B, C]
    mean_p = probs.mean(dim=0, keepdim=True)       # [1, B, C]
    disc = torch.abs(probs - mean_p).mean()
    return disc



def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    计算高斯核矩阵
    source: [B, D]
    target: [B, D]
    """
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0) # [2B, D]
    
    # 计算所有样本间的L2距离: ||x-y||^2 = ||x||^2 + ||y||^2 - 2x.y
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0 - total1) ** 2).sum(2) # [2B, 2B]
    
    # 动态计算带宽 sigma
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    
    # 多核设置：带宽以 kernel_mul 为倍数变化
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    # 计算多核高斯响应并求和
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val) # [2B, 2B]

def mmd_rbf(source, target, kernel_mul=2.5, kernel_num=7, fix_sigma=None):
    """
    多核 MMD Loss (Multi-Kernel Maximum Mean Discrepancy)
    source: [B, D]
    target: [B, D]
    """
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
                              kernel_mul=kernel_mul, 
                              kernel_num=kernel_num, 
                              fix_sigma=fix_sigma)
    
    # 将核矩阵分为四块: XX, YY, XY, YX
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    # MMD = E[K(x,x)] + E[K(y,y)] - 2E[K(x,y)]
    loss = torch.mean(XX + YY - XY - YX)
    return loss


class GeometricRegularizationLoss(nn.Module):
    """
    黎曼切空间几何正则化损失

    目标：
    1. 类内紧致：同类样本在切空间中距离更小
    2. 类间分离：不同类样本中心距离更大
    """

    def __init__(self, margin=2.0, alpha=1.0, beta=1.0):
        """
        Parameters:
        -----------
        margin : float
            类间分离的最小距离阈值
        alpha : float
            类内损失权重
        beta : float
            类间损失权重
        """
        super().__init__()
        self.margin = margin
        self.alpha = alpha
        self.beta = beta

    def forward(self, features, labels):
        """
        Parameters:
        -----------
        features : torch.Tensor
            形状 (B, D)，切空间特征
        labels : torch.Tensor
            形状 (B,)，类别标签

        Returns:
        --------
        loss : torch.Tensor
            几何正则化损失
        """
        unique_labels = torch.unique(labels)
        centroids = []
        intra_loss = 0.0
        total_samples = 0

        # 计算类内紧致损失
        for c in unique_labels:
            mask = (labels == c)
            class_feat = features[mask]

            if len(class_feat) == 0:
                continue

            # 计算类中心
            centroid = class_feat.mean(dim=0)
            centroids.append(centroid)

            # 类内损失：样本到类中心的L2距离
            distances = torch.norm(class_feat - centroid.unsqueeze(0), dim=1)
            intra_loss += distances.sum()
            total_samples += len(class_feat)

        if len(centroids) == 0:
            return torch.tensor(0., device=features.device, requires_grad=True)

        # 归一化类内损失
        intra_loss = intra_loss / total_samples

        # 计算类间分离损失
        centroids = torch.stack(centroids)
        inter_loss = 0.0
        n_classes = len(centroids)

        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                # 计算类中心之间的距离
                dist = torch.norm(centroids[i] - centroids[j])
                # Hinge loss：如果距离小于margin，则产生损失
                inter_loss += torch.relu(self.margin - dist)

        # 归一化类间损失
        n_pairs = n_classes * (n_classes - 1) / 2
        if n_pairs > 0:
            inter_loss = inter_loss / n_pairs

        # 总损失
        total_loss = self.alpha * intra_loss + self.beta * inter_loss

        return total_loss

# def contrastive_alignment_loss(z, text_embeds, labels, temperature=0.25):
#     """
#     Args:
#         z: [B, D] - EEG 特征 (建议在输入前已归一化，或在函数内归一化)
#         text_embeds: [C, D] - 所有类别的文本原型 (已去中心化 + 归一化)
#         labels: [B] - 真实标签索引
#         temperature: float - 温度系数，关键参数！
#     """
 
#     # 2. 计算 Logits (余弦相似度矩阵)
#     # [B, D] @ [D, C] -> [B, C]
#     # 结果是每个样本与所有类别的相似度
#     logits = torch.matmul(z, text_embeds.T)
#     # 3. 温度缩放 (Temperature Scaling)
#     # 你的原型之间现在是负相关(-0.3)，相似度数值范围很小。
#     # 如果不除以温度，Softmax 会非常平滑（欠拟合）。
#     # 除以 0.07 会放大差异，迫使模型做出“尖锐”的判断。
#     logits = logits / temperature
#     # 4. 标准交叉熵损失
#     # 这会自动进行 Softmax，并最大化正确类别的概率，同时最小化错误类别的概率
#     loss = F.cross_entropy(logits, labels, label_smoothing=0.1)
#     return loss