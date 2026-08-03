import numpy as np
import torch
import torch.nn.functional as F
import os
import random
from scipy.linalg import inv, sqrtm


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # if benchmark=True, deterministic will be False
    torch.backends.cudnn.enabled = False


def zscore_subject_wise(all_data, eps=1e-6):
    """
    all_data: list[sessions][subjects][samples]
              sample shape = (T, C, F)
    return: normalized all_data (same structure)
    """

    norm_data = []

    for sess_data in all_data:                 # sessions
        sess_norm = []

        for subj_data in sess_data:            # subjects
            # subj_data: list of samples
            # stack all samples along time
            # shape: (N_samples, T, C, F)
            subj_array = np.stack(subj_data, axis=0)

            # merge sample and time dims
            # shape: (N_samples*T, C, F)
            flat = subj_array.reshape(-1, subj_array.shape[2], subj_array.shape[3])

            mean = flat.mean(axis=0, keepdims=True)   # (1, C, F)
            std  = flat.std(axis=0, keepdims=True)    # (1, C, F)

            # normalize
            subj_norm = (subj_array - mean) / (std + eps)

            # back to list of samples
            sess_norm.append([subj_norm[i] for i in range(subj_norm.shape[0])])

        norm_data.append(sess_norm)

    return norm_data



def zscore_subject_seedv(all_data, eps=1e-6):
    """
    all_data[sessions][subjects][samples] -> np.ndarray (T, D)
    D = C * F
    """
    all_data_z = []

    for session in all_data:
        session_z = []

        for subj_samples in session:
            # subj_samples: list of (T, D)
            subj_samples = [np.asarray(x) for x in subj_samples]

            # (N, T, D)
            subj_array = np.stack(subj_samples, axis=0)

            N, T, D = subj_array.shape

            # subject 内统计 (samples × time)
            flat = subj_array.reshape(-1, D)
            mean = flat.mean(axis=0, keepdims=True)
            std  = flat.std(axis=0, keepdims=True)

            flat_z = (flat - mean) / (std + eps)
            subj_array_z = flat_z.reshape(N, T, D)

            # 还原为 list，保持原始结构
            subj_samples_z = [subj_array_z[i] for i in range(N)]

            session_z.append(subj_samples_z)

        all_data_z.append(session_z)

    return all_data_z




def apply_euclidean_alignment(all_data):
    """
    对嵌套结构的 EEG DE 特征进行欧几里得对齐 (EA)
    输入格式: all_data[session_idx][subject_idx][sample_idx] -> shape (3, 32, 5)
    输出格式: 同样结构的 aligned_data
    """
    aligned_data = []
    
    # 遍历 session
    for s_idx, session in enumerate(all_data):
        session_aligned = []
        
        # 遍历 subject
        for sub_idx, subject_samples in enumerate(session):
            # 1. 转换格式以计算协方差: [N, 3, 32, 5] -> [N, 32, 15] 
            # 将时间步和频段合并，只保留通道作为协方差维度
            samples_np = np.array(subject_samples) # Shape: (N, 3, 32, 5)
            N, T, C, F = samples_np.shape
            
            # 变形为 (N, C, T*F) 以便计算通道间的协方差
            # 这里的目的是看 32 个通道之间的相互关系
            reshaped_samples = samples_np.transpose(0, 2, 1, 3).reshape(N, C, T * F)
            
            # 2. 计算该被试的平均协方差矩阵 R
            # R = (1/N) * sum(X * X.T)
            R = np.zeros((C, C))
            for i in range(N):
                # 矩阵与其转置相乘得到 C*C 矩阵
                R += np.dot(reshaped_samples[i], reshaped_samples[i].T)
            R /= N
            
            # 3. 计算 R 的负二分之一次方 (R^-1/2)
            # 这是对齐变换矩阵
            # 使用 sqrtm 计算矩阵平方根，inv 计算逆
            R_inv_sqrt = inv(sqrtm(R)).real # 取实部防止复数噪声
            
            # 4. 对该被试的所有样本进行变换
            # X_aligned = R^-1/2 * X
            sub_aligned = []
            for i in range(N):
                # 对每个样本的 32 个通道进行线性变换
                # 保持原始维度 (3, 32, 5) 返回
                sample_to_align = reshaped_samples[i] # (32, 15)
                aligned_sample = np.dot(R_inv_sqrt, sample_to_align) # (32, 15)
                
                # 还原回 (3, 32, 5)
                aligned_sample = aligned_sample.reshape(C, T, F).transpose(1, 0, 2)
                sub_aligned.append(aligned_sample)
                
            session_aligned.append(sub_aligned)
            print(f"Session {s_idx} Subject {sub_idx} EA 对齐完成")
            
        aligned_data.append(session_aligned)
        
    return aligned_data

# 使用示例:
# aligned_all_data = apply_euclidean_alignment(all_data)


def global_normalization_after_ea(aligned_all_data):
    """
    在 EA 之后进行全局 Z-score 归一化
    输入: aligned_all_data[sessions][subjects][samples] -> (3, 32, 5)
    """
    # 1. 将所有样本收集到一个大数组里，方便算均值和标准差
    all_samples_list = []
    for session in aligned_all_data:
        for subject in session:
            for sample in subject:
                all_samples_list.append(sample)
    
    # 转换为 numpy 数组: shape (Total_N, 3, 32, 5)
    all_samples_np = np.array(all_samples_list)
    
    # 2. 计算全局均值和标准差
    # 我们针对 (通道, 频段) 计算，即在 (Total_N, 时间步) 这两个维度上取平均
    # 结果 shape 将会是 (1, 1, 32, 5)
    global_mean = np.mean(all_samples_np, axis=(0, 1), keepdims=True)
    global_std = np.std(all_samples_np, axis=(0, 1), keepdims=True)
    
    # 防止除以 0
    global_std[global_std == 0] = 1.0
    
    # 3. 应用归一化
    normalized_data = []
    for session in aligned_all_data:
        session_norm = []
        for subject in session:
            # 这里的 subject 是一个样本列表
            sub_np = np.array(subject) # (N, 3, 32, 5)
            # 广播机制自动应用 global_mean 和 global_std
            sub_norm_np = (sub_np - global_mean) / global_std
            session_norm.append(list(sub_norm_np))
        normalized_data.append(session_norm)
        
    print("全局归一化完成！")
    return normalized_data, (global_mean, global_std)

# 集成调用示例：
# 1. 先做 EA
# aligned_data = apply_euclidean_alignment(all_data)
# 2. 再做全局归一化
# final_data, stats = global_normalization_after_ea(aligned_data)


def flatten_trials(data, label):
    """
    输入:
        data[s][sub][trial][sample]
        label[s][sub][trial][sample]
    输出:
        new_data[s][sub][sample]
        new_label[s][sub][sample]
    """
    new_data = []
    new_label = []

    num_sessions = len(data)

    for s in range(num_sessions):
        session_data = []
        session_label = []

        num_subjects = len(data[s])

        for sub in range(num_subjects):
            # 展平 trials：不关心 sample 的 shape
            flat_data = np.concatenate(data[s][sub], axis=0)
            flat_label = np.concatenate(label[s][sub], axis=0)

            session_data.append(flat_data)
            session_label.append(flat_label)

        new_data.append(session_data)
        new_label.append(session_label)

    return new_data, new_label



def get_preds(
    model,
    dataloader,
    text_prototypes,
    source_centroids,
    device,
    fusion_type='feature',
    tau=0.5
):
    """
    Sample-wise 分布距离加权的统一评估函数

    Args:
        fusion_type:
            - 'feature': 特征级融合
            - 'decision': 决策级融合
        source_centroids: [n_sources, embed_dim]
    """

    model.eval()
    all_y_true = []
    all_y_pred = []

    if source_centroids is None:
        raise ValueError("source_centroids 不能为空")

    source_centroids = source_centroids.to(device)  # [n_sources, D]

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)

        # ---------------------------------------------------
        # 1. 前向传播
        # ---------------------------------------------------
        _, z_tgt_all = model([], x_batch)
        # list -> [n_sources, B, D]
        z_tgt_stack = torch.stack(z_tgt_all, dim=0)

        # ---------------------------------------------------
        # 2. Sample-wise 权重计算（核心修改）
        # ---------------------------------------------------
        # dists: [n_sources, B]
        dists = torch.norm(
            z_tgt_stack - source_centroids.unsqueeze(1),
            p=2,
            dim=-1
        )

        # softmax over source dimension → 每个样本一组权重
        weights = F.softmax(-dists / tau, dim=0)  # [n_sources, B]

        # reshape for broadcasting
        weights = weights.unsqueeze(-1)  # [n_sources, B, 1]

        # ---------------------------------------------------
        # 3. 融合策略
        # ---------------------------------------------------
        if fusion_type == 'feature':
            # 特征级融合
            z_fused = torch.sum(z_tgt_stack * weights, dim=0)  # [B, D]
            z_final = F.normalize(z_fused, dim=-1)
            final_logits = torch.matmul(z_final, text_prototypes.T)

        elif fusion_type == 'decision':
            # 决策级融合
            logits_all = torch.matmul(z_tgt_stack, text_prototypes.T)  # [S, B, C]
            probs_all = F.softmax(logits_all, dim=-1)
            final_probs = torch.sum(probs_all * weights, dim=0)  # [B, C]
            final_logits = final_probs

        else:
            raise ValueError(f"不支持的 fusion_type: {fusion_type}")

        # ---------------------------------------------------
        # 4. 预测
        # ---------------------------------------------------
        preds = final_logits.argmax(dim=-1)

        all_y_true.append(y_batch.cpu().numpy())
        all_y_pred.append(preds.cpu().numpy())

    return np.concatenate(all_y_true), np.concatenate(all_y_pred)


def get_preds_avg(model, dataloader, text_prototypes, device, fusion_type='feature'):
    """
    平均融合策略（无任何冗余计算）
    """
    model.eval()
    all_y_true = []
    all_y_pred = []

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)

        # ---------------------------------------------------
        # 1. 前向传播：获取所有分支特征
        # ---------------------------------------------------
        _, z_tgt_all = model([], x_batch)
        z_tgt_stack = torch.stack(z_tgt_all, dim=0)  # [n_sources, B, D]

        # ---------------------------------------------------
        # 2. 融合
        # ---------------------------------------------------
        if fusion_type == 'feature':
            # === 特征平均 ===
            z_final = torch.mean(z_tgt_stack, dim=0)
            z_final = F.normalize(z_final, dim=-1)
            final_logits = torch.matmul(z_final, text_prototypes.T)

        elif fusion_type == 'decision':
            # === 概率平均 ===
            logits_all = torch.matmul(z_tgt_stack, text_prototypes.T)
            probs_all = F.softmax(logits_all, dim=-1)
            final_logits = torch.mean(probs_all, dim=0)

        else:
            raise ValueError(f"不支持的融合类型: {fusion_type}")

        # ---------------------------------------------------
        # 3. 预测
        # ---------------------------------------------------
        preds = final_logits.argmax(dim=-1)

        all_y_true.append(y_batch.cpu().numpy())
        all_y_pred.append(preds.cpu().numpy())

    return np.concatenate(all_y_true), np.concatenate(all_y_pred)