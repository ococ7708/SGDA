"""
黎曼几何预处理模块
功能：直接从原始EEG信号构建SPD协方差矩阵，并映射到切空间
"""

import numpy as np
from pyriemann.estimation import Covariances  # 注意是复数
from pyriemann.tangentspace import TangentSpace


def raw_eeg_to_riemann_tangent(raw_eeg_data, metric='riemann',
                                align=True, reg=1e-6):
    """
    直接从原始EEG信号转换为黎曼切空间特征

    Parameters:
    -----------
    raw_eeg_data : np.ndarray
        形状为 (n_trials, n_channels, n_times) 的原始EEG信号
    metric : str, default='riemann'
        黎曼度量：'riemann' (AIRM) 或 'logeuclid'
    align : bool, default=True
        是否进行切空间对齐（跨被试场景推荐）
    reg : float, default=1e-6
        协方差矩阵正则化项，保证正定性

    Returns:
    --------
    tangent_feats : np.ndarray
        形状为 (n_trials, tangent_dim) 的切空间特征
    cov_matrices : np.ndarray
        形状为 (n_trials, n_channels, n_channels) 的协方差矩阵
    """
    n_trials, n_channels, n_times = raw_eeg_data.shape
    tangent_dim = n_channels * (n_channels + 1) // 2

    # Step 1: 计算协方差矩阵（使用OAS估计器保证正定性）
    cov_estimator = Covariances(estimator='oas')
    cov_matrices = cov_estimator.fit_transform(raw_eeg_data)

    # 添加正则化项保证数值稳定性
    cov_matrices = cov_matrices + reg * np.eye(n_channels)

    # Step 2: 映射到切空间（自动完成对齐）
    ts = TangentSpace(metric=metric)
    tangent_feats = ts.fit_transform(cov_matrices)

    return tangent_feats.astype(np.float32), cov_matrices


def process_de_features_to_riemann(de_features, metric='riemann',
                                    align=True, reg=1e-6):
    """
    将DE特征转换为黎曼切空间特征
    （用于SGDA中已提取DE特征的情况）

    Parameters:
    -----------
    de_features : np.ndarray
        形状为 (n_trials, n_channels, n_freq_bands) 的DE特征
    metric, align, reg : 同上

    Returns:
    --------
    tangent_feats : np.ndarray
        形状为 (n_trials, tangent_dim) 的切空间特征
    """
    n_trials, n_channels, n_bands = de_features.shape

    # 将 (C, F) 视为多变量，计算通道间协方差
    cov_matrices = np.zeros((n_trials, n_channels, n_channels))

    for i in range(n_trials):
        # de_features[i] shape: (C, F)
        cov = np.cov(de_features[i])  # shape: (C, C)
        cov = cov + reg * np.eye(n_channels)
        cov_matrices[i] = cov

    # 映射到切空间
    ts = TangentSpace(metric=metric)
    tangent_feats = ts.fit_transform(cov_matrices)

    return tangent_feats.astype(np.float32)