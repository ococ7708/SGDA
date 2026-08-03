# -*- coding: utf-8 -*-
"""
Final DREAMER EEG Preprocessing Script (Baseline / Stimuli Aware, Parallel)
=======================================================================

✔ 精确适配 DREAMER 官方 mat 结构
✔ 明确区分 baseline / stimuli
✔ baseline 仅用于 DE 层面校正（不做 LDS）
✔ stimuli 做 DE + LDS
✔ subject 级多进程并行
✔ 参数化全部预处理设置
✔ 一次处理，多次复用

最终数据结构：
all_data[session][subject][trial][sample]
  sample.shape = (T, 14, 5)
"""

import os
from pkgutil import get_data
import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt, resample
from scipy.linalg import solve_discrete_are
from joblib import Parallel, delayed
import multiprocessing

# ============================================================
# 预处理配置
# ============================================================

def get_default_preprocess_cfg():
    return dict(
        raw_fs=128,   #原始数据采样率就是128hz
        target_fs=128,
        window_sec=1.0,
        bands={
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta':  (13, 30),
            'gamma': (30, 45),
        },
        lds_alpha=0.98,
        label_threshold=3.0,
        use_baseline=True,
        eps=1e-8,
    )

# ============================================================
# 滤波器构建
# ============================================================

def build_bandpass_filters(cfg):
    filters = {}
    nyq = 0.5 * cfg['target_fs']
    for name, (l, h) in cfg['bands'].items():
        b, a = butter(4, [l / nyq, h / nyq], btype='band')
        filters[name] = (b, a)
    return filters

# ============================================================
# DE 特征
# ============================================================

def compute_de(x, eps):
    var = np.var(x, axis=0) + eps
    return 0.5 * np.log(2 * np.pi * np.e * var)


def extract_de_features(eeg, cfg):
    feats = []
    for name in cfg['bands']:
        b, a = cfg['filters'][name]
        filtered = filtfilt(b, a, eeg, axis=0)
        feats.append(compute_de(filtered, cfg['eps']))
    return np.stack(feats, axis=-1)  # (C, F)

# ============================================================
# LDS 平滑（仅 stimuli 使用）
# ============================================================

def lds_smooth(features, alpha):
    N, D = features.shape
    A = alpha * np.eye(D)
    Q = np.eye(D)
    R = np.eye(D)

    P = solve_discrete_are(A.T, np.eye(D), Q, R)
    K = P @ np.linalg.inv(P + R)

    x_hat = np.zeros_like(features)
    x_hat[0] = features[0]
    for t in range(1, N):
        x_hat[t] = A @ x_hat[t - 1] + K @ (features[t] - A @ x_hat[t - 1])
    return x_hat

# ============================================================
# EEG → DE samples (stimuli: with LDS)
# ============================================================

def eeg_to_samples_stimuli(eeg, cfg):
    if cfg['raw_fs'] != cfg['target_fs']:
        eeg = resample(eeg, int(len(eeg) * cfg['target_fs'] / cfg['raw_fs']), axis=0)

    win_size = int(cfg['window_sec'] * cfg['target_fs'])
    num_win = eeg.shape[0] // win_size
    if num_win == 0:
        return None

    feats = []
    for w in range(num_win):
        seg = eeg[w * win_size:(w + 1) * win_size]
        feats.append(extract_de_features(seg, cfg))

    feats = np.stack(feats, axis=0)  # (T, C, F)
    T, C, F = feats.shape
    feats = lds_smooth(feats.reshape(T, -1), cfg['lds_alpha'])
    return feats.reshape(T, C, F)

# ============================================================
# EEG → baseline DE mean (no LDS)
# ============================================================

def eeg_to_baseline_mean(eeg, cfg):
    if cfg['raw_fs'] != cfg['target_fs']:
        eeg = resample(eeg, int(len(eeg) * cfg['target_fs'] / cfg['raw_fs']), axis=0)

    win_size = int(cfg['window_sec'] * cfg['target_fs'])
    num_win = eeg.shape[0] // win_size
    if num_win == 0:
        return None

    feats = []
    for w in range(num_win):
        seg = eeg[w * win_size:(w + 1) * win_size]
        feats.append(extract_de_features(seg, cfg))

    feats = np.stack(feats, axis=0)
    return feats.mean(axis=0, keepdims=True)  # (1, C, F)

# ============================================================
# 单个 subject 处理（并行单元）
# ============================================================

def process_one_subject(subj, cfg):
    eeg_struct = subj['EEG'][0, 0]
    valences = subj['ScoreValence'][0, 0][:, 0]
    arousals = subj['ScoreArousal'][0, 0][:, 0]

    baseline_trials = eeg_struct['baseline'][0, 0]
    stimuli_trials = eeg_struct['stimuli'][0, 0]

    subj_trials = []
    subj_labels_val = []
    subj_labels_aro = []

    for tr in range(18):
        stim_eeg = stimuli_trials[tr, 0]
        stim_feat = eeg_to_samples_stimuli(stim_eeg, cfg)
        if stim_feat is None:
            continue

        if cfg['use_baseline']:
            base_eeg = baseline_trials[tr, 0]
            base_mean = eeg_to_baseline_mean(base_eeg, cfg)
            if base_mean is not None:
                stim_feat = stim_feat - base_mean

        subj_trials.append(stim_feat)
        subj_labels_val.append([
            int(valences[tr] >= cfg['label_threshold'])
        ])
        subj_labels_aro.append([
            int(arousals[tr] >= cfg['label_threshold'])
        ])

    return subj_trials, subj_labels_val,subj_labels_aro

# ============================================================
# 主预处理流程（并行）
# ============================================================

def preprocess_dreamer(mat_path, save_path, cfg, force_reprocess=False):
    if os.path.exists(save_path) and not force_reprocess:
        print('[INFO] Load processed data from disk')
        return np.load(save_path, allow_pickle=True).item()

    print('[INFO] Start DREAMER preprocessing (parallel)')

    mat = sio.loadmat(mat_path)
    subjects = mat['DREAMER'][0, 0]['Data'][0] # 23 subjects

    num_jobs = min(multiprocessing.cpu_count(), 8)

    results = Parallel(n_jobs=num_jobs)(
        delayed(process_one_subject)(subj, cfg)
        for subj in subjects
    )

    all_data = [[]]
    all_label_val = [[]]
    all_label_aro = [[]]

    for subj_trials, subj_labels_val, subj_labels_aro in results:
        all_data[0].append(subj_trials)
        all_label_val[0].append(subj_labels_val)
        all_label_aro[0].append(subj_labels_aro)

    save_obj = dict(all_data=all_data, all_label_val=all_label_val, all_label_aro=all_label_aro,preprocess_cfg=cfg)
    np.save(save_path, save_obj)
    print('[INFO] Finished, saved to:', save_path)
    return save_obj

# ============================================================
# Example usage
# ============================================================

def get_data(setting=None):  #实验处从此读取数据

    obj = np.load(setting.dataset_path, allow_pickle=True).item()

    all_data = obj['all_data']
    if setting.labeltype=='val':
        all_label = obj['all_label_val']
    else:
        all_label = obj['all_label_aro']

    seg_data,seg_label=segment(all_data,all_label,setting.sample_length, setting.stride)
    return seg_data,seg_label,14,5,2


def segment(data,label,sample_length,stride):
    seg_data = []
    seg_label = []
    for ses_i, session in enumerate(data):
        seg_session = []
        seg_session_label = []
        for sub_i, subject in enumerate(data[ses_i]):
            seg_sub = []
            seg_sub_label = []
            for t_i, trail in enumerate(data[ses_i][sub_i]):
                trail = np.array(trail)  # 把trial转成np数组
                trail = np.asarray(trail)
                num_sample = (len(trail) - sample_length) // stride + 1
                seg_trail = np.empty((num_sample, sample_length, 14, 5))
                seg_trail_label=np.full(num_sample,label[ses_i][sub_i][t_i][0])
                # Cutting a one-dimensional array through a sliding window to form a two-dimensional array
                for i in range(num_sample):
                    seg_trail[i] = trail[i * stride:i * stride + sample_length]
                seg_sub.append(seg_trail)
                seg_sub_label.append(seg_trail_label)
            seg_session.append(seg_sub)
            seg_session_label.append(seg_sub_label)
        seg_data.append(seg_session)
        seg_label.append(seg_session_label)
    return seg_data,seg_label


if __name__ == '__main__':
    cfg = get_default_preprocess_cfg()
    cfg['filters'] = build_bandpass_filters(cfg)#预先生成滤波器

    mat_path = '/home/dataset/DREAMER/DREAMER.mat'
    save_path = '/home/ch/datasets/DREAMER/DE_processed_1s.npy'

    preprocess_dreamer(mat_path, save_path, cfg, force_reprocess=False)
