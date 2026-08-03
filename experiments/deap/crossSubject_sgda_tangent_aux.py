"""
SGDA + Tangent Geometry Auxiliary Regularization
==================================================
保留原始 SGDA 完整结构（Conv2D SFE + DSFE + semantic prototype），
额外从 DE 特征构造 SPD → 切空间向量，通过轻量投影层约束 SFE 输出。

设计原则：
  1. 主分支完全不动 —— DE → Conv2D SFE → DSFE → semantic prototype
  2. 辅助分支 —— 同一样本 DE → SPD → tangent (528) → Linear投影 (512)
  3. L_geo_aux = MSE(SFE_out, projected_tangent)，λ = 0.001
  4. 评估时不用 tangent 特征，完全复用基线 get_preds

对照基线: crossSubjects_deap.py (Acc 65.81%)
"""

import os
import sys
import csv
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader

# ===== 路径处理 =====
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# ===== 数据与工具 =====
from data_utils.load_data import get_data
from data_utils.text_to_vector import label_to_vector
from data_utils.constants.path_mapper import path_mapper

from utils.args import get_args_parser
from config.setting import preset_setting, set_setting_by_args

from utils.mix_utils import (
    setup_seed,
    flatten_trials,
    get_preds,
)

from utils.log_utils import save_csubs_results_csv
from utils.loss import align_loss, mmd_linear, discrepancy_loss

# ===== 模型（Conv2D SFE，同基线） =====
from models.model import SGDA_model


def _format_float_for_path(value):
    return str(value).replace(".", "p").replace("-", "m")


def _std(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _append_csv_row(path, fieldnames, row):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _write_run_config(path, config):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def _get_optional_int(args, name):
    return getattr(args, name, None)


def _limit_subjects_and_trials(data, label, max_subjects=None, max_trials=None):
    if max_subjects is not None:
        data = [session[:max_subjects] for session in data]
        label = [session[:max_subjects] for session in label]

    if max_trials is not None:
        data = [
            [subject[:max_trials] for subject in session]
            for session in data
        ]
        label = [
            [subject[:max_trials] for subject in session]
            for session in label
        ]

    return data, label


def _limit_samples_per_subject(data, label, tangent_data, max_samples=None):
    if max_samples is None:
        return data, label, tangent_data

    limited_data = []
    limited_label = []
    limited_tangent = []

    for session_data, session_label, session_tangent in zip(data, label, tangent_data):
        out_session_data = []
        out_session_label = []
        out_session_tangent = []

        for subject_data, subject_label, subject_tangent in zip(
            session_data,
            session_label,
            session_tangent
        ):
            keep = min(max_samples, len(subject_data))
            out_session_data.append(subject_data[:keep])
            out_session_label.append(subject_label[:keep])
            out_session_tangent.append(subject_tangent[:keep])

        limited_data.append(out_session_data)
        limited_label.append(out_session_label)
        limited_tangent.append(out_session_tangent)

    return limited_data, limited_label, limited_tangent


# ============================================================
# 1. DE 特征 → 切空间向量（辅助分支用）
# ============================================================
def compute_tangent_features(all_data, metric='riemann', reg=1e-6):
    """
    为每个 DE 样本构造 SPD 协方差矩阵并映射到切空间。

    每个样本 (T, C, F) → reshape (C, T*F) → OAS-SPD → log-map → tangent (D)
    按被试独立拟合 TangentSpace，保证局部近似质量最优。

    Returns:
        tangent_data: 与 all_data 平行的嵌套列表，每个样本变为 1D tangent 向量
        tangent_dim: 切空间维度
    """
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    n_sessions = len(all_data)
    n_subjects = len(all_data[0])

    print(f"[Tangent] 为 {n_sessions} session × {n_subjects} subjects 计算切空间特征...")

    tangent_data = [[None for _ in range(n_subjects)] for _ in range(n_sessions)]

    for ses_idx in range(n_sessions):
        for sub_idx in range(n_subjects):
            trials = all_data[ses_idx][sub_idx]
            if len(trials) == 0:
                continue

            # 展平该被试所有样本
            flat_samples = []
            trial_n_samples = []
            for trial in trials:
                trial_n_samples.append(len(trial))
                for s in trial:
                    T, C, F = s.shape
                    flat_samples.append(s.reshape(C, T * F))

            stacked = np.stack(flat_samples, axis=0)          # (N_total, C, T*F)

            # OAS 收缩估计 → SPD
            cov_estimator = Covariances(estimator='oas')
            covs = cov_estimator.fit_transform(stacked)        # (N_total, C, C)
            covs = covs + reg * np.eye(C)

            # 被试独立拟合切空间
            ts = TangentSpace(metric=metric)
            tangents = ts.fit_transform(covs).astype(np.float32)  # (N_total, tangent_dim)

            # 被试内 z-score
            mean = tangents.mean(axis=0, keepdims=True)
            std = tangents.std(axis=0, keepdims=True) + 1e-6
            tangents = (tangents - mean) / std

            # 按 trial 拆分回嵌套结构
            offset = 0
            trial_tangents = []
            for n_s in trial_n_samples:
                trial_tangents.append([tangents[offset + i] for i in range(n_s)])
                offset += n_s

            tangent_data[ses_idx][sub_idx] = trial_tangents

    tangent_dim = tangent_data[0][0][0][0].shape[0]
    print(f"[Tangent] 完成, 切空间维度: {tangent_dim}")
    return tangent_data, tangent_dim


# ============================================================
# 2. 数据加载（DE 特征 + 切空间特征）
# ============================================================
def load_data_with_tangent(args, device):
    """加载 DE 数据并计算对应的切空间特征"""
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setting.dataset_path = path_mapper['deap']
    setting.dataset = 'deap'
    setting.experiment_mode = 'subject-independent'
    setting.onehot = False
    setting.label_used = ['valence']
    setting.only_seg = False
    setting.bounds = [5, 5.0001]
    setting.sessions = [1]
    setting.sample_length = 3
    setting.stride = 1

    # 读取 DE 特征
    data, label, channels, num_freq_bands, num_classes = get_data(setting)

    data, label = _limit_subjects_and_trials(
        data,
        label,
        max_subjects=_get_optional_int(args, "max_subjects"),
        max_trials=_get_optional_int(args, "max_trials")
    )

    # 计算切空间特征（在被试内展平前）
    tangent_data, tangent_dim = compute_tangent_features(data)

    # 展平 trial 维度。tangent_data 必须使用原始 trial 结构的 label 做对齐；
    # 如果先 flatten label，再传给 tangent_data，会造成嵌套结构错配。
    tangent_data, _ = flatten_trials(tangent_data, label)
    data, label = flatten_trials(data, label)
    data, label, tangent_data = _limit_samples_per_subject(
        data,
        label,
        tangent_data,
        max_samples=_get_optional_int(args, "max_samples_per_subject")
    )

    # 文本原型
    text_dim, allClassLabelvector = label_to_vector(
        dataset=setting.dataset,
        LM=args.LM,
        LabelTextMapper=None,
        device=device
    )

    return (
        data, tangent_data, label, allClassLabelvector,
        channels, num_freq_bands, text_dim, tangent_dim
    )


# ============================================================
# 3. 切空间投影器（轻量，仅作辅助正则）
# ============================================================
class TangentProjector(nn.Module):
    """将切空间向量线性投影到 SFE 输出维度"""
    def __init__(self, tangent_dim, sfe_dim):
        super().__init__()
        self.proj = nn.Linear(tangent_dim, sfe_dim)

    def forward(self, x):
        return self.proj(x)


# ============================================================
# 4. 主实验
# ============================================================
def run_experiment(args):
    """主实验流程（LOSO 跨被试）"""

    # ===== 基础配置 =====
    setup_seed(args.seed)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.LM = "clip"
    args.dataset = "deap"

    args.epochs = 30 if args.epochs is None else args.epochs
    args.batch_size = 64 if args.batch_size is None else args.batch_size
    args.lr = 1e-3 if args.lr is None else args.lr
    aux_lambda = args.aux_lambda  # ★ 切空间辅助损失权重（极小）

    args.experiment_name = 'results_deap_tangent_aux'
    args.output_dir = os.path.join(project_root, f"results/{args.experiment_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = (
        f"ep{args.epochs}_bs{args.batch_size}_lr{_format_float_for_path(args.lr)}_"
        f"lambda{_format_float_for_path(aux_lambda)}_seed{args.seed}_{timestamp}"
    )
    run_dir = os.path.join(args.output_dir, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)
    epoch_log_path = os.path.join(run_dir, "epoch_log.csv")
    summary_path = os.path.join(args.output_dir, "summary_tangent_aux_runs.csv")
    config_path = os.path.join(run_dir, "run_config.json")
    run_config = {
        "run_id": run_id,
        "script": os.path.abspath(__file__),
        "dataset": args.dataset,
        "label_used": ["valence"],
        "lm": args.LM,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "aux_lambda": aux_lambda,
        "sfe_dim": 512,
        "bd_dim": 128,
        "max_subjects": _get_optional_int(args, "max_subjects"),
        "max_trials": _get_optional_int(args, "max_trials"),
        "max_samples_per_subject": _get_optional_int(args, "max_samples_per_subject"),
        "created_at": timestamp,
    }
    _write_run_config(config_path, run_config)
    print(f"Run directory: {run_dir}")

    print(f"实验输出目录: {args.output_dir}")
    print(f"辅助损失 λ = {aux_lambda}")

    sfe_dim = 512
    bd_dim = 128

    # ===== 数据准备 =====
    (all_data, all_tangent, all_label, allClassLabelvector,
     num_electrodes, num_freq_bands, text_dim, tangent_dim) = load_data_with_tangent(args, args.device)

    print(f"DE 特征形状示例: {all_data[0][0].shape}")
    print(f"切空间特征形状示例: {all_tangent[0][0].shape}")
    print(f"切空间维度: {tangent_dim}, SFE 维度: {sfe_dim}")

    # 文本原型
    run_config.update({
        "num_sessions": len(all_data),
        "num_subjects": len(all_data[0]),
        "num_electrodes": num_electrodes,
        "num_freq_bands": num_freq_bands,
        "text_dim": text_dim,
        "tangent_dim": tangent_dim,
    })
    _write_run_config(config_path, run_config)

    sorted_indices = sorted(allClassLabelvector.keys())
    vectors_in_order = [allClassLabelvector[i] for i in sorted_indices]
    text_prototypes = torch.tensor(
        np.array(vectors_in_order, dtype=np.float32)
    ).to(args.device)

    n_sessions = len(all_data)
    n_subjects = len(all_data[0])

    results_acc, results_macrof1, results_microf1 = {}, {}, {}
    epoch_log_fields = [
        "run_id", "session_idx", "target_subject", "epoch",
        "loss", "cls", "mmd", "disc", "aux",
        "acc", "macro_f1", "micro_f1", "best_acc",
        "alpha", "beta", "aux_lambda", "epochs", "batch_size", "lr", "seed",
    ]

    # ===== 按 session 做 LOSO =====
    for session_idx in range(n_sessions):
        results_acc_session = []
        results_macrof1_session = []
        results_microf1_session = []

        for target_sub in range(n_subjects):
            setup_seed(args.seed)

            print(f"\n{'='*60}")
            print(f"Session {session_idx} | Target subject {target_sub+1}/{n_subjects}")
            print(f"{'='*60}")

            source_ids = [s for s in range(n_subjects) if s != target_sub]

            # ===== 构建 source dataloader（同时返回 DE + tangent） =====
            source_loaders = []
            for sid in source_ids:
                X_de = torch.tensor(np.array(all_data[session_idx][sid]), dtype=torch.float32)
                X_tg = torch.tensor(np.array(all_tangent[session_idx][sid]), dtype=torch.float32)
                y = torch.tensor(np.array(all_label[session_idx][sid]), dtype=torch.long)

                ds = TensorDataset(X_de, X_tg, y)
                dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                drop_last=False, num_workers=0, pin_memory=True)
                source_loaders.append(dl)

            # ===== target 数据 =====
            X_tgt_de = torch.tensor(np.array(all_data[session_idx][target_sub]), dtype=torch.float32)
            X_tgt_tg = torch.tensor(np.array(all_tangent[session_idx][target_sub]), dtype=torch.float32)
            y_tgt_all = torch.tensor(np.array(all_label[session_idx][target_sub]), dtype=torch.long)

            ds_tgt = TensorDataset(X_tgt_de, X_tgt_tg, y_tgt_all)

            dl_tgt_train = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True,
                                      drop_last=False, num_workers=0, pin_memory=True)
            dl_tgt_eval = DataLoader(
                TensorDataset(X_tgt_de, y_tgt_all),   # 评估只用 DE 特征
                batch_size=args.batch_size, shuffle=False,
                num_workers=0, pin_memory=True
            )

            # ===== 模型 =====
            n_sources = len(source_loaders)

            model = SGDA_model(
                n_sources,
                num_electrodes,
                eeg_dim=sfe_dim,
                bottleneck_dim=bd_dim,
                text_dim=text_dim,
                dropout=0.1
            ).to(args.device)

            # 切空间投影器（轻量）
            tangent_projector = TangentProjector(tangent_dim, sfe_dim).to(args.device)

            optimizer = optim.Adam(
                list(model.parameters()) + list(tangent_projector.parameters()),
                lr=args.lr,
                weight_decay=1e-4
            )

            # ===== 训练步数 =====
            steps_per_epoch = min(len(dl) for dl in source_loaders)
            total_steps = args.epochs * steps_per_epoch

            print(f"epochs={args.epochs}, steps/epoch={steps_per_epoch}")

            tgt_iter = iter(dl_tgt_train)
            source_iters = [iter(dl) for dl in source_loaders]

            step = 0
            best_acc = 0
            best_macro_f1 = 0
            best_micro_f1 = 0

            # ===== 训练 =====
            for epoch in range(args.epochs):
                model.train()
                tangent_projector.train()
                epoch_loss = 0.0
                epoch_aux_loss = 0.0

                for _ in range(steps_per_epoch):
                    step += 1

                    # 动态权重（同基线）
                    gamma = 2.0 / (1.0 + np.exp(-10.0 * step / total_steps)) - 1.0
                    alpha = float(gamma)
                    beta = float(gamma) / 100

                    # ---- source batch ----
                    x_src_list, y_src_list = [], []
                    tangent_src_list = []

                    for i, src_iter in enumerate(source_iters):
                        try:
                            xb_de, xb_tg, yb = next(src_iter)
                        except StopIteration:
                            source_iters[i] = iter(source_loaders[i])
                            xb_de, xb_tg, yb = next(source_iters[i])

                        x_src_list.append(xb_de.to(args.device))
                        y_src_list.append(yb.to(args.device))
                        tangent_src_list.append(xb_tg.to(args.device))

                    # ---- target batch ----
                    try:
                        x_tgt_de, x_tgt_tg, _ = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(dl_tgt_train)
                        x_tgt_de, x_tgt_tg, _ = next(tgt_iter)

                    x_tgt = x_tgt_de.to(args.device)
                    x_tgt_tg = x_tgt_tg.to(args.device)

                    optimizer.zero_grad()

                    # 主分支前向（同基线）
                    z_src_all, z_tgt_all, sfe_src_list, _ = model(
                        x_src_list,
                        x_tgt,
                        return_sfe_features=True
                    )

                    # ---- 主损失（同基线） ----
                    total_cls, total_mmd = 0.0, 0.0

                    for i in range(n_sources):
                        cls = align_loss(z_src_all[i], text_prototypes, y_src_list[i])
                        mmd = mmd_linear(z_src_all[i], z_tgt_all[i])
                        total_cls += cls
                        total_mmd += mmd

                    total_cls /= n_sources
                    total_mmd /= n_sources
                    disc = discrepancy_loss(z_tgt_all, text_prototypes)

                    main_loss = total_cls + alpha * total_mmd + beta * disc

                    # ---- 辅助几何损失（新增） ----
                    # 投影切空间特征 → SFE 维度，与 SFE 输出做 MSE
                    L_aux = 0.0
                    for i in range(n_sources):
                        tangent_proj = tangent_projector(tangent_src_list[i])  # [B, 512]
                        L_aux += F.mse_loss(sfe_src_list[i], tangent_proj)
                    L_aux /= n_sources

                    loss = main_loss + aux_lambda * L_aux

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_aux_loss += L_aux.item()

                # ===== 打印 =====
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch + 1:3d} | "
                        f"Loss={epoch_loss / steps_per_epoch:.4f} "
                        f"CLS={total_cls:.4f} "
                        f"MMD={total_mmd:.6f} "
                        f"DISC={disc:.6f} "
                        f"AUX={epoch_aux_loss / steps_per_epoch:.6f}"
                    )

                # ===== 计算 source centroid（在语义空间，同基线） =====
                model.eval()
                source_centroids = []

                with torch.no_grad():
                    for i in range(n_sources):
                        feats = []
                        for xb_de, _, _ in source_loaders[i]:
                            f = model.sfe(xb_de.to(args.device))
                            f = model.branches[i](f)
                            feats.append(f)
                        centroid = torch.cat(feats, dim=0).mean(dim=0)
                        source_centroids.append(centroid)

                source_centroids_tensor = torch.stack(source_centroids, dim=0)

                # ===== 评估（完全同基线 get_preds） =====
                y_true, y_pred = get_preds(
                    model,
                    dl_tgt_eval,
                    text_prototypes,
                    source_centroids=source_centroids_tensor,
                    device=args.device,
                    fusion_type='feature',
                    tau=1.0
                )

                test_acc = float((y_true == y_pred).mean())
                macro_f1 = float(f1_score(y_true, y_pred, average='macro'))
                micro_f1 = float(f1_score(y_true, y_pred, average='micro'))

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_macro_f1 = macro_f1
                    best_micro_f1 = micro_f1

                    print(f"  >> New best acc: {best_acc:.4f} (epoch {epoch+1})")

                _append_csv_row(
                    epoch_log_path,
                    epoch_log_fields,
                    {
                        "run_id": run_id,
                        "session_idx": session_idx,
                        "target_subject": target_sub,
                        "epoch": epoch + 1,
                        "loss": epoch_loss / steps_per_epoch,
                        "cls": float(total_cls.detach().cpu()),
                        "mmd": float(total_mmd.detach().cpu()),
                        "disc": float(disc.detach().cpu()),
                        "aux": epoch_aux_loss / steps_per_epoch,
                        "acc": test_acc,
                        "macro_f1": macro_f1,
                        "micro_f1": micro_f1,
                        "best_acc": best_acc,
                        "alpha": alpha,
                        "beta": beta,
                        "aux_lambda": aux_lambda,
                        "epochs": args.epochs,
                        "batch_size": args.batch_size,
                        "lr": args.lr,
                        "seed": args.seed,
                    }
                )

                if best_acc == 1:
                    print("Early stop (perfect accuracy)")
                    break

            print(f"Final acc (target {target_sub+1}): {best_acc * 100:.2f}%")

            results_acc_session.append(best_acc)
            results_macrof1_session.append(best_macro_f1)
            results_microf1_session.append(best_micro_f1)

        results_acc[session_idx] = results_acc_session
        results_macrof1[session_idx] = results_macrof1_session
        results_microf1[session_idx] = results_microf1_session

    # ===== 汇总结果 =====
    all_results_acc = []
    for key in results_acc:
        all_results_acc.extend(results_acc[key])
    all_results_acc = np.array(all_results_acc)

    print(f"\n{'='*60}")
    print(f"Overall Acc: {all_results_acc.mean() * 100:.2f}% "
          f"+/- {all_results_acc.std() * 100:.2f}%")
    print(f"{'='*60}")

    # ===== 保存 =====
    out_path = os.path.join(
        run_dir,
        f"subject_results_deap_tangent_aux_valence_ep{args.epochs}_bd{bd_dim}_bs{args.batch_size}_lambda{aux_lambda}.csv"
    )

    save_csubs_results_csv(
        results_acc,
        results_macrof1,
        results_microf1,
        out_path
    )

    all_results_macro = []
    all_results_micro = []
    for key in results_macrof1:
        all_results_macro.extend(results_macrof1[key])
        all_results_micro.extend(results_microf1[key])
    all_results_macro = np.array(all_results_macro)
    all_results_micro = np.array(all_results_micro)

    summary_row = {
        "run_id": run_id,
        "created_at": timestamp,
        "dataset": args.dataset,
        "label": "valence",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "aux_lambda": aux_lambda,
        "bd_dim": bd_dim,
        "sfe_dim": sfe_dim,
        "tangent_dim": tangent_dim,
        "num_subjects": n_subjects,
        "max_subjects": _get_optional_int(args, "max_subjects"),
        "max_trials": _get_optional_int(args, "max_trials"),
        "max_samples_per_subject": _get_optional_int(args, "max_samples_per_subject"),
        "acc_mean": float(all_results_acc.mean()),
        "acc_std": _std(all_results_acc),
        "macro_f1_mean": float(all_results_macro.mean()),
        "macro_f1_std": _std(all_results_macro),
        "micro_f1_mean": float(all_results_micro.mean()),
        "micro_f1_std": _std(all_results_micro),
        "subject_results_csv": out_path,
        "epoch_log_csv": epoch_log_path,
        "run_config_json": config_path,
    }
    _append_csv_row(summary_path, list(summary_row.keys()), summary_row)

    print(f"Results saved to {out_path}")
    print(f"Summary appended to {summary_path}")


if __name__ == '__main__':
    parser = get_args_parser()
    parser.set_defaults(epochs=200, batch_size=64, lr=1e-3)
    parser.add_argument('--aux_lambda', type=float, default=0.01,
                        help='weight of SPD/tangent auxiliary regularization')
    parser.add_argument('--max_subjects', type=int, default=None,
                        help='debug only: limit the number of DEAP subjects')
    parser.add_argument('--max_trials', type=int, default=None,
                        help='debug only: limit the number of trials per subject')
    parser.add_argument('--max_samples_per_subject', type=int, default=None,
                        help='debug only: limit flattened samples per subject')
    args = parser.parse_args()
    run_experiment(args)
