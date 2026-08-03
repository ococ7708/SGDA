"""
DE-Riemann-SGDA：在 DE 频谱特征上构建 SPD 协方差 + 切空间黎曼表征
=================================================================

与 crossSubject_reiman_deap.py 的核心差异：

  旧版：raw EEG(32,7680) → np.cov → 单个 SPD → tangent(528)
        → 5 个频段信息完全丢失，仅保留空域协方差

  新版：raw EEG → bandpass(5频段) → DE 提取 → 分段(3,32,5)
        → reshape(32,15) → OAS-SPD → tangent(528)
        → 5 个频段 × 3 个时间步全部编码进 SPD 结构

对比维度：
  - 旧版每被试 40 个切空间样本（每 trial 1 个）
  - 新版每被试 ~2320 个切空间样本（每滑动窗口 1 个）

对照申报书技术路线：
  欧氏基线 → SGDA基线 → 纯黎曼几何基线 → 切空间语义对齐 → 完整Riemann-SGDA
  本脚本实现后三个阶段，且 SPD 构建在 DE 频谱特征之上
"""

import os
import sys
import torch
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
    get_preds_avg,
)

from utils.log_utils import save_csubs_results_csv
from utils.loss import align_loss, mmd_linear, discrepancy_loss
from utils.loss import GeometricRegularizationLoss

# ===== 模型（MLP SFE，与旧版相同） =====
from models.model_riemann import SGDA_model


# ============================================================
# 1. DE 特征 → 黎曼切空间转换（核心新增）
# ============================================================
def convert_de_to_tangent(all_data, metric='riemann', reg=1e-6):
    """
    将 DE 特征样本批量转换为黎曼切空间向量。

    每个样本 shape (T, C, F)，如 (3, 32, 5)：
      1. reshape → (C, T*F) 即 (32, 15)，合并时间步和频段
      2. 用 OAS 收缩估计计算 C×C SPD 协方差矩阵
      3. Log-map 映射到切空间 → (C*(C+1)//2,) 即 (528,)

    按被试批量处理（pyriemann 批处理），避免逐样本循环。
    """
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    n_sessions = len(all_data)
    n_subjects = len(all_data[0])

    print(f"[DE→Tangent] 开始转换，{n_sessions} session × {n_subjects} subjects")

    for ses_idx in range(n_sessions):
        for sub_idx in range(n_subjects):
            trials = all_data[ses_idx][sub_idx]
            if len(trials) == 0:
                continue

            # 展平该被试所有 trial 的样本，记录每个 trial 的样本数
            flat_samples = []
            trial_n_samples = []
            for trial in trials:
                trial_n_samples.append(len(trial))
                for s in trial:
                    T, C, F = s.shape
                    flat_samples.append(s.reshape(C, T * F))

            stacked = np.stack(flat_samples, axis=0)      # (N_total, C, T*F)

            # OAS 收缩 → 正定 SPD
            cov_estimator = Covariances(estimator='oas')
            covs = cov_estimator.fit_transform(stacked)     # (N_total, C, C)
            covs = covs + reg * np.eye(C)

            # Log-map → 切空间
            ts = TangentSpace(metric=metric)
            tangent_all = ts.fit_transform(covs).astype(np.float32)  # (N_total, tangent_dim)

            # 按 trial 拆分回嵌套结构
            offset = 0
            for t_idx, n_s in enumerate(trial_n_samples):
                all_data[ses_idx][sub_idx][t_idx] = [
                    tangent_all[offset + i] for i in range(n_s)
                ]
                offset += n_s

    tangent_dim = all_data[0][0][0][0].shape[0]
    print(f"[DE→Tangent] 转换完成，切空间维度: {tangent_dim}")
    return all_data, tangent_dim


# ============================================================
# 2. 数据加载（正常 DE 预处理 + 黎曼切空间转换）
# ============================================================
def load_data_de_riemann(args, device, riemann_metric='riemann'):
    """加载数据：DE 预处理 → 切空间转换"""
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setting.dataset_path = path_mapper['deap']
    setting.dataset = 'deap'
    setting.experiment_mode = 'subject-independent'
    setting.onehot = False
    setting.label_used = ['valence']          # arousal 时改为 ['arousal']
    setting.bounds = [5, 5.0001]
    setting.sessions = [1]
    setting.sample_length = 3
    setting.stride = 1
    setting.only_seg = False                  # 走完整 DE 预处理流水线

    # ---- 正常 DE 预处理（bandpass + DE 提取 + 分段） ----
    data, label, channels, num_freq_bands, num_classes = get_data(setting)

    print(f"DE 预处理后 — 通道数: {channels}, 频段数: {num_freq_bands}, 类别数: {num_classes}")
    print(f"样本示例 shape: {data[0][0][0][0].shape}  (T, C, F)")

    # ---- DE 样本 → 黎曼切空间 ----
    data, tangent_dim = convert_de_to_tangent(data, metric=riemann_metric)

    # ---- 文本原型 ----
    text_dim, allClassLabelvector = label_to_vector(
        dataset=setting.dataset,
        LM=args.LM,
        LabelTextMapper=None,
        device=device
    )

    # ---- 展平 trial ----
    data, label = flatten_trials(data, label)

    # ---- 被试级 z-score（切空间向量版） ----
    for s_idx in range(len(data)):
        for sub_idx in range(len(data[s_idx])):
            arr = data[s_idx][sub_idx]           # (N, tangent_dim)
            mean = arr.mean(axis=0, keepdims=True)
            std = arr.std(axis=0, keepdims=True) + 1e-6
            data[s_idx][sub_idx] = (arr - mean) / std

    print(f"展平 + z-score 后每被试样本数示例: {data[0][0].shape}")

    return (
        data, label, allClassLabelvector,
        tangent_dim, num_classes, text_dim, channels
    )


# ============================================================
# 3. 切空间自适应加权融合评估
# ============================================================
def evaluate_with_tangent_fusion(model, dataloader, text_prototypes,
                                  sfe_centroids, device, tau=1.0):
    """基于切空间距离的样本级自适应加权融合（与旧版相同）"""
    model.eval()
    all_y_true, all_y_pred = [], []

    sfe_centroids = sfe_centroids.to(device)

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)

        _, z_tgt_all, _, sfe_tgt = model([], x_batch, return_sfe_features=True)
        z_tgt_stack = torch.stack(z_tgt_all, dim=0)           # [n_sources, B, sem_dim]

        # 切空间距离 → 融合权重
        dists = torch.norm(
            sfe_tgt.unsqueeze(0) - sfe_centroids.unsqueeze(1),
            p=2, dim=-1
        )                                                      # [n_sources, B]
        weights = F.softmax(-dists / tau, dim=0).unsqueeze(-1) # [n_sources, B, 1]

        # 加权融合
        z_fused = torch.sum(z_tgt_stack * weights, dim=0)      # [B, sem_dim]
        z_final = F.normalize(z_fused, dim=-1)
        final_logits = torch.matmul(z_final, text_prototypes.T)

        preds = final_logits.argmax(dim=-1)
        all_y_true.append(y_batch.cpu().numpy())
        all_y_pred.append(preds.cpu().numpy())

    return np.concatenate(all_y_true), np.concatenate(all_y_pred)


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

    args.epochs = 50
    args.batch_size = 64
    args.lr = 1e-3

    args.experiment_name = 'results_deap_de_riemann'
    args.output_dir = os.path.join(project_root, f"results/{args.experiment_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"实验输出目录：{args.output_dir}")
    print(f"使用设备：{args.device}")

    sfe_dim = 512
    bd_dim = 128

    # ===== 数据准备 =====
    (all_data, all_label, allClassLabelvector,
     input_dim, num_classes, text_dim, channels) = load_data_de_riemann(
        args, args.device, riemann_metric='riemann'
    )

    print(f"Tanget space input dim: {input_dim}")
    print(f"Number of classes: {num_classes}")
    print(f"Channels (EEG): {channels}")

    # 文本原型
    sorted_indices = sorted(allClassLabelvector.keys())
    vectors_in_order = [allClassLabelvector[i] for i in sorted_indices]
    text_prototypes = torch.tensor(
        np.array(vectors_in_order, dtype=np.float32)
    ).to(args.device)

    n_sessions = len(all_data)
    n_subjects = len(all_data[0])

    # 结果容器
    results_acc = {}
    results_macrof1 = {}
    results_microf1 = {}
    results_acc_avg = {}
    results_macrof1_avg = {}
    results_microf1_avg = {}

    # 几何正则化
    geo_loss_fn = GeometricRegularizationLoss(
        margin=2.0, alpha=0.1, beta=0.1
    ).to(args.device)

    # ===== LOSO 主循环 =====
    for session_idx in range(n_sessions):
        results_acc_session = []
        results_macrof1_session = []
        results_microf1_session = []
        results_acc_avg_session = []
        results_macrof1_avg_session = []
        results_microf1_avg_session = []

        for target_sub in range(n_subjects):
            setup_seed(args.seed)
            print(f"\n{'='*60}")
            print(f"Session {session_idx} | Target subject {target_sub+1}/{n_subjects}")
            print(f"{'='*60}")

            source_ids = [s for s in range(n_subjects) if s != target_sub]

            # ---- 构建 source dataloaders ----
            source_loaders = []
            for sid in source_ids:
                X = torch.tensor(np.array(all_data[session_idx][sid]), dtype=torch.float32)
                y = torch.tensor(np.array(all_label[session_idx][sid]), dtype=torch.long)
                ds = TensorDataset(X, y)
                dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                                drop_last=False, num_workers=0, pin_memory=True)
                source_loaders.append(dl)

            # ---- target 数据 ----
            X_tgt_all = torch.tensor(np.array(all_data[session_idx][target_sub]), dtype=torch.float32)
            y_tgt_all = torch.tensor(np.array(all_label[session_idx][target_sub]), dtype=torch.long)
            ds_tgt = TensorDataset(X_tgt_all, y_tgt_all)

            dl_tgt_train = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True,
                                      drop_last=False, num_workers=0, pin_memory=True)
            dl_tgt_eval = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=False,
                                     num_workers=0, pin_memory=True)

            # ---- 模型 ----
            n_sources = len(source_loaders)
            model = SGDA_model(
                n_sources=n_sources,
                num_electrodes=input_dim,
                eeg_dim=sfe_dim,
                bottleneck_dim=bd_dim,
                text_dim=text_dim,
                dropout=0.1
            ).to(args.device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

            steps_per_epoch = min(len(dl) for dl in source_loaders)
            total_steps = args.epochs * steps_per_epoch
            print(f"epochs={args.epochs}, steps/epoch={steps_per_epoch}")

            tgt_iter = iter(dl_tgt_train)
            source_iters = [iter(dl) for dl in source_loaders]

            step = 0
            total_cls = total_mmd = disc = L_geo = 0.0
            best_acc = 0; best_macro_f1 = 0; best_micro_f1 = 0
            best_acc_avg = 0; best_macro_f1_avg = 0; best_micro_f1_avg = 0
            sfe_source_centroids = None

            # ---- 训练循环 ----
            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0.0
                epoch_geo_loss = 0.0

                for _ in range(steps_per_epoch):
                    step += 1

                    gamma = 2.0 / (1.0 + np.exp(-10.0 * step / total_steps)) - 1.0
                    alpha_w = float(gamma)
                    beta_w = float(gamma) / 100

                    # source batch
                    x_src_list, y_src_list = [], []
                    for i, src_iter in enumerate(source_iters):
                        try:
                            xb, yb = next(src_iter)
                        except StopIteration:
                            source_iters[i] = iter(source_loaders[i])
                            xb, yb = next(source_iters[i])
                        x_src_list.append(xb.to(args.device))
                        y_src_list.append(yb.to(args.device))

                    # target batch
                    try:
                        x_tgt, _ = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(dl_tgt_train)
                        x_tgt, _ = next(tgt_iter)
                    x_tgt = x_tgt.to(args.device)

                    optimizer.zero_grad()

                    z_src_all, z_tgt_all, sfe_src_list, sfe_tgt = model(
                        x_src_list, x_tgt, return_sfe_features=True
                    )

                    # 损失计算
                    total_cls, total_mmd = 0.0, 0.0
                    for i in range(n_sources):
                        total_cls += align_loss(z_src_all[i], text_prototypes, y_src_list[i])
                        total_mmd += mmd_linear(z_src_all[i], z_tgt_all[i])
                    total_cls /= n_sources
                    total_mmd /= n_sources
                    disc = discrepancy_loss(z_tgt_all, text_prototypes)

                    L_geo = 0.0
                    for i in range(n_sources):
                        L_geo += geo_loss_fn(sfe_src_list[i], y_src_list[i])
                    L_geo /= n_sources

                    lambda_geo = 0.1 * gamma
                    loss = total_cls + alpha_w * total_mmd + beta_w * disc + lambda_geo * L_geo

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()
                    epoch_geo_loss += L_geo.item()

                # 打印
                print(f"Epoch {epoch+1:3d}/{args.epochs} | "
                      f"Loss={epoch_loss/steps_per_epoch:.4f} "
                      f"CLS={total_cls:.4f} MMD={total_mmd:.6f} "
                      f"DISC={disc:.6f} GEO={L_geo:.6f}")

                # ---- 质心计算：每 5 epoch 更新 ----
                if (epoch + 1) % 5 == 0 or sfe_source_centroids is None:
                    model.eval()
                    sfe_source_centroids = []
                    with torch.no_grad():
                        for i in range(n_sources):
                            feats = []
                            for xb_s, _ in source_loaders[i]:
                                f = model.sfe(xb_s.to(args.device))
                                feats.append(f)
                            sfe_source_centroids.append(
                                torch.cat(feats, dim=0).mean(dim=0)
                            )
                    sfe_source_centroids = torch.stack(sfe_source_centroids, dim=0)

                # ---- 评估：自适应融合 ----
                y_true_adp, y_pred_adp = evaluate_with_tangent_fusion(
                    model, dl_tgt_eval, text_prototypes,
                    sfe_source_centroids, args.device, tau=1.0
                )
                test_acc = (y_true_adp == y_pred_adp).mean()
                macro_f1 = f1_score(y_true_adp, y_pred_adp, average='macro')
                micro_f1 = f1_score(y_true_adp, y_pred_adp, average='micro')

                if test_acc > best_acc:
                    best_acc = test_acc
                    best_macro_f1 = macro_f1
                    best_micro_f1 = micro_f1

                # ---- 评估：平均融合 ----
                y_true_avg, y_pred_avg = get_preds_avg(
                    model, dl_tgt_eval, text_prototypes,
                    args.device, fusion_type='feature'
                )
                test_acc_avg = (y_true_avg == y_pred_avg).mean()
                macro_f1_avg = f1_score(y_true_avg, y_pred_avg, average='macro')
                micro_f1_avg = f1_score(y_true_avg, y_pred_avg, average='micro')

                if test_acc_avg > best_acc_avg:
                    best_acc_avg = test_acc_avg
                    best_macro_f1_avg = macro_f1_avg
                    best_micro_f1_avg = micro_f1_avg

                print(f"  Eval: adp_acc={test_acc:.4f}, avg_acc={test_acc_avg:.4f} | "
                      f"best_adp={best_acc:.4f}, best_avg={best_acc_avg:.4f}")

                if best_acc == 1.0:
                    print("Early stop (perfect accuracy)")
                    break

            print(f"Final (target {target_sub+1}): "
                  f"adp={best_acc*100:.2f}%, avg={best_acc_avg*100:.2f}%")

            results_acc_session.append(best_acc)
            results_macrof1_session.append(best_macro_f1)
            results_microf1_session.append(best_micro_f1)
            results_acc_avg_session.append(best_acc_avg)
            results_macrof1_avg_session.append(best_macro_f1_avg)
            results_microf1_avg_session.append(best_micro_f1_avg)

        results_acc[session_idx] = results_acc_session
        results_macrof1[session_idx] = results_macrof1_session
        results_microf1[session_idx] = results_microf1_session
        results_acc_avg[session_idx] = results_acc_avg_session
        results_macrof1_avg[session_idx] = results_macrof1_avg_session
        results_microf1_avg[session_idx] = results_microf1_avg_session

    # ===== 汇总 =====
    def summarize(results_dict, label):
        all_vals = []
        for key in results_dict:
            all_vals.extend(results_dict[key])
        all_vals = np.array(all_vals)
        print(f"\n[{label}] Acc: {all_vals.mean()*100:.2f}% +/- {all_vals.std()*100:.2f}%")
        return all_vals

    summarize(results_acc, "Adaptive Fusion (tangent space)")
    summarize(results_acc_avg, "Average Fusion (baseline)")

    # ===== 保存 =====
    for fusion_name, (acc, maf1, mif1) in [
        ("adaptive", (results_acc, results_macrof1, results_microf1)),
        ("average", (results_acc_avg, results_macrof1_avg, results_microf1_avg)),
    ]:
        out_path = os.path.join(
            args.output_dir,
            f"csubs_deap_de_riemann_valence_{fusion_name}_bd{bd_dim}_bs{args.batch_size}.csv"
        )
        save_csubs_results_csv(acc, maf1, mif1, out_path)
        print(f"Results saved to {out_path}")


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    run_experiment(args)