"""
Riemann-SGDA：融合几何先验与LLM语义引导的跨被试脑电情绪识别
基于 DEAP 数据集 | 递进式实验框架

对照申报书技术路线：
  欧氏基线 → SGDA基线 → 纯黎曼几何基线 → 切空间语义对齐 → 完整Riemann-SGDA
  本脚本实现后三个阶段：SPD流形建模 + 切空间语义对齐 + 几何正则约束
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

# ===== 模型 =====
from models.model_riemann import SGDA_model


def load_data_riemann(args, device, use_riemann=True, riemann_metric='riemann'):
    """
    加载数据并进行黎曼预处理

    SPD流形 → Log-map → 切空间特征（C*(C+1)//2 维）
    """
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

    data, label, channels, num_freq_bands, num_classes = get_data(
        setting,
        use_riemann=use_riemann,
        riemann_metric=riemann_metric
    )

    if use_riemann:
        input_dim = channels * (channels + 1) // 2  # 切空间维度
    else:
        input_dim = channels * num_freq_bands

    text_dim, allClassLabelvector = label_to_vector(
        dataset=setting.dataset,
        LM=args.LM,
        LabelTextMapper=None,
        device=device
    )

    data, label = flatten_trials(data, label)

    return (
        data, label, allClassLabelvector,
        input_dim, num_classes, text_dim
    )


def run_experiment(args):
    """主实验流程（LOSO 跨被试）"""

    # ===== 基础配置 =====
    setup_seed(args.seed)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.LM = "clip"
    args.dataset = "deap"

    args.epochs = 50
    args.batch_size = 8
    args.lr = 1e-3

    args.experiment_name = 'results_deap_riemann'
    args.output_dir = os.path.join(project_root, f"results/{args.experiment_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"实验输出目录：{args.output_dir}")
    print(f"使用设备：{args.device}")

    sfe_dim = 512
    bd_dim = 128

    # ===== 数据准备（黎曼切空间特征） =====
    (all_data, all_label, allClassLabelvector,
     input_dim, num_classes, text_dim) = load_data_riemann(
        args, args.device, use_riemann=True, riemann_metric='riemann'
    )

    print(f"Input dimension (tangent space): {input_dim}")
    print(f"Number of classes: {num_classes}")

    # 文本原型充当类别原型
    sorted_indices = sorted(allClassLabelvector.keys())
    vectors_in_order = [allClassLabelvector[i] for i in sorted_indices]
    text_prototypes = torch.tensor(
        np.array(vectors_in_order, dtype=np.float32)
    ).to(args.device)

    n_sessions = len(all_data)
    n_subjects = len(all_data[0])

    # 结果容器：同时记录自适应融合和平均融合
    results_acc = {}
    results_macrof1 = {}
    results_microf1 = {}
    results_acc_avg = {}
    results_macrof1_avg = {}
    results_microf1_avg = {}

    # 初始化几何正则化损失（作用于切空间 SFE 特征）
    geo_loss_fn = GeometricRegularizationLoss(
        margin=2.0, alpha=0.1, beta=0.1
    ).to(args.device)

    # ===== 按 session 做 LOSO =====
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

            # ===== 构建 source dataloader =====
            source_loaders = []
            for sid in source_ids:
                X = torch.tensor(
                    np.array(all_data[session_idx][sid]),
                    dtype=torch.float32
                )
                y = torch.tensor(
                    np.array(all_label[session_idx][sid]),
                    dtype=torch.long
                )

                ds = TensorDataset(X, y)
                dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=False,
                    num_workers=0,
                    pin_memory=True
                )
                source_loaders.append(dl)

            # ===== target 数据 =====
            X_tgt_all = torch.tensor(
                np.array(all_data[session_idx][target_sub]),
                dtype=torch.float32
            )
            y_tgt_all = torch.tensor(
                np.array(all_label[session_idx][target_sub]),
                dtype=torch.long
            )

            ds_tgt = TensorDataset(X_tgt_all, y_tgt_all)

            dl_tgt_train = DataLoader(
                ds_tgt,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=0,
                pin_memory=True
            )
            dl_tgt_eval = DataLoader(
                ds_tgt,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True
            )

            # ===== 模型 =====
            n_sources = len(source_loaders)

            model = SGDA_model(
                n_sources=n_sources,
                num_electrodes=input_dim,
                eeg_dim=sfe_dim,
                bottleneck_dim=bd_dim,
                text_dim=text_dim,
                dropout=0.1
            ).to(args.device)

            optimizer = optim.Adam(
                model.parameters(),
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
            total_cls = total_mmd = disc = L_geo = 0.0
            best_acc = 0
            best_macro_f1 = 0
            best_micro_f1 = 0
            best_acc_avg = 0
            best_macro_f1_avg = 0
            best_micro_f1_avg = 0
            sfe_source_centroids = None  # cached every 5 epochs

            # ===== 训练循环 =====
            for epoch in range(args.epochs):
                model.train()
                epoch_loss = 0.0
                epoch_geo_loss = 0.0
                t0 = torch.cuda.Event(enable_timing=True) if args.device.type == 'cuda' else None
                t1 = torch.cuda.Event(enable_timing=True) if args.device.type == 'cuda' else None
                if t0: t0.record()

                for _ in range(steps_per_epoch):
                    step += 1

                    gamma = 2.0 / (1.0 + np.exp(-10.0 * step / total_steps)) - 1.0
                    alpha_w = float(gamma)
                    beta_w = float(gamma) / 100

                    x_src_list, y_src_list = [], []
                    for i, src_iter in enumerate(source_iters):
                        try:
                            xb, yb = next(src_iter)
                        except StopIteration:
                            source_iters[i] = iter(source_loaders[i])
                            xb, yb = next(source_iters[i])
                        x_src_list.append(xb.to(args.device))
                        y_src_list.append(yb.to(args.device))

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

                    total_cls, total_mmd = 0.0, 0.0
                    for i in range(n_sources):
                        cls = align_loss(z_src_all[i], text_prototypes, y_src_list[i])
                        mmd = mmd_linear(z_src_all[i], z_tgt_all[i])
                        total_cls += cls
                        total_mmd += mmd
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

                if t0: t1.record(); torch.cuda.synchronize()

                # ----- 每 epoch 打印 -----
                elapsed = f"{t0.elapsed_time(t1)/1000:.1f}s" if t0 else "?"
                print(f"Epoch {epoch+1:3d}/{args.epochs} [{elapsed}] | "
                      f"Loss={epoch_loss/steps_per_epoch:.4f} "
                      f"CLS={total_cls:.4f} MMD={total_mmd:.6f} "
                      f"DISC={disc:.6f} GEO={L_geo:.6f}")

                # ----- 每5 epoch计算质心+评估，其他epoch复用缓存的质心 -----
                if (epoch + 1) % 5 == 0 or sfe_source_centroids is None:
                    model.eval()
                    sfe_source_centroids = []
                    with torch.no_grad():
                        for i in range(n_sources):
                            feats = []
                            for xb_s, _ in source_loaders[i]:
                                f = model.sfe(xb_s.to(args.device))
                                feats.append(f)
                            centroid = torch.cat(feats, dim=0).mean(dim=0)
                            sfe_source_centroids.append(centroid)
                    sfe_source_centroids = torch.stack(sfe_source_centroids, dim=0)

                # 自适应融合评估
                model.eval()
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

                # 平均融合评估
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

    # ===== 汇总结果 =====
    def summarize(results_dict, label):
        all_vals = []
        for key in results_dict:
            all_vals.extend(results_dict[key])
        all_vals = np.array(all_vals)
        print(f"\n[{label}] Acc: {all_vals.mean()*100:.2f}% +/- {all_vals.std()*100:.2f}%")
        return all_vals

    summarize(results_acc, "Adaptive Fusion (tangent space)")
    summarize(results_acc_avg, "Average Fusion (baseline)")

    # ===== 保存结果 =====
    for fusion_name, (acc, maf1, mif1) in [
        ("adaptive", (results_acc, results_macrof1, results_microf1)),
        ("average", (results_acc_avg, results_macrof1_avg, results_microf1_avg)),
    ]:
        out_path = os.path.join(
            args.output_dir,
            f"csubs_deap_riemann_valence_{fusion_name}_bd{bd_dim}_bs{args.batch_size}.csv"
        )
        save_csubs_results_csv(acc, maf1, mif1, out_path)
        print(f"Results saved to {out_path}")


def evaluate_with_tangent_fusion(model, dataloader, text_prototypes,
                                  sfe_centroids, device, tau=1.0):
    """
    基于切空间（SFE特征）距离的自适应加权融合评估

    申报书方案：计算目标样本在切空间中与各源域中心的距离，
    通过 softmax(-dist/tau) 生成权重，加权融合各分支的语义空间输出。
    """
    model.eval()
    all_y_true = []
    all_y_pred = []

    sfe_centroids = sfe_centroids.to(device)  # [n_sources, sfe_dim]

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)

        # 获取 SFE 切空间特征 + 分支语义空间特征
        _, z_tgt_all, _, sfe_tgt = model([], x_batch, return_sfe_features=True)

        z_tgt_stack = torch.stack(z_tgt_all, dim=0)  # [n_sources, B, sem_dim]

        # 切空间距离 → 融合权重
        dists = torch.norm(
            sfe_tgt.unsqueeze(0) - sfe_centroids.unsqueeze(1),
            p=2, dim=-1
        )  # [n_sources, B]
        weights = F.softmax(-dists / tau, dim=0).unsqueeze(-1)  # [n_sources, B, 1]

        # 加权融合语义空间特征
        z_fused = torch.sum(z_tgt_stack * weights, dim=0)  # [B, sem_dim]
        z_final = F.normalize(z_fused, dim=-1)
        final_logits = torch.matmul(z_final, text_prototypes.T)

        preds = final_logits.argmax(dim=-1)

        all_y_true.append(y_batch.cpu().numpy())
        all_y_pred.append(preds.cpu().numpy())

    return np.concatenate(all_y_true), np.concatenate(all_y_pred)


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    run_experiment(args)

