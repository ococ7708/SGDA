"""
DE-SPD-Align-SGDA：DE 频谱特征 → SPD 协方差 → LOSO 统一切空间对齐
=====================================================================

核心改进（相对 crossSubject_reiman_deap.py）：
  旧版：raw EEG → np.cov → 单个SPD/trial → 逐被试 TangentSpace
        → 频段丢失 + 各被试切空间参考点不同

  新版：raw EEG → DE提取(5频段保留) → 分段 → 每样本 OAS-SPD
        → 每个 LOSO fold 用 source SPDs 拟合统一 TangentSpace
        → source & target 映射到同一切空间

技术路线对照：
  ① 欧氏基线（crossSubjects_deap.py）            → Acc: 65.81%
  ② SGDA 基线（crossSubjects_deap.py）            → Acc: 65.81%
  ③ 纯黎曼基线（crossSubject_reiman_deap.py）     → Acc: 65.39%
  ④ 本脚本：DE + 统一切空间对齐                    → 预期更高
  ⑤ 后续：+ 几何正则 + 超参调优
"""

import os, sys
import torch, torch.optim as optim
import numpy as np, torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from data_utils.load_data import get_data
from data_utils.text_to_vector import label_to_vector
from data_utils.constants.path_mapper import path_mapper
from utils.args import get_args_parser
from config.setting import preset_setting, set_setting_by_args
from utils.mix_utils import setup_seed, get_preds_avg
from utils.log_utils import save_csubs_results_csv
from utils.loss import align_loss, mmd_linear, discrepancy_loss, GeometricRegularizationLoss
from models.model_riemann import SGDA_model


TRIAL_SUBJECT_LIMIT = 3


# ============================================================
# 1. DE 样本 → SPD 矩阵（不做切空间映射）
# ============================================================
def convert_de_to_spd(all_data, reg=1e-6):
    """
    每个 DE 样本 (T, C, F) → reshape (C, T*F) → OAS-SPD (C×C)
    切空间映射推迟到 LOSO fold 内，以保证跨被试参考点统一。
    """
    from pyriemann.estimation import Covariances

    for ses_idx in range(len(all_data)):
        for sub_idx in range(len(all_data[ses_idx])):
            trials = all_data[ses_idx][sub_idx]
            if not trials:
                continue

            flat, lengths = [], []
            for trial in trials:
                lengths.append(len(trial))
                for s in trial:
                    T, C, F = s.shape
                    flat.append(s.reshape(C, T * F))

            stacked = np.stack(flat, axis=0)
            covs = Covariances(estimator='oas').fit_transform(stacked) + reg * np.eye(C)

            offset = 0
            for t_idx, n in enumerate(lengths):
                all_data[ses_idx][sub_idx][t_idx] = [covs[offset + i] for i in range(n)]
                offset += n

    print(f"[DE→SPD] 完成, SPD: {C}x{C}")
    return all_data


# ============================================================
# 2. 数据加载
# ============================================================
def load_data(args, device):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setting.dataset_path = path_mapper['deap']
    setting.dataset = 'deap'
    setting.experiment_mode = 'subject-independent'
    setting.onehot = False
    setting.label_used = ['valence']
    setting.bounds = [5, 5.0001]
    setting.sessions = [1]
    setting.sample_length = 3
    setting.stride = 1
    setting.only_seg = False

    data, label, channels, num_freq_bands, num_classes = get_data(setting)

    available_subjects = len(data[0])
    if available_subjects < 2:
        raise ValueError("DE-SPD-Align needs at least 2 subjects for LOSO training.")

    subject_limit = min(TRIAL_SUBJECT_LIMIT, available_subjects)
    data = [session[:subject_limit] for session in data]
    label = [session[:subject_limit] for session in label]
    print(f"Trial run: using first {subject_limit}/{available_subjects} subjects only.")
    print(f"DE 预处理: {channels}ch x {num_freq_bands}bands, {num_classes}类")

    data = convert_de_to_spd(data)

    text_dim, allClassLabelvector = label_to_vector(
        dataset=setting.dataset, LM=args.LM, LabelTextMapper=None, device=device
    )
    return data, label, allClassLabelvector, channels, num_classes, text_dim


# ============================================================
# 3. 切空间自适应融合评估
# ============================================================
def evaluate_tangent_fusion(model, dataloader, text_prototypes,
                            sfe_centroids, device, tau=1.0):
    model.eval()
    y_true, y_pred = [], []
    sfe_centroids = sfe_centroids.to(device)

    for xb, yb in dataloader:
        xb = xb.to(device)
        _, z_tgt_all, _, sfe_tgt = model([], xb, return_sfe_features=True)
        z_stack = torch.stack(z_tgt_all, dim=0)
        dists = torch.norm(sfe_tgt.unsqueeze(0) - sfe_centroids.unsqueeze(1), p=2, dim=-1)
        w = F.softmax(-dists / tau, dim=0).unsqueeze(-1)
        z_fused = torch.sum(z_stack * w, dim=0)
        logits = torch.matmul(F.normalize(z_fused, dim=-1), text_prototypes.T)
        y_true.append(yb.cpu().numpy())
        y_pred.append(logits.argmax(-1).cpu().numpy())

    return np.concatenate(y_true), np.concatenate(y_pred)


# ============================================================
# 4. 主实验
# ============================================================
def run(args):
    setup_seed(args.seed)
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.LM = "clip"
    args.epochs = 200
    args.batch_size = 64
    args.lr = 1e-3

    args.experiment_name = 'results_deap_de_spd_align'
    args.output_dir = os.path.join(project_root, f"results/{args.experiment_name}")
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"输出: {args.output_dir} | 设备: {args.device}")

    sfe_dim = 512
    bd_dim = 128

    # ---- 加载 SPD 数据 ----
    all_data_spd, all_label, allClassLabelvector, channels, num_classes, text_dim = \
        load_data(args, args.device)

    tangent_dim = channels * (channels + 1) // 2

    sorted_ids = sorted(allClassLabelvector.keys())
    text_prototypes = torch.tensor(
        np.array([allClassLabelvector[i] for i in sorted_ids], dtype=np.float32)
    ).to(args.device)

    n_sessions = len(all_data_spd)
    n_subjects = len(all_data_spd[0])

    results_acc = {}
    results_macrof1 = {}
    results_microf1 = {}
    results_acc_avg = {}
    results_macrof1_avg = {}
    results_microf1_avg = {}

    geo_loss_fn = GeometricRegularizationLoss(margin=2.0, alpha=0.1, beta=0.1).to(args.device)
    from pyriemann.tangentspace import TangentSpace

    for session_idx in range(n_sessions):
        r_acc, r_mac, r_mic = [], [], []
        r_acc_a, r_mac_a, r_mic_a = [], [], []

        for target_sub in range(n_subjects):
            setup_seed(args.seed)
            print(f"\n{'='*55}")
            print(f"Session {session_idx} | Target {target_sub+1}/{n_subjects}")
            print(f"{'='*55}")

            source_ids = [s for s in range(n_subjects) if s != target_sub]

            # ===== 核心：source SPDs → 统一 TangentSpace → transform all =====
            source_spds = []
            for sid in source_ids:
                for trial in all_data_spd[session_idx][sid]:
                    source_spds.extend(trial)
            source_spds = np.stack(source_spds, axis=0)

            ts = TangentSpace(metric='riemann')
            ts.fit(source_spds)

            tangent_data, tangent_labels = [], []
            for sub in range(n_subjects):
                trial_tangents = []
                for trial in all_data_spd[session_idx][sub]:
                    if not trial:
                        continue
                    trial_tangents.append(
                        ts.transform(np.stack(trial, axis=0)).astype(np.float32)
                    )
                sub_t = np.concatenate(trial_tangents, axis=0)

                # 展平标签
                flat_l = []
                for tl in all_label[session_idx][sub]:
                    flat_l.append(np.array(tl).reshape(-1))
                sub_l = np.concatenate(flat_l)

                # 被试级 z-score
                m = sub_t.mean(axis=0, keepdims=True)
                s = sub_t.std(axis=0, keepdims=True) + 1e-6
                sub_t = (sub_t - m) / s

                tangent_data.append(sub_t)
                tangent_labels.append(sub_l)

            # ===== DataLoaders =====
            source_loaders = []
            for sid in source_ids:
                ds = TensorDataset(
                    torch.tensor(tangent_data[sid], dtype=torch.float32),
                    torch.tensor(tangent_labels[sid], dtype=torch.long)
                )
                source_loaders.append(DataLoader(
                    ds, batch_size=args.batch_size, shuffle=True,
                    drop_last=False, num_workers=0, pin_memory=True
                ))

            ds_tgt = TensorDataset(
                torch.tensor(tangent_data[target_sub], dtype=torch.float32),
                torch.tensor(tangent_labels[target_sub], dtype=torch.long)
            )
            dl_tgt_train = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True,
                                      drop_last=False, num_workers=0, pin_memory=True)
            dl_tgt_eval = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=False,
                                     num_workers=0, pin_memory=True)

            # ===== 模型 =====
            n_sources = len(source_loaders)
            model = SGDA_model(
                n_sources=n_sources, num_electrodes=tangent_dim,
                eeg_dim=sfe_dim, bottleneck_dim=bd_dim,
                text_dim=text_dim, dropout=0.1
            ).to(args.device)

            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

            steps_per_epoch = min(len(dl) for dl in source_loaders)
            total_steps = args.epochs * steps_per_epoch
            print(f"epochs={args.epochs}, steps/epoch={steps_per_epoch}, "
                  f"samples/src~{tangent_data[source_ids[0]].shape[0]}")

            tgt_iter = iter(dl_tgt_train)
            src_iters = [iter(dl) for dl in source_loaders]

            step = 0
            best_acc = 0
            best_mac = 0
            best_mic = 0
            best_acc_a = 0
            best_mac_a = 0
            best_mic_a = 0
            sfe_centroids = None

            for epoch in range(args.epochs):
                model.train()
                ep_loss = 0.0

                for _ in range(steps_per_epoch):
                    step += 1
                    gamma = 2.0 / (1.0 + np.exp(-10.0 * step / total_steps)) - 1.0
                    aw = float(gamma)
                    bw = float(gamma) / 100

                    xs, ys = [], []
                    for i, it in enumerate(src_iters):
                        try:
                            xb, yb = next(it)
                        except StopIteration:
                            src_iters[i] = iter(source_loaders[i])
                            xb, yb = next(src_iters[i])
                        xs.append(xb.to(args.device))
                        ys.append(yb.to(args.device))

                    try:
                        xt, _ = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(dl_tgt_train)
                        xt, _ = next(tgt_iter)
                    xt = xt.to(args.device)

                    optimizer.zero_grad()
                    zs, zt, sfe_src, sfe_tgt = model(xs, xt, return_sfe_features=True)

                    cls_val = sum(align_loss(zs[i], text_prototypes, ys[i]) for i in range(n_sources)) / n_sources
                    mmd_val = sum(mmd_linear(zs[i], zt[i]) for i in range(n_sources)) / n_sources
                    disc_val = discrepancy_loss(zt, text_prototypes)
                    geo_val = sum(geo_loss_fn(sfe_src[i], ys[i]) for i in range(n_sources)) / n_sources

                    loss = cls_val + aw * mmd_val + bw * disc_val + 0.1 * gamma * geo_val
                    loss.backward()
                    optimizer.step()
                    ep_loss += loss.item()

                if (epoch + 1) % 5 == 0 or sfe_centroids is None:
                    model.eval()
                    sfe_centroids = []
                    with torch.no_grad():
                        for i in range(n_sources):
                            feats = [model.sfe(xb.to(args.device)) for xb, _ in source_loaders[i]]
                            sfe_centroids.append(torch.cat(feats, dim=0).mean(dim=0))
                    sfe_centroids = torch.stack(sfe_centroids, dim=0)

                yt, yp = evaluate_tangent_fusion(
                    model, dl_tgt_eval, text_prototypes, sfe_centroids, args.device
                )
                acc = (yt == yp).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_mac = f1_score(yt, yp, average='macro')
                    best_mic = f1_score(yt, yp, average='micro')

                yt_a, yp_a = get_preds_avg(model, dl_tgt_eval, text_prototypes, args.device, fusion_type='feature')
                acc_a = (yt_a == yp_a).mean()
                if acc_a > best_acc_a:
                    best_acc_a = acc_a
                    best_mac_a = f1_score(yt_a, yp_a, average='macro')
                    best_mic_a = f1_score(yt_a, yp_a, average='micro')

                print(f"Ep {epoch+1:3d} | Loss={ep_loss/steps_per_epoch:.4f} | "
                      f"adp={acc:.4f} avg={acc_a:.4f} | best_adp={best_acc:.4f} best_avg={best_acc_a:.4f}")

                if best_acc == 1.0:
                    break

            print(f"Final t{sub}: adp={best_acc*100:.2f}% avg={best_acc_a*100:.2f}%")
            r_acc.append(best_acc)
            r_mac.append(best_mac)
            r_mic.append(best_mic)
            r_acc_a.append(best_acc_a)
            r_mac_a.append(best_mac_a)
            r_mic_a.append(best_mic_a)

        results_acc[session_idx] = r_acc
        results_macrof1[session_idx] = r_mac
        results_microf1[session_idx] = r_mic
        results_acc_avg[session_idx] = r_acc_a
        results_macrof1_avg[session_idx] = r_mac_a
        results_microf1_avg[session_idx] = r_mic_a

    # ===== 汇总 =====
    for label, d in [("Adaptive", results_acc), ("Average", results_acc_avg)]:
        vals = np.concatenate([np.array(d[k]) for k in d])
        print(f"[{label}] Acc: {vals.mean()*100:.2f}% +/- {vals.std()*100:.2f}%")

    for fname, acc, maf, mif in [
        ("adaptive", results_acc, results_macrof1, results_microf1),
        ("average", results_acc_avg, results_macrof1_avg, results_microf1_avg),
    ]:
        path = os.path.join(args.output_dir,
            f"csubs_deap_de_spd_align_valence_{fname}_bd{bd_dim}_bs{args.batch_size}.csv")
        save_csubs_results_csv(acc, maf, mif, path)
        print(f"Saved: {path}")


if __name__ == '__main__':
    run(get_args_parser().parse_args())
