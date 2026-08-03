import os
import sys
import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader

# ===== 路径处理 =====
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# ===== 数据与工具 =====
from data_utils.load_dreamer import get_data
from data_utils.text_to_vector import label_to_vector
from data_utils.constants.path_mapper import path_mapper

from utils.args import get_args_parser
from config.setting import preset_setting, set_setting_by_args

from utils.mix_utils import (
    setup_seed,
    flatten_trials,
    zscore_subject_wise,
    get_preds
)

from utils.mix_utils import (apply_euclidean_alignment, global_normalization_after_ea)

from utils.log_utils import save_csubs_results_csv
from utils.loss import align_loss, mmd_linear, discrepancy_loss

# ===== 模型 =====
from models.model import SGDA_model



def load_data(args ,device):
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setting.dataset_path = path_mapper['dreamer']  #需要提前去对应脚本提取好DE特征并放到对应路径
    setting.dataset = 'dreamer'
    setting.experiment_mode = 'subject-independent'
    setting.labeltype='val'  # 唤醒度把“val”换“aro”
    setting.onehot = False
    setting.sessions = [1]
    setting.sample_length = 3
    setting.stride = 1

    # 读取数据
    data, label, channels, num_freq_bands, num_classes = get_data(setting)

    # 文本标签向量获取
    text_dim, allClassLabelvector = label_to_vector(
        dataset=setting.dataset,
        LM=args.LM,
        LabelTextMapper=None,
        device=device
    )

    # trial 展平
    data, label = flatten_trials(data, label)

    return (
        data, label, allClassLabelvector,
        channels, num_freq_bands, text_dim
    )


def run_experiment(args):
    # ===== 基础配置 =====
    setup_seed(args.seed)

    args.device = torch.device("cuda:1")
    args.LM = "clip"
    args.dataset = "dreamer"

    args.epochs = 200
    args.batch_size = 64
    args.lr = 1e-3

    args.experiment_name = 'results_dreamer'
    args.output_dir = os.path.join(project_root, f"results/{args.experiment_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"实验输出目录: {args.output_dir}")

    bd_dim = 128

    # ===== 数据准备 =====
    (all_data, all_label, allClassLabelvector,
     num_electrodes, num_freq_bands, text_dim) = load_data(args, args.device)

    all_data = apply_euclidean_alignment(all_data)
    all_data, _ = global_normalization_after_ea(all_data)

    # 文本原型充当类别原型
    sorted_indices = sorted(allClassLabelvector.keys())
    vectors_in_order = [allClassLabelvector[i] for i in sorted_indices]
    text_prototypes = torch.tensor(
        np.array(vectors_in_order, dtype=np.float32)
    ).to(args.device)
    prototypes_list = text_prototypes

    n_sessions = len(all_data)
    n_subjects = len(all_data[0])

    results_acc, results_macrof1, results_microf1 = {}, {}, {}

    # ===== 按 session 做 LOSO =====
    for session_idx in range(n_sessions):
        results_acc_session = []
        results_macrof1_session = []
        results_microf1_session = []

        for target_sub in range(n_subjects):
            setup_seed(args.seed)

            print(f"\nSession {session_idx} | Target subject {target_sub}")

            source_ids = [s for s in range(n_subjects) if s != target_sub]

            # ===== 构建 source dataloader =====
            source_loaders = []
            for sid in source_ids:
                X = torch.tensor(
                    np.array(all_data[session_idx][sid]),
                    dtype=torch.float32
                ).to(args.device)

                y = torch.tensor(
                    np.array(all_label[session_idx][sid]),
                    dtype=torch.long
                ).to(args.device)

                ds = TensorDataset(X, y)
                dl = DataLoader(
                    ds,
                    batch_size=args.batch_size,
                    shuffle=True,
                    drop_last=True
                )
                source_loaders.append(dl)

            # ===== target 数据 =====
            X_tgt_all = torch.tensor(
                np.array(all_data[session_idx][target_sub]),
                dtype=torch.float32
            ).to(args.device)

            y_tgt_all = torch.tensor(
                np.array(all_label[session_idx][target_sub]),
                dtype=torch.long
            ).to(args.device)

            ds_tgt = TensorDataset(X_tgt_all, y_tgt_all)

            dl_tgt_train = DataLoader(
                ds_tgt,
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True
            )

            dl_tgt_eval = DataLoader(
                ds_tgt,
                batch_size=args.batch_size,
                shuffle=False
            )

            # ===== 模型 =====
            n_sources = len(source_loaders)

            model = SGDA_model(
                n_sources,
                num_electrodes,
                eeg_dim=text_dim,
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
            best_acc = 0

            # ===== 训练 =====
            for epoch in range(args.epochs):

                model.train()
                epoch_loss = 0.0

                for _ in range(steps_per_epoch):
                    step += 1

                    # 动态权重
                    gamma = 2.0 / (1.0 + np.exp(-10.0 * step / total_steps)) - 1.0
                    alpha = float(gamma)
                    beta = float(gamma) / 100

                    x_src_list, y_src_list = [], []

                    # source batch
                    for i, src_iter in enumerate(source_iters):
                        try:
                            xb, yb = next(src_iter)
                        except StopIteration:
                            source_iters[i] = iter(source_loaders[i])
                            xb, yb = next(source_iters[i])

                        x_src_list.append(xb.to(args.device))
                        y_src_list.append(yb.to(args.device))

                    # target batch（不使用标签）
                    try:
                        x_tgt, _ = next(tgt_iter)
                    except StopIteration:
                        tgt_iter = iter(dl_tgt_train)
                        x_tgt, _ = next(tgt_iter)

                    x_tgt = x_tgt.to(args.device)

                    optimizer.zero_grad()

                    z_src_all, z_tgt_all = model(x_src_list, x_tgt)

                    # ===== loss =====
                    total_cls, total_mmd = 0.0, 0.0

                    for i in range(n_sources):
                        cls = align_loss(
                            z_src_all[i],
                            text_prototypes,
                            y_src_list[i]
                        )

                        mmd = mmd_linear(
                            z_src_all[i],
                            z_tgt_all[i]
                        )

                        total_cls += cls
                        total_mmd += mmd

                    total_cls /= n_sources
                    total_mmd /= n_sources

                    disc = discrepancy_loss(z_tgt_all, text_prototypes)

                    loss = total_cls + alpha * total_mmd + beta * disc

                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                # ===== 打印loss =====
                if (epoch + 1) % 10 == 0:
                    print(
                        f"Epoch {epoch+1} | "
                        f"Loss={epoch_loss/steps_per_epoch:.4f} "
                        f"CLS={total_cls:.4f} "
                        f"MMD={total_mmd:.6f} "
                        f"DISC={disc:.6f}"
                    )

                # ===== 计算 source centroid =====
                model.eval()
                source_centroids = []

                with torch.no_grad():
                    for i in range(n_sources):
                        feats = []

                        for xb_s, _ in source_loaders[i]:
                            
                            f = model.sfe(xb_s.to(args.device))
                            f = model.branches[i](f)
                            feats.append(f)

                        centroid = torch.cat(feats, dim=0).mean(dim=0)
                        source_centroids.append(centroid)

                source_centroids_tensor = torch.stack(source_centroids, dim=0)

                # ===== 在 target 上评估 =====
                y_true, y_pred = get_preds(
                    model,
                    dl_tgt_eval,
                    prototypes_list,
                    source_centroids=source_centroids_tensor,
                    device=args.device,
                    fusion_type='feature',
                    tau=1.0
                )

                test_acc = (y_true == y_pred).mean()

                if test_acc > best_acc:
                    best_acc = test_acc
                    macro_f1 = f1_score(y_true, y_pred, average='macro')
                    micro_f1 = f1_score(y_true, y_pred, average='micro')

                    print(f"New best acc: {best_acc:.4f} (epoch {epoch})")

                if best_acc == 1:
                    print("Early stop (perfect accuracy)")
                    break

            print(f"Final acc (target {target_sub}): {best_acc*100:.2f}%")

            results_acc_session.append(best_acc)
            results_macrof1_session.append(macro_f1)
            results_microf1_session.append(micro_f1)

        results_acc[session_idx] = results_acc_session
        results_macrof1[session_idx] = results_macrof1_session
        results_microf1[session_idx] = results_microf1_session

    # ===== 汇总结果 =====
    all_results_acc = []

    for key in results_acc:
        all_results_acc.extend(results_acc[key])

    all_results_acc = np.array(all_results_acc)

    print(
        f"\nOverall Acc: {all_results_acc.mean()*100:.2f}% "
        f"+/- {all_results_acc.std()*100:.2f}%"
    )

    # ===== 保存 =====
    out_path = os.path.join(
        args.output_dir,
        f"csubs_dreamer_val_bd{bd_dim}_bs{args.batch_size}.csv"
    )

    save_csubs_results_csv(
        results_acc,
        results_macrof1,
        results_microf1,
        out_path
    )

    print(f"Results saved to {out_path}")


if __name__ == '__main__':
    args = get_args_parser().parse_args() 
    run_experiment(args)