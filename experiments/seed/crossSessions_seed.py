import os
import sys
import torch
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import TensorDataset, DataLoader

# ===== path =====
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

# ===== data & utils =====
from data_utils.load_data import get_data
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

from utils.log_utils import save_csess_results_csv
from utils.loss import align_loss, mmd_linear, discrepancy_loss

# ===== model =====
from models.model import SGDA_model


def load_data(args, device):
    """load SEED + text embeddings"""
    if args.setting is not None:
        setting = preset_setting[args.setting](args)
    else:
        setting = set_setting_by_args(args)

    setting.dataset_path = path_mapper['seed_de_lds']
    setting.dataset = 'seed_de_lds'
    setting.experiment_mode = 'subject-independent'
    setting.onehot = False
    setting.sample_length = 3
    setting.sessions = [1, 2, 3]
    setting.stride = 1

    data, label, channels, num_freq_bands, num_classes = get_data(setting)

    text_dim, allClassLabelvector = label_to_vector(
        dataset=setting.dataset,
        LM=args.LM,
        LabelTextMapper=None,
        device=device
    )

    data, label = flatten_trials(data, label)

    return data, label, allClassLabelvector, channels, num_freq_bands, text_dim


def run_experiment(args):
    """cross-session: train on first N-1 sessions, test on last"""

    setup_seed(args.seed)

    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("当前设备：", args.device)
    args.LM = "clip"

    args.epochs = 200
    args.batch_size = 128
    args.lr = 1e-3

    args.experiment_name = 'results_seed'
    args.output_dir = os.path.join(project_root, f"results/{args.experiment_name}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"output dir: {args.output_dir}")

    bd_dim = 512

    # ===== data =====
    all_data, all_label, allClassLabelvector, \
        num_electrodes, num_freq_bands, text_dim = load_data(args, args.device)

    all_data = zscore_subject_wise(all_data)

    sorted_indices = sorted(allClassLabelvector.keys())
    vectors_in_order = [allClassLabelvector[i] for i in sorted_indices]
    text_prototypes = torch.tensor(np.array(vectors_in_order, dtype=np.float32)).to(args.device)
    prototypes_list = text_prototypes

    n_sessions = len(all_data)
    n_subjects = len(all_data[0])

    results_acc, results_macrof1, results_microf1 = [], [], []

    # ===== per subject =====
    for sub in range(n_subjects):
        print(f"\nSubject {sub} (cross-session)")

        # first N-1 sessions as source
        source_loaders = []
        for ses_id in range(n_sessions - 1):
            X = torch.tensor(np.array(all_data[ses_id][sub]), dtype=torch.float32).to(args.device)
            y = torch.tensor(np.array(all_label[ses_id][sub]), dtype=torch.long).to(args.device)

            ds = TensorDataset(X, y)
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
            source_loaders.append(dl)

        # last session as target
        X_tgt = torch.tensor(np.array(all_data[-1][sub]), dtype=torch.float32).to(args.device)
        y_tgt = torch.tensor(np.array(all_label[-1][sub]), dtype=torch.long).to(args.device)

        ds_tgt = TensorDataset(X_tgt, y_tgt)
        dl_tgt_train = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=True, drop_last=True)
        dl_tgt_eval = DataLoader(ds_tgt, batch_size=args.batch_size, shuffle=False)

        n_sources = len(source_loaders)

        model = SGDA_model(
            n_sources,
            num_electrodes,
            eeg_dim=text_dim,
            bottleneck_dim=bd_dim,
            text_dim=text_dim,
            dropout=0.0
        ).to(args.device)

        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

        steps_per_epoch = min(len(dl) for dl in source_loaders)
        total_steps = max(1, args.epochs * steps_per_epoch)

        tgt_iter = iter(dl_tgt_train)
        source_iters = [iter(dl) for dl in source_loaders]

        best_acc = 0
        step = 0

        # ===== training =====
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0.0

            for _ in range(steps_per_epoch):
                step += 1

                # schedule for adaptation strength
                gamma = 2.0 / (1.0 + np.exp(-10.0 * step / total_steps)) - 1.0
                alpha = float(gamma)
                beta = float(gamma) / 100

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
                z_src_all, z_tgt_all = model(x_src_list, x_tgt)

                # losses
                total_cls, total_mmd = 0.0, 0.0

                
                for i in range(n_sources):
                    total_cls += align_loss(z_src_all[i], prototypes_list, y_src_list[i])
                    total_mmd += mmd_linear(z_src_all[i], z_tgt_all[i])

                total_cls /= n_sources
                total_mmd /= n_sources

                disc = discrepancy_loss(z_tgt_all, prototypes_list)

                loss = total_cls + alpha * total_mmd + beta * disc
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1} | "
                    f"Loss={epoch_loss/steps_per_epoch:.4f} "
                    f"CLS={total_cls:.4f} "
                    f"MMD={total_mmd:.6f} "
                    f"DISC={disc:.6f}"
                )

            # ===== recompute source centroids =====
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

            source_centroids = torch.stack(source_centroids, dim=0)

            # ===== evaluation =====
            y_true, y_pred = get_preds(
                model,
                dl_tgt_eval,
                prototypes_list,
                source_centroids=source_centroids,
                device=args.device,
                fusion_type='feature',
                tau=1.0
            )

            acc = (y_true == y_pred).mean()

            if acc > best_acc:
                best_acc = acc
                macro_f1 = f1_score(y_true, y_pred, average='macro')
                micro_f1 = f1_score(y_true, y_pred, average='micro')

                print(f"New best acc: {best_acc:.4f} (epoch {epoch})")

            if best_acc == 1:
                break

        print(f"Final acc (sub {sub}): {best_acc*100:.2f}%")

        results_acc.append(best_acc)
        results_macrof1.append(macro_f1)
        results_microf1.append(micro_f1)

    # ===== summary =====
    results_acc = np.array(results_acc)

    print(
        f"\nMean acc: {results_acc.mean()*100:.2f}% "
        f"+/- {results_acc.std()*100:.2f}%"
    )

    out_path = os.path.join(
        args.output_dir,
        f"csess_seed_bd{bd_dim}_bs{args.batch_size}.csv"
    )

    save_csess_results_csv(
        results_acc,
        results_macrof1,
        results_microf1,
        out_path
    )


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    run_experiment(args)