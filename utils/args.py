import argparse
import os
import torch

from config.setting import preset_setting
from data_utils.load_data import available_dataset


def get_args_parser():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    parser = argparse.ArgumentParser(
        "cross-subject EEG emotion recognition",
        add_help=True
    )

    # =========================
    # 1. 通用 / 训练参数
    # =========================
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--epochs', type=int, default=200, help="训练轮数")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3, help="学习率")
    parser.add_argument('--patience', type=int, default=50, help="早停")
    parser.add_argument('--device', type=str,default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--seed', type=int, default=42)

    # =========================
    # 2. 实验设置
    # =========================
    parser.add_argument('--experiment_name', type=str, default="--")
    parser.add_argument('--output_dir', type=str,
                        default=project_root + "/exp_output")
    parser.add_argument('--is_vis', default=False, action='store_true')
    parser.add_argument('--LM', default='bert',
                        help="'clip', 'bert', 'sbert', 'roberta_go'")
    parser.add_argument('--setting', default=None,choices=preset_setting,help='using preset setting')
    parser.add_argument('-metrics', default=['acc', 'macro-f1'], type=str, nargs='+', help='which metrics used to evaluate')

    # =========================
    # 3. 数据集读取和处理参数
    # =========================
    parser.add_argument('--dataset', default='seed_de_lds',
                        choices=available_dataset)
    parser.add_argument('--dataset_path', type=str,
                        default='/home/ch/datasets/SEED')
    parser.add_argument('--low_pass', type=float, default=0.3)
    parser.add_argument('--high_pass', type=float, default=50)
    parser.add_argument('--time_window', type=float, default=1)
    parser.add_argument('--overlap', type=float, default=0)
    parser.add_argument('--sample_length', type=int, default=1)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--feature_type', type=str, default='de_lds')
    parser.add_argument('--eog_clean', action='store_true', default=False)
    parser.add_argument('--only_seg', action='store_true', default=False)
    parser.add_argument('--normalize', default=True)
    

    # =========================
    # 4. 划分 / 标签设置
    # =========================
    parser.add_argument('--cross_trail', type=str, default='true')
    parser.add_argument('--experiment_mode', type=str,
                        default='subject-dependent')
    parser.add_argument('-bounds', default=None, type=float, nargs='+', help="emotion score bounds:[low, high]")
    parser.add_argument('--onehot', action='store_true', default=True)
    parser.add_argument('--label_used', type=str, nargs='+', default=None,
                        help="valence, arousal, dominance, liking")
    parser.add_argument('--keep_dim', action='store_true', default=False)
    parser.add_argument('-sessions', default=None, type=int, nargs='+', help="which sessions used to train")  # none


    # =========================
    # 其他/未用到但保留，避免报错
    # =========================
    parser.add_argument('-split_type', default='kfold', type=str, choices=['kfold', 'leave-one-out', 'front-back','train-val-test'],
                        help="choose which method to split dataset")
    parser.add_argument('-fold_num', default=5, type=int, help='the number of folds')  # 折数，在使用k-fold分割方法时使用，默认值为5。
    parser.add_argument('-fold_shuffle', default='true', type=str, help='whether shuffle when using k-fold split')
    parser.add_argument('-front', default=9, type=int, help='convert the first few data sets into training sets')
    parser.add_argument('-pr', default=None, type=int, nargs='+', help="which primary rounds to train")
    parser.add_argument('-sr', default=None, type=int, nargs='+', help="which secondary rounds to train")


    return parser