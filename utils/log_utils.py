import csv
import numpy as np

def save_csess_results_csv(results_acc, results_macrof1, results_microf1, save_path):
    """
    将被试实验结果（acc/macrof1/microf1）保存为CSV文件，包含均值和标准差
    
    参数：
        results_acc: list - 每个被试的准确率结果
        results_macrof1: list - 每个被试的宏平均F1结果
        results_microf1: list - 每个被试的微平均F1结果
        save_path: str - CSV文件保存路径（如 "/home/xxx/exp_results.csv"）
    """
    # 1. 计算各指标的均值和标准差（样本标准差，ddof=1，符合统计学规范）
    mean_acc, std_acc = np.mean(results_acc), np.std(results_acc, ddof=1)
    mean_macrof1, std_macrof1 = np.mean(results_macrof1), np.std(results_macrof1, ddof=1)
    mean_microf1, std_microf1 = np.mean(results_microf1), np.std(results_microf1, ddof=1)
    
    # 2. 构建CSV数据（表头 + 被试数据 + 统计行）
    csv_data = []
    # 表头
    csv_data.append(["subject_id", "acc", "macro_f1", "micro_f1"])
    # 逐被试数据（按索引匹配，确保三个数组长度一致）
    for idx in range(len(results_acc)):
        csv_data.append([
            idx,  # 被试ID从1开始
            round(results_acc[idx], 6),  # 保留6位小数，避免冗余
            round(results_macrof1[idx], 6),
            round(results_microf1[idx], 6)
        ])
    # 均值±标准差行（标注清晰，方便后续查看）
    csv_data.append([
        "mean±std",
        f"{mean_acc:.6f}±{std_acc:.6f}",
        f"{mean_macrof1:.6f}±{std_macrof1:.6f}",
        f"{mean_microf1:.6f}±{std_microf1:.6f}"
    ])
    
    # 3. 写入CSV文件（Linux/Windows通用，指定UTF-8编码避免乱码）
    try:
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        print(f"[OK] Results saved to: {save_path}")
    except PermissionError:
        print(f"[ERR] Permission denied: {save_path}")
    except Exception as e:
        print(f"[ERR] Save failed: {str(e)}")


def save_csubs_results_csv(results_acc, results_macrof1, results_microf1, save_path):
    """
    保存跨session的被试实验结果为CSV，包含各session统计值+全局统计值
    
    参数：
        results_acc: dict - key=session_idx, value=该session下所有被试的acc列表
        results_macrof1: dict - key=session_idx, value=该session下所有被试的macrof1列表
        results_microf1: dict - key=session_idx, value=该session下所有被试的microf1列表
        save_path: str - CSV文件保存路径（Linux/Windows通用）
    """
    # 校验输入数据合法性
    if not (results_acc.keys() == results_macrof1.keys() == results_microf1.keys()):
        raise ValueError("Session indices mismatch across result dicts!")
    n_sessions = len(results_acc)
    if n_sessions == 0:
        raise ValueError("No valid session data!")
    
    # 1. 预处理数据：展平所有session结果 + 计算各session/全局统计值
    all_acc, all_macrof1, all_microf1 = [], [], []  # 全局展平结果
    session_stats = {}  # 存储每个session的统计值
    
    for session_idx in sorted(results_acc.keys()):
        # 提取当前session的所有被试结果
        acc_list = results_acc[session_idx]
        macrof1_list = results_macrof1[session_idx]
        microf1_list = results_microf1[session_idx]
        
        # 校验当前session的被试数量一致
        if not (len(acc_list) == len(macrof1_list) == len(microf1_list)):
            raise ValueError(f"Subject count mismatch in session {session_idx}!")
        
        # 计算当前session的均值+标准差（样本标准差ddof=1）
        session_stats[session_idx] = {
            "acc_mean": np.mean(acc_list),
            "acc_std": np.std(acc_list, ddof=1),
            "macrof1_mean": np.mean(macrof1_list),
            "macrof1_std": np.std(macrof1_list, ddof=1),
            "microf1_mean": np.mean(microf1_list),
            "microf1_std": np.std(microf1_list, ddof=1)
        }
        
        # 展平到全局列表
        all_acc.extend(acc_list)
        all_macrof1.extend(macrof1_list)
        all_microf1.extend(microf1_list)
    
    # 计算全局统计值
    global_stats = {
        "acc_mean": np.mean(all_acc),
        "acc_std": np.std(all_acc, ddof=1),
        "macrof1_mean": np.mean(all_macrof1),
        "macrof1_std": np.std(all_macrof1, ddof=1),
        "microf1_mean": np.mean(all_microf1),
        "microf1_std": np.std(all_microf1, ddof=1)
    }
    
    # 2. 构建CSV数据（分三部分：原始数据、session统计、全局统计）
    csv_data = []
    
    # 第一部分：原始数据（session+被试ID+各指标值）
    csv_data.append(["session_idx", "subject_id", "acc", "macro_f1", "micro_f1"])
    for session_idx in sorted(results_acc.keys()):
        acc_list = results_acc[session_idx]
        macrof1_list = results_macrof1[session_idx]
        microf1_list = results_microf1[session_idx]
        
        for sub_idx, (acc, macro, micro) in enumerate(zip(acc_list, macrof1_list, microf1_list)):
            csv_data.append([
                session_idx,
                sub_idx,  # 被试ID从1开始
                round(acc, 6),
                round(macro, 6),
                round(micro, 6)
            ])
    
    # 分隔行（增强可读性）
    csv_data.append(["---", "---", "---", "---", "---"])
    
    # 第二部分：各session的均值±标准差
    csv_data.append(["session_idx", "stat_type", "acc (mean±std)", "macro_f1 (mean±std)", "micro_f1 (mean±std)"])
    for session_idx in sorted(session_stats.keys()):
        stats = session_stats[session_idx]
        csv_data.append([
            session_idx,
            "mean±std",
            f"{stats['acc_mean']:.6f}±{stats['acc_std']:.6f}",
            f"{stats['macrof1_mean']:.6f}±{stats['macrof1_std']:.6f}",
            f"{stats['microf1_mean']:.6f}±{stats['microf1_std']:.6f}"
        ])
    
    # 分隔行
    csv_data.append(["---", "---", "---", "---", "---"])
    
    # 第三部分：全局（所有session展平）的均值±标准差
    csv_data.append(["global", "stat_type", "acc (mean±std)", "macro_f1 (mean±std)", "micro_f1 (mean±std)"])
    csv_data.append([
        "all_sessions",
        "mean±std",
        f"{global_stats['acc_mean']:.6f}±{global_stats['acc_std']:.6f}",
        f"{global_stats['macrof1_mean']:.6f}±{global_stats['macrof1_std']:.6f}",
        f"{global_stats['microf1_mean']:.6f}±{global_stats['microf1_std']:.6f}"
    ])
    
    # 3. 写入CSV文件（异常处理+编码兼容）
    try:
        with open(save_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)
        print(f"[OK] Results saved to: {save_path}")
        print(f"[INFO] {n_sessions} sessions, {len(all_acc)} total subjects")
    except PermissionError:
        print(f"[ERR] Permission denied: {save_path}")
    except Exception as e:
        print(f"[ERR] Save failed: {str(e)}")