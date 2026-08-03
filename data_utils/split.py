import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, LeaveOneOut, StratifiedKFold, train_test_split
import random


def index_to_data(data, label, train_indexes, test_indexes, val_indexes, keep_dim=False):
    train_data = []
    train_label = []
    val_data = []
    val_label = []
    test_data = []
    test_label = []
    if keep_dim:
        # 每个被试者的45个trial包裹成一团保持原始维度加入
        for train_index in train_indexes:
            train_data.append(data[train_index])
            train_label.append(label[train_index])
        for test_index in test_indexes:
            test_data.append(data[test_index])
            test_label.append(label[test_index])
        if val_indexes[0] != -1:
            for val_index in val_indexes:
                val_data.append(data[val_index])
                val_label.append(label[val_index])
    else:
        # 每个被试者的45个trial扁平化再加入训练集和验证集
        for train_index in train_indexes:
            train_data.extend(data[train_index])
            train_label.extend(label[train_index])
        for test_index in test_indexes:
            test_data.extend(data[test_index])
            test_label.extend(label[test_index])
        if val_indexes[0] != -1:
            for val_index in val_indexes:
                val_data.extend(data[val_index])
                val_label.extend(label[val_index])
        train_data = np.array(train_data)
        test_data = np.array(test_data)
        train_label = np.array(train_label)
        test_label = np.array(test_label)
        val_data = np.array(val_data)
        val_label = np.array(val_label)
    return train_data, train_label, val_data, val_label, test_data, test_label


def get_split_index(data, label, setting=None):
    tts = {}
    if setting.split_type == "kfold":
        # 使用 scikit-learn 库中的 KFold 类进行交叉验证的设置
        kf = KFold(setting.fold_num, shuffle=True if setting.fold_shuffle == 'true' or setting.fold_shuffle == 'True' else False,
                   random_state=setting.seed if setting.fold_shuffle == 'true' else None)
        tts['train'] = [list(train_index) for train_index, _ in kf.split(label)]  # 对于SEED，label长度是15（一个session中被试者个数）
        tts['test'] = [list(test_index) for _, test_index in kf.split(label)]
    elif setting.split_type == "leave-one-out":
        loo = LeaveOneOut()
        tts['train'] = [list(train_index) for train_index, _ in loo.split(label)]
        tts['test'] = [list(test_index) for _, test_index in loo.split(label)]
    elif setting.split_type == "front-back":
        if setting.front >= len(label):
            print(f"using front-back split type and {setting.experiment_mode} experiment mode")
            print(f"front size {setting.front} > split part num {len(label)}")
            print("please check your experiment mode or split type")
            exit(1)
        tts['train'] = [[i for i in range(setting.front)]]
        tts['test'] = [[setting.front + i for i in range(len(label) - setting.front)]]
    elif setting.split_type == "train-val-test":
        if setting.experiment_mode == "subject-dependent":
            # data need to be split balanced
            # input data : [[not-repetitive] * trails], label : [[repetitive] * trails]
            # output : split index
            tts['test'] = [[]]
            tts['train'] = [[]]
            tts['val'] = [[]]
            groups = {}
            for index, value in enumerate(label):
                # print("index:", index)
                # print("value:", value)
                if isinstance(value[0], np.ndarray):
                    value_key = tuple(value[0])
                else:
                    value_key = value[0]
                if value_key in groups:
                    groups[value_key].append(index)
                else:
                    groups[value_key] = [index]
            # print(groups)
            others = []
            for indexes in groups.values():
                random.shuffle(indexes)
                total_length = len(indexes)
                test_num = int(setting.test_size * total_length)
                val_num = int(setting.val_size * total_length)
                train_num = int((1-setting.test_size-setting.val_size)*total_length)
                tts['test'][0].extend(indexes[:test_num])
                tts['val'][0].extend(indexes[test_num:test_num+val_num])
                tts['train'][0].extend(indexes[test_num+val_num:test_num+val_num+train_num])
                others.extend(indexes[test_num+val_num+train_num:])
            if len(others) != 0:
                random.shuffle(others)
                expect_test_num = int(len(label) * setting.test_size)
                expect_val_num = int(len(label) * setting.val_size)
                test_num = expect_test_num - len(tts['test'][0])
                val_num = expect_val_num - len(tts['val'][0])
                tts['test'][0].extend(others[:test_num])
                tts['val'][0].extend(others[test_num:test_num+val_num])
                tts['train'][0].extend(others[test_num+val_num:])
        else:
            tts['test'] = [[]]
            tts['train'] = [[]]
            tts['val'] = [[]]
            indexes = [i for i in range(len(label))]
            random.shuffle(indexes)
            total_length = len(indexes)
            test_num = int(setting.test_size * total_length)
            val_num = int(setting.val_size * total_length)
            train_num = total_length - test_num - val_num
            tts['test'][0].extend(indexes[:test_num])
            tts['val'][0].extend(indexes[test_num:test_num + val_num])
            tts['train'][0].extend(indexes[test_num + val_num:])
    else:
        print("wrong split type, please check out")
        exit(1)
    assert setting.sr is None or (max(setting.sr) <= len(label) and min(setting.sr) > 0), \
        "secondary rounds out of limit or secondary rounds set less than 0"
    if setting.sr is not None:
        tts['train'] = [tts['train'][i-1] for i in setting.sr]
        tts['test'] = [tts['test'][i-1] for i in setting.sr]
        if 'val' in tts:
            tts['val'] = [tts['val'][i-1] for i in setting.sr]
    if 'val' not in tts:
        tts['val'] = [[-1] for _ in tts['train']]
    return tts


def merge_to_part(data, label, setting=None):
    """
    其作用是根据实验模式，将数据和标签从（session, subject, trail, sample,...）的形式合并为不同的形式
    According to experiment mode, merge (session, subject, trail, sample) to (corresponding_part, sample)
    :param data: -> (session, subject, trail, sample, ...)
    :param label: -> (session, subject, trail, sample, ...)
    :param setting: -> setting for dataset process
    setting.experiment_mode: choices->["subject-dependent", "subject-independent", "cross-session"]
    setting.sessions: which sessions we choose to use, index start from 1, default is all
    :return: if not subject-dependent:
                 data: -> (corresponding_part, sample)
                 label: ->(corresponding_part, sample)
             else if subject-dependent and cross-trail:
                 data: -> (subject, trail, sample)
                 label: -> (subject, trail, sample)
                 else subject-dependent and not cross-trail 不是“跨试次”则无需区分试次标签
                 data: -> (subject, , sample)
                 label: -> (subject, , sample)
    """
    # 用于实验的sessions
    assert setting.sessions is None or (max(setting.sessions) <= len(label) and min(setting.sessions) >= 0), \
        "sessions set fault, session not exist in dataset"
    if setting.sessions is None:
        sessions = range(len(data))
    else:
        sessions = [i - 1 for i in setting.sessions]
    m_data = []
    m_label = []


    #  默认跨试次trial
    if setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'true':
        m_data = [[] for _ in range(len(data[0]) * len(sessions))]
        m_label = [[] for _ in range(len(data[0]) * len(sessions))]
        for session_pos, i in enumerate(sessions):
            for idx1, subject in enumerate(data[i]):
                for idx2, trail in enumerate(subject):
                    # 不同session的同个被试者各自独占一个m_data的索引，加入所有完整的trial
                    m_data[session_pos * len(data[i]) + idx1].append(trail)
                    m_label[session_pos * len(data[i]) + idx1].append(label[i][idx1][idx2])
    elif setting.experiment_mode == "subject-dependent" and setting.cross_trail == 'false':
        m_data = [[] for _ in range(len(data[0]) * len(sessions))]
        m_label = [[] for _ in range(len(data[0]) * len(sessions))]
        for session_pos, i in enumerate(sessions):
            for idx1, subject in enumerate(data[i]):
                for idx2, trail in enumerate(subject):
                    for sample in trail:
                        # 不同session的每个被试者各占一个m_data的索引，加入所有trial展平后的sample
                        m_data[session_pos * len(data[0])+idx1].append([sample])
                        m_label[session_pos * len(data[0]) + idx1].append([label[i][idx1][idx2]])
    elif setting.experiment_mode == "subject-independent":
        m_data = [[[] for _ in range(len(data[0]))]]  # 外层长度为1，内层长度为subject数量
        m_label = [[[] for _ in range(len(data[0]))]]
        for i in sessions:
            for idx, subject in enumerate(data[i]):
                for trail in subject:
                    # 几个session的同个被试者只占同一个m_data[0]的索引
                    m_data[0][idx].extend(trail)
        for i in sessions:
            for idx, subject in enumerate(label[i]):
                for trail in subject:
                    m_label[0][idx].extend(trail)
    elif setting.experiment_mode == "cross-session":
        m_data = [[[] for _ in range(len(sessions))]]  # 外层长度为1，内层长度为session数量
        m_label = [[[] for _ in range(len(sessions))]]
        for i in sessions:
            for subject in data[i]:
                for trail in subject:
                    #  同一个session的不同的subject的不同trial独立且杂糅在一起
                    m_data[0][i].extend(trail)
        for i in sessions:
            for subject in label[i]:
                for trail in subject:
                    m_label[0][i].extend(trail)
    assert setting.pr is None or (max(setting.pr)<=len(m_label) and min(setting.pr) > 0), \
        "primary rounds out of limit or primary rounds set less than 0"
    if setting.pr is not None:
        m_data = [m_data[i-1] for i in setting.pr]
        m_label = [m_label[i-1] for i in setting.pr]
    return m_data, m_label
