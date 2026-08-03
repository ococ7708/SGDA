import numpy as np
import pickle

def read_seedv_feature(dir_path,read_sessions):

    data=[[[] for _ in range(16)] for _ in range(len(read_sessions))]
    label=[[[] for _ in range(16)] for _ in range(len(read_sessions))]

    # 由于各个subject的的3session都存在同一个文件里，所以按sub编号在最外层循环
    for i in range(16):

        dir_path_label=dir_path+f"/{str(i+1)}_label.npy"
        dir_path_data = dir_path + f"/{str(i+1)}_data.npy"
        # 读取文件并反序列化
        label_bytes = np.load(dir_path_label, allow_pickle=True).item()
        data_bytes = np.load(dir_path_data, allow_pickle=True).item()

        sub_label = pickle.loads(label_bytes)  # dict
        sub_data = pickle.loads(data_bytes)    # dict

        # 按session载入数据
        for j,ses_id in enumerate(read_sessions):
            for k in range(15):
                data[j][i].append(sub_data[(ses_id-1)*15+k])
                label[j][i].append(sub_label[(ses_id-1)*15+k])

    return data,label


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
                seg_trail = np.zeros((num_sample, sample_length, len(trail[0])))
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