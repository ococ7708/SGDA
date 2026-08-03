import fnmatch
import json
import os
import pickle
from scipy.io import loadmat
#从 SciPy 库中导入 loadmat 函数。loadmat 函数通常用于读取 MATLAB 的.mat 文件格式的数据，并将其转换为 Python 中的数据结构，例如字典等。
import numpy as np
import multiprocessing as mp
from functools import partial
import mne
import xmltodict
import pickle#可以将 Python 对象转换为字节流，以便保存到文件中或在网络上传输，也可以从字节流中恢复出原来的 Python 对象。

from data_utils.preprocess import preprocess, label_process
from .preprocess import lds
from .constants.riemann_preprocess import process_de_features_to_riemann

def get_data(setting=None, **kwargs):#setting默认值是None
    if setting is None:
        print(f"Error: Setting not set")

    # 以统一格式获取数据，加载数据集并集成到（session, subject, trail）格式中
    data, baseline, label, sample_rate, channels = get_uniform_data(setting.dataset, setting.dataset_path, **kwargs)

    use_riemann = kwargs.get('use_riemann', False)
    if use_riemann:
        # 黎曼变换已在 read_deap_preprocessed 中完成，数据已为切空间向量，跳过预处理
        all_data = data
        feature_dim = channels * (channels + 1) // 2
    else:
        # preprocess the eeg signal（带通滤波、基准校准、样本分割等）
        all_data, feature_dim = preprocess(data=data, baseline=baseline, sample_rate=sample_rate,
                                         pass_band=setting.pass_band, extract_bands=setting.extract_bands,
                                         sample_length=setting.sample_length, stride=setting.stride,
                                         time_window=setting.time_window, overlap=setting.overlap,
                                         only_seg=setting.only_seg if setting.dataset not in extract_dataset else True,
                                         feature_type=setting.feature_type,
                                         eog_clean=setting.eog_clean)

    all_data, all_label, num_classes = label_process(data=all_data, label=label, bounds=setting.bounds, onehot=setting.onehot, label_used=setting.label_used)
    return all_data, all_label, channels, feature_dim, num_classes


available_dataset = [  # 后缀没写lds则代表平滑方式是movingAVE seed_de即seed_de_movingAVE
    "seed_raw", "seediv_raw", "deap", "deap_raw", "hci", "dreamer", "seed_de", "seed_de_lds", "seed_psd", "seed_psd_lds", "seed_dasm", "seed_dasm_lds"
    , "seed_rasm", "seed_rasm_lds", "seed_asm", "seed_asm_lds", "seed_dcau", "seed_dcau_lds", "seediv_de_lds", "seediv_de_movingAve",
    "seediv_psd_movingAve", "seediv_psd_lds", "faced_de", "faced_psd", "faced_de_lds", "faced_psd_lds"
]

extract_dataset = {
    "seed_de", "seed_de_lds", "seed_psd", "seed_psd_lds", "seed_dasm", "seed_dasm_lds"
    , "seed_rasm", "seed_rasm_lds", "seed_asm", "see_und_asm_lds", "seed_dcau", "seed_dcau_lds", "seediv_de_lds", "seediv_de_movingAve",
    "seediv_psd_movingAve", "seediv_psd_lds", "faced_de", "faced_psd", "faced_de_lds", "faced_psd_lds"
}

def get_uniform_data(dataset, dataset_path, **kwargs):
    """
    Mainly aimed at the structure of different datasets,
    it is divided into the form of (session, subject, trail, channel, raw_data).
    :param dataset: the dataset used to train
    :param dataset_path: the dir of the dataset location
    :return: data, baseline, label, and sample rate of the original dataset
    """
    func = {  # 这是一个字典，键与函数对应
        "seed_raw": read_seed_raw,
        "deap": read_deap_preprocessed,
        "dreamer": read_dreamer,
        "deap_raw": read_deap_raw,
        "seediv_raw": read_seedIV_raw,
        "hci": read_hci
    }
    if dataset.startswith("seediv") and dataset != "seediv_raw":
        data, baseline, label, sample_rate, channels = read_seedIV_feature(dataset_path, feature_type=dataset[7:])
        # 提取从位置8开始到末尾的字段。 如dataset=seediv_de_lds，则dataset[7:]为de_lds
    elif dataset.startswith("seed") and not dataset.startswith("seediv") and dataset != "seed_raw":
        # call the read_seed_feature function when using the feature provided by seed official
        data, baseline, label, sample_rate, channels = read_seed_feature(dataset_path, feature_type=dataset[5:])
    elif dataset.startswith("faced") and dataset != "faced_raw":
        data, baseline, label, sample_rate, channels = read_faced_feature(dataset_path, feature_type=dataset[6:])
    else:
        data, baseline, label, sample_rate, channels = func[dataset](dataset_path, **kwargs)
    return data, baseline, label, sample_rate, channels

def read_faced_feature(dir_path, feature_type="de", label_type="emotion"):
    """
    input : 122 files(122 subjects)
    output : EEG signal with a trail as the basic unit 一个试次是一个基本单元
    output shape : (session(1), subject, trail, channel, raw_data), (session(1), subject, trail, label),
    Extract the EEG data of each subject from the faced dataset (feature)
    :param dir_path: The file location of the features in the faced dataset.
    :param feature_type: extract feature type
    :param label_type: Choose whether to use fine-grained("emotion"->9) or coarse-grained("valence"->3d) classification.
    :return: EEG features
    单个subject的de特征原始维度：视频数28×电极32×时长30×频段5
    单个subjet文件的label：3+3+3+3+4+3+3+3+3（negative 12 + neutral 4 + positive 12）
    """
    is_lds = feature_type.endswith("_lds")  # 返回bool
    # linear dynamic system approach.lds使用线性动态系统方法处理数据。

    dir_path += "/" + feature_type[:-4].upper() if is_lds else feature_type.upper()#upper()是大写的意思
    data = [[[] for _ in range(123)]]
    # emotion : Anger, Disgust, Fear, Sadness, Neutral, Amusement, Inspiration, Joy, Tenderness
    # valence : Negative, Neutral, Positive

    label = [[[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,4,5,5,5,6,6,6,7,7,7,8,8,8] if label_type =="emotion" else [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,2,2,2,2]  for _ in range(123)]]
    for i in range(123):
        file_path = dir_path + f"/sub{str(i).zfill(3)}.pkl.pkl"
        #f是格式化字符串字面值的标志 没有f就会被当作包含 {str(i).zfill(3)} 这个文本的字符串，而不是将 i 进行格式化计算后的结果。
        with open(file_path, "rb") as feature_file:
            sub_feature = pickle.load(feature_file)
            sub_feature = sub_feature.transpose(0, 2, 1, 3)
            if is_lds:
                for j in range(len(sub_feature)):
                    sub_feature[j] = lds(sub_feature[j])
            data[0][i] = sub_feature
    return data, None, label, None, 32

def read_seed_raw(dir_path):
    # input : 45 files(3 sessions, 15 round) containing all 15 trails with a sampling rate of 200 Hz
    # output : EEG signal with a trail as the basic unit and sample rate of the original dataset
    # output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label)

    # Extract the EEG data of each subject from the SEED dataset, and partition the data of each session
    dir_path += "/Preprocessed_EEG"
    eeg_files = [['1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
                  '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
                  '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
                  '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
                  '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],
                 ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
                  '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
                  '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
                  '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
                  '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],
                 ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
                  '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
                  '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
                  '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
                  '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']
                 ]
    # Extract the label for all trail in three sessions
    label = np.array(loadmat(f"{dir_path}/label.mat")['label'])
    labels = np.tile(label[0]+1, (3, 15, 1))

    # create the empty list of (3, 15, 15) => (session, subject, trail)
    eeg_data = [[[[] for _ in range(15)] for _ in range(15)] for _ in range(3)]
    # Loop processing of EEG mat files
    for session_files, session_id in zip(eeg_files, range(3)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            eeg_data[session_id] = pool.map(
                partial(parallel_read_seed_raw, dir_path), eeg_files[session_id])

    return eeg_data, None, labels, 200, 62

def parallel_read_seed_raw(dir_path, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    label_datas = []
    for i in range(15):
        trail_data = subject_data[keys[i]]
        trail_datas.append(trail_data[:,1:])
    return trail_datas



def read_seed_feature(dir_path, feature_type="de"):
    """
    input : 45 files(3 sessions, 15 round) containing all 15 trails with a sampling rate of 200 Hz
    output : EEG signal with a trail as the basic unit
    output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label),
    Extract the EEG data of each subject from the SEED dataset (feature) , and partition the data of each session
    :param dir_path: The file location of the features in the seed dataset.
    :param feature_type: extract feature type
    :return: the eeg features from seed dataset
    """

    dir_path += "/ExtractedFeatures"
    eeg_files = [['1_20131027.mat', '2_20140404.mat', '3_20140603.mat',
                  '4_20140621.mat', '5_20140411.mat', '6_20130712.mat',
                  '7_20131027.mat', '8_20140511.mat', '9_20140620.mat',
                  '10_20131130.mat', '11_20140618.mat', '12_20131127.mat',
                  '13_20140527.mat', '14_20140601.mat', '15_20130709.mat'],
                 ['1_20131030.mat', '2_20140413.mat', '3_20140611.mat',
                  '4_20140702.mat', '5_20140418.mat', '6_20131016.mat',
                  '7_20131030.mat', '8_20140514.mat', '9_20140627.mat',
                  '10_20131204.mat', '11_20140625.mat', '12_20131201.mat',
                  '13_20140603.mat', '14_20140615.mat', '15_20131016.mat'],
                 ['1_20131107.mat', '2_20140419.mat', '3_20140629.mat',
                  '4_20140705.mat', '5_20140506.mat', '6_20131113.mat',
                  '7_20131106.mat', '8_20140521.mat', '9_20140704.mat',
                  '10_20131211.mat', '11_20140630.mat', '12_20131207.mat',
                  '13_20140610.mat', '14_20140627.mat', '15_20131105.mat']
                 ]
    feature_index = {
        "de": 0, "de_lds": 1, "psd": 2, "psd_lds": 3, "dasm": 4, "dasm_lds": 5,
        "rasm": 6, "rasm_lds": 7, "asm": 8, "asm_lds": 9, "dcau": 10, "dcau_lds": 11
    }

    # 读取mat 文件，返回Python字典结构，再访问键名label的元素，#loadmat 返回数据或为MATLAB特殊格式，转换为NumPy数组便于后续处理
    label = np.array(loadmat(f"{dir_path}/label.mat")['label'])  # 形状为 (1, 15) 的二维数组: array([[1, 0, -1, ...]])
    # (3, 15, 1)即沿三个维度的复制次数。结果形状：(3, 15, 15) 3个session×15个subject × 15个trials
    label = np.tile(label[0] + 1, (3, 15, 1))

    fi = feature_index[feature_type]  # 特性类型-->特征类型数值编号

    eeg_data = [[] for _ in range(3)]
    # Define a function to read a single MAT file
    for session_files, session_id in zip(eeg_files, range(3)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            result_session = pool.map(  # 并行地将新函数parallel_read_seed_feature应用于可迭代对象eeg_files[session_id]的每个元素
                partial(parallel_read_seed_feature, fi, dir_path), eeg_files[session_id])
        for i in range(15):  # 每个session有15个被试
            eeg_data[session_id].append(result_session[i])  # 为每个session内部填上数据
    return eeg_data, None, label, None,  62
    # 以SEED为例 eeg_data维度是3×15×15×235×62×5 完全的list
    # label是3×15×15 每层都是numpy数组 label[0][0][0]是int


def parallel_read_seed_feature(fi, dir_path, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    # [3:] 切片操作跳过前 3 个系统键（通常是 '__header__', '__version__', '__globals__'）
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(15):  # 每个被试文件有15个trail
        '''
        每个 trial 有 12 种不同的特征（对应 feature_index 字典的 12 个选项）
        transpose:(通道，时间点，频段)->（时间点，通道，频段）将时间维度放在最外层，便于计算各时间点的跨通道特征
        例： SEED中62×235×5 变为 235×62×5
        '''
        trail_data = list(np.array(subject_data[keys[i * 12+fi]]).transpose((1, 0, 2)))
        trail_datas.append(trail_data)
    return trail_datas

def read_seedIV_raw(dir_path):
    # input : 45 files(3 sessions, 15 round)
    # output : EEG signal with a trail as the basic unit and sample rate of the original dataset
    # output shape : (session, subject, trail, channel, raw_data), (session, subject, trail, label)

    dir_path += "/eeg_raw_data"
    eeg_files = [['1_20160518.mat', '2_20150915.mat', '3_20150919.mat',
                  '4_20151111.mat', '5_20160406.mat', '6_20150507.mat',
                  '7_20150715.mat', '8_20151103.mat', '9_20151028.mat',
                  '10_20151014.mat', '11_20150916.mat', '12_20150725.mat',
                  '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
                 ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat',
                  '4_20151118.mat', '5_20160413.mat', '6_20150511.mat',
                  '7_20150717.mat', '8_20151110.mat', '9_20151119.mat',
                  '10_20151021.mat', '11_20150921.mat', '12_20150804.mat',
                  '13_20151125.mat', '14_20151208.mat', '15_20150514.mat', ],
                 ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat',
                  '4_20151123.mat', '5_20160420.mat', '6_20150512.mat',
                  '7_20150721.mat', '8_20151117.mat', '9_20151209.mat',
                  '10_20151023.mat', '11_20151011.mat', '12_20150807.mat',
                  '13_20161130.mat', '14_20151215.mat', '15_20150527.mat', ]
                 ]

    # exctract the label for all trail in three sessions, label shape : (3, 24)
    label = np.zeros((3, 15, 24), dtype=int)
    ses_label1 = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
    ses_label2 = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
    ses_label3 = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    ses_label1s = np.tile(ses_label1, (1, 15, 1))
    ses_label2s = np.tile(ses_label2, (1, 15, 1))
    ses_label3s = np.tile(ses_label3, (1, 15, 1))
    label[0] = ses_label1s
    label[1] = ses_label2s
    label[2] = ses_label3s

    # Add a father session folder to each file
    for i, session in enumerate(eeg_files):
        eeg_files[i] = [f"{i + 1}/{sub_file}" for sub_file in session]

    # create the empty list of (3, 15, 24) => (session, subject, trail)
    eeg_data = [[[[] for _ in range(24)] for _ in range(15)] for _ in range(3)]
    # Loop processing of EEG mat files
    for session_files, session_id in zip(eeg_files, range(3)):
        # Create a pool of worker processes
        with mp.Pool(processes=5) as pool:
            # Map the parallel_read_seed_feature function to each file in the list
            eeg_data[session_id] = pool.map(
                partial(parallel_read_seedIV_raw, dir_path), eeg_files[session_id])
    return eeg_data, None, label, 200, 62

def parallel_read_seedIV_raw(dir_path, file):
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(24):
        trail_data = subject_data[keys[i]]
        trail_datas.append(trail_data[:,1:])
    return trail_datas


def read_seedIV_feature(dir_path, feature_type="de_lds"):
    # 读取seed IV数据集
    # input file : three folder each contains one session of 15 subjects' eeg data
    # output shape : (session(3), subject, trail, channel, feature), (session(3), subject, trail, label)
    # use the feature under eeg_feature_smooth dir, it has 3 dir, each dir represent 15 subejct
    # in each dir, it contains 15 subject files
    dir_path += "/eeg_feature_smooth"
    eeg_files = [['1_20160518.mat', '2_20150915.mat', '3_20150919.mat',
                  '4_20151111.mat', '5_20160406.mat', '6_20150507.mat',
                  '7_20150715.mat', '8_20151103.mat', '9_20151028.mat',
                  '10_20151014.mat', '11_20150916.mat', '12_20150725.mat',
                  '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
                 ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat',
                  '4_20151118.mat', '5_20160413.mat', '6_20150511.mat',
                  '7_20150717.mat', '8_20151110.mat', '9_20151119.mat',
                  '10_20151021.mat', '11_20150921.mat', '12_20150804.mat',
                  '13_20151125.mat', '14_20151208.mat', '15_20150514.mat',],
                 ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat',
                  '4_20151123.mat', '5_20160420.mat', '6_20150512.mat',
                  '7_20150721.mat', '8_20151117.mat', '9_20151209.mat',
                  '10_20151023.mat', '11_20151011.mat', '12_20150807.mat',
                  '13_20161130.mat', '14_20151215.mat', '15_20150527.mat', ]
                 ]

    #exctract the label for all trail in three sessions, label shape : (3, 24)
    label = np.zeros((3,15,24), dtype=int)
    ses_label1 = [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3]
    ses_label2 = [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1]
    ses_label3 = [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0]
    ses_label1s = np.tile(ses_label1, (1,15,1))
    ses_label2s = np.tile(ses_label2, (1,15,1))
    ses_label3s = np.tile(ses_label3, (1,15,1))
    label[0] = ses_label1s
    label[1] = ses_label2s
    label[2] = ses_label3s

    # Add a father session folder to each file
    for i, session in enumerate(eeg_files):
        eeg_files[i] = [f"{i+1}/{sub_file}" for sub_file in session]

    feature_index = {
        "de_movingAve": 0, "de_lds": 1, "psd_movingAve": 2, "psd_lds": 3
    }
    fi = feature_index[feature_type]

    eeg_data = [[] for _ in range(3)]
    # Define a function to read a single Mat file
    for ses_id, session_files in enumerate(eeg_files):
        with mp.Pool(processes=5) as pool:
            result_session = pool.map(
                partial(parallel_read_seedIV_feature, fi, dir_path, label), eeg_files[ses_id]
            )
        for i in range(15):
            eeg_data[ses_id].append(result_session[i])
    return eeg_data, None, label, None, 62
def parallel_read_seedIV_feature(fi, dir_path, label, file):
    subject_data = loadmat(f"{dir_path}/{file}")
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(24):
        trail_data = list(np.array(subject_data[keys[i*4+fi]].transpose((1,0,2))))
        trail_datas.append(trail_data)
    return trail_datas


def read_deap_preprocessed(dir_path, use_riemann=False, riemann_metric='riemann'):
    """
    读取 DEAP 预处理数据（.dat 格式）

    Parameters:
    -----------
    dir_path : str
        数据目录路径
    use_riemann : bool, default=False
        是否应用黎曼几何预处理
    riemann_metric : str, default='riemann'
        黎曼度量类型：'riemann' 或 'logeuclid'

    Returns:
    --------
    data : list
        shape: (session=1, subject=32, trial=40, channel=32, time=7680)
    baseline : None
    label : list
        shape: (session=1, subject=32, trial=40, label_dim=4)
    sample_rate : int
        采样率 128Hz
    channels : int
        EEG 通道数 32
    """
    ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4',
                'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8',
                'PO4', 'O2']

    data = [[]]
    label = [[]]
    fs = 128
    pre_time = 3
    end_time = 63
    pretrail = pre_time * fs

    eeg_files = ["s{}.dat".format(str(i).zfill(2)) for i in range(1, 33)]

    for s_i, subject_file in enumerate(eeg_files):
        full_path = os.path.join(dir_path, subject_file)
        if not os.path.exists(full_path):
            print(f"[WARN] File not found: {full_path}")
            continue

        sub_data = pickle.load(open(full_path, "rb"), encoding="latin")

        # 基线校正
        baseline = np.mean([sub_data['data'][:, :32, i * fs:(i + 1) * fs] for i in range(3)], axis=0)
        for sec in range(pre_time, end_time):
            sub_data['data'][:, :32, sec * fs: (sec + 1) * fs] -= baseline

        sub_data_list = []
        sub_label_list = []
        for t_i, (trail_data, trail_label) in enumerate(zip(sub_data['data'], sub_data['labels'])):
            # 只取前 32 个 EEG 通道，去掉前 3 秒基线
            sub_data_list.append(trail_data[:32, pretrail:])  # shape: (32, 7680)
            sub_label_list.append(trail_label)

        data[0].append(sub_data_list)
        label[0].append(sub_label_list)

    # ===== 黎曼几何预处理（可选）=====
    if use_riemann:
        print("[Riemann] Applying Tangent Space Transformation...")

        n_sessions = len(data)
        n_subjects = len(data[0])
        riemann_data = []

        # 获取维度信息
        sample_trial = np.array(data[0][0][0])  # (n_channels, n_time_points)
        n_channels = sample_trial.shape[0]

        for ses_idx in range(n_sessions):
            ses_data = []
            for sub_idx in range(n_subjects):
                # 获取该被试的所有 trial: list of (32, 7680)
                subject_trials = data[ses_idx][sub_idx]
                subject_array = np.array(subject_trials)  # (n_trials, 32, 7680)

                # 转换为黎曼切空间特征
                # 注意：这里需要将时间序列视为"频段"来计算协方差
                # 或者先提取频域特征再计算协方差
                tangent_feats = process_de_features_to_riemann(
                    subject_array, metric=riemann_metric, align=True, reg=1e-6
                )

                tangent_list = [tangent_feats[i:i+1] for i in range(len(tangent_feats))]
                ses_data.append(tangent_list)
            riemann_data.append(ses_data)

        data = riemann_data
        print(f"[Riemann] Transform complete. Tangent dim: {tangent_feats.shape[1]}")

    return data, None, label, 128, 32


def read_deap_raw(dir_path):
    """
    读取 DEAP 原始数据（.bdf 格式）

    Returns:
    --------
    all_raw_data : list
        shape: (session=1, subject=32, trial=40, channel=32, time=7680)
    None : baseline placeholder
    label : list
        shape: (session=1, subject=32, trial=40, label_dim=4)
    sample_rate : int
        原始采样率 512Hz
    channels : int
        EEG 通道数 32
    """
    import mne  # 需要安装: pip install mne

    Geneva_ch_names = ['Fp1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1',
                       'P3', 'P7', 'PO3', 'O1', 'Oz', 'Pz', 'Fp2', 'AF4', 'Fz', 'F4',
                       'F8', 'FC6', 'FC2', 'Cz', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8',
                       'PO4', 'O2']
    Twente_ch_names = ['Fp1', 'AF3', 'F7', 'F3', 'FC1', 'FC5', 'T7', 'C3', 'CP1', 'CP5',
                       'P7', 'P3', 'Pz', 'PO3', 'O1', 'Oz', 'O2', 'PO4', 'P4', 'P8',
                       'CP6', 'CP2', 'C4', 'T8', 'FC6', 'FC2', 'F4', 'F8', 'AF4',
                       'Fp2', 'Fz', 'Cz']
    transfer_index = [Twente_ch_names.index(s) for s in Geneva_ch_names]

    fs = 512
    pre_time = 3
    end_time = 63
    pretrail = pre_time * fs

    # 状态码定义
    start_code1 = 4
    start_code2 = 1638148
    start_code3 = 5832452

    eeg_files = ["s{}.bdf".format(str(i).zfill(2)) for i in range(1, 33)]
    label_file = ["s{}.dat".format(str(i).zfill(2)) for i in range(1, 33)]

    all_raw_data = [[]]
    label = [[]]

    for s_i, subject_file in enumerate(eeg_files):
        # 读取原始.bdf 文件
        bdf_path = os.path.join(dir_path, "data_original", subject_file)
        if not os.path.exists(bdf_path):
            print(f"[WARN] BDF file not found: {bdf_path}")
            continue

        sub_bdf_data = mne.io.read_raw_bdf(bdf_path, preload=True, verbose=False)

        # 读取标签（从预处理的.dat 文件获取）
        label_path = os.path.join(dir_path, "data_preprocessed_python", label_file[s_i])
        label_data = pickle.load(open(label_path, "rb"), encoding="latin")['labels']

        # 读取状态通道
        status = np.array(sub_bdf_data.get_data()[47]).astype(int)
        changes = np.diff(status) != 0
        changes = np.insert(changes, 0, True)
        indices = np.where(changes)[0]

        # 读取前 32 个 EEG 通道
        raw_data = np.array(sub_bdf_data.get_data()[:32])

        sub_raw_data = []
        sub_label = []
        pre_code = 0

        for begin, end in zip(indices, np.append(indices[1:], len(status))):
            if pre_code in [start_code1, start_code2, start_code3]:
                # 提取 60 秒数据
                trail_raw_data = raw_data[:32, end - 60 * fs:end].tolist()

                # 前 22 个被试需要重排通道顺序
                if s_i < 22:
                    trail_raw_data = [trail_raw_data[tmp_i] for tmp_i in transfer_index]

                sub_raw_data.append(trail_raw_data)
            pre_code = status[begin]

        for t_i, trail_label in enumerate(label_data):
            sub_label.append(trail_label)

        all_raw_data[0].append(sub_raw_data)
        label[0].append(sub_label)

    return all_raw_data, None, label, 512, 32


def read_dreamer(dir_path, last_seconds = 60, base_seconds = 4):
    # input : 1 file (23 subjects' data)
    # subject data struct :
    #   Age, Gender, EEG, ECG, Valence(18 * 1), Arousal(18 * 1), Dominance(18 * 1)
    # subject's EEG data struct:
    #   sample rate : 128, num of electrodes : 14, num of subjects : 23
    #   electrodes : { 'AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'}
    # output shape : (session(1), subject, trail, channel, raw_data)
    file_path = dir_path + "/DREAMER.mat"
    data = loadmat(file_path)["DREAMER"]
    # data : [Data, EEG_sample_rate, ECG_sample_rate, EEG_electrodes, noOfSubjects, noOfVideoSequences
    # , Disclaimer, Provider, Version, Acknowledgement]
    # Data : [Age, Gender, EEG, ECG, ScoreValence, ScoreArousal, ScoreDominance]
    # EEG : [baseline, stimuli]
    # baseline & stimuli : [18, 1]
    #
    all_stimuli = [[[[] for _ in range(18)] for _ in range(23)]]
    all_base = [[[[] for _ in range(18)] for _ in range(23)]]
    all_labels = [[[[] for _ in range(18)] for _ in range(23)]]
    for subject in range(23):
        for trail in range(18):
            trail_stim = data[0,0]["Data"][0, subject]["EEG"][0, 0]["stimuli"][0, 0][trail, 0]
            trail_base = data[0,0]["Data"][0, subject]["EEG"][0, 0]["baseline"][0, 0][trail, 0]
            trail_valence = data[0,0]["Data"][0, subject]["ScoreValence"][0, 0][trail, 0]
            trail_arousal = data[0,0]["Data"][0, subject]["ScoreArousal"][0, 0][trail, 0]
            trail_dominance = data[0, 0]["Data"][0, subject]["ScoreDominance"][0, 0][trail, 0]
            trail_label = np.array([trail_valence, trail_arousal, trail_dominance])
            # print(trail_stim)
            # trail_stim shape : [128 * seconds(199), channel(14)]
            # trail_label shape : [3]
            all_stimuli[0][subject][trail] = trail_stim[-last_seconds*128:].transpose()
            all_base[0][subject][trail] = trail_base[-base_seconds*128:].transpose()
            all_labels[0][subject][trail] = trail_label
            # all_stimuli[0][subject][trail] shape : [channel(14), seconds(last_seconds) * sample rate(128)]
            # all_labels[0][subject][trail] shape : [3]
    return all_stimuli, all_base, all_labels, 128, 14


def read_hci(dir_path):
    # 30 subjects, [20, 20, 17, 20, 20, 20, 20, 20, 14, 20, 20, 0, 20, 20, 0, 16, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    # input : 1 dir ( contains 1200 file )
    # output shape (session(1), subject, trail, channel, raw_data)
    baseline_sec = 30
    dir_path = dir_path + "/Sessions/"
    file_names = [name for name in os.listdir(dir_path)]
    emo_states = ['@feltVlnc', '@feltArsl']
    data = [[[] for _ in range(30)]]
    base = [[[] for _ in range(30)]]
    labels = [[[] for _ in range(30)]]

    for file in file_names:
        sub_dir = dir_path + file
        label_file = sub_dir + "/session.xml"
        with open(label_file) as f:
            label_info = xmltodict.parse('\n'.join(f.readlines()))
        label_info = json.loads(json.dumps(label_info))["session"]
        if not '@feltArsl' in label_info:
            continue
        trail_label = np.array([int(label_info[k]) for k in emo_states])
        sub = int(label_info['subject']['@id'])
        trail_file = [sub_dir+"/"+f for f in os.listdir(sub_dir) if fnmatch.fnmatch(f,'*.bdf')][0]
        raw = mne.io.read_raw_bdf(trail_file, preload=True, stim_channel='Status', verbose=False)
        events = mne.find_events(raw, stim_channel='Status', verbose=False)
        montage = mne.channels.make_standard_montage(kind='biosemi32')
        raw.set_montage(montage, on_missing='ignore')
        raw.pick(raw.ch_names[:32])
        start_samp, end_samp = events[0][0] + 1, events[1][0] - 1
        baseline = raw.copy().crop(raw.times[0], raw.times[end_samp])
        baseline = baseline.resample(128)
        baseline_data = baseline.to_data_frame().to_numpy()[:, 1:].swapaxes(1, 0)
        baseline_data = baseline_data[:, :baseline_sec * 128]
        baseline_data = baseline_data.reshape(32, baseline_sec, 128).mean(axis=1)

        trail_bdf = raw.copy().crop(raw.times[start_samp], raw.times[end_samp])
        trail_bdf = trail_bdf.resample(128)
        trail_data = trail_bdf.to_data_frame().to_numpy()[:,1:].swapaxes(1,0)
        data[0][sub-1].append(trail_data)
        base[0][sub-1].append(baseline_data)
        labels[0][sub-1].append(trail_label)

    filter_d_l_b = [(d,l,b) for d,l,b in zip(data[0], labels[0], base[0]) if l != []]
    data[0], labels[0], base[0] = zip(*filter_d_l_b) if filter_d_l_b else ([],[],[])
    return data, base, labels, 128, 32
