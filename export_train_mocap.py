'''
数据预处理
读取一个数据集里面的所有bvh动画文件
'''
import os
import sys
import numpy as np
import yaml
import argparse
from os.path import join as pjoin
from utils import extract
from utils import utils_func


BASEPATH = os.path.dirname(os.path.abspath(__file__))   # 当前路径
sys.path.insert(0, BASEPATH)
sys.path.insert(0, pjoin(BASEPATH, '..'))
sys.path.insert(0, pjoin(BASEPATH, '..', '..'))

def get_bvh_files(directory):
    return [os.path.join(directory, f) for f in sorted(list(os.listdir(directory)))
            if os.path.isfile(os.path.join(directory, f))
            and f.endswith('.bvh') and f != 'rest.bvh']

def set_init(dic, key, value):
    try:
        dic[key]
    except KeyError:
        dic[key] = value


def generate_database_xia(bvh_path, dataset_config='xia_dataset.yml'):
    '''
    读取xia数据集里面的所有bvh文件,划分训练集和测试集，保存结果
    '''
    npast = 10
    with open(dataset_config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)      # 读取xia数据集的配置文件
    content_namedict = [full_name.split('_')[0] for full_name in cfg["content_full_names"]] # 内容信息：28类风格
    content_test_cnt = cfg["content_test_cnt"]  # 每种内容用于测试的文件数量

    bvh_files = get_bvh_files(bvh_path)     # 获取所有bvh文件的路径名

    test_cnt = {}  # indexed by content_style
    test_X = []                  # 测试集的位置
    test_Q = []                  # 测试集的旋转
    test_content_style = []      # 测试集的风格和内容
    train_X = []                 # 训练集的位置
    train_Q = []                 # 训练集的旋转
    train_window = 50
    train_offset = 20
    test_window = 50
    test_offset = 20

    for i, item in enumerate(bvh_files):
        filename = item.split('/')[-1]
        style, content_idx, _ = filename.split('_')         # 风格、内容下标
        content = content_namedict[int(content_idx) - 1]    # 内容下标转为内容名称
        content_style = "%s_%s" % (content, style)          # 该文件的内容+风格

        anim = extract.read_bvh(item)
        print('Processing %i of %i (%s) %s.' % (i, len(bvh_files), item, filename))
        # Whether this should be a test clip
        set_init(test_cnt, content_style, 0)
        if test_cnt[content_style] < content_test_cnt[content]: # 成为测试片段
            # 按照测试窗口划分
            i = 0
            while i + test_window < anim.pos.shape[0]: # fnum
                test_X.append(anim.pos[i: i + test_window])
                test_Q.append(anim.quats[i: i + test_window])
                test_content_style.append(content_style)
                i += test_offset
            test_cnt[content_style] += 1
        else:       # 成为训练片段，按照窗口划分
            i = 0
            while i + train_window < anim.pos.shape[0]:  # fnum
                train_X.append(anim.pos[i: i + train_window])
                train_Q.append(anim.quats[i: i + train_window])
                i += train_offset

    train_X = np.asarray(train_X)     # 训练集位置 [B, F, J, 3]
    train_Q = np.asarray(train_Q)     # 训练集旋转 [B, F, J, 4]
    test_X = np.asarray(test_X)       # 测试集位置 [B, F, J, 3]
    test_Q = np.asarray(test_Q)       # 测试集旋转 [B, F, J, 4]
    test_content_style = np.asarray(test_content_style)   # 测试集内容和风格信息

    # 输出信息：
    print("train_X.shape:",train_X.shape)
    print("train_Q.shape:",train_Q.shape)
    print("test_X.shape:",test_X.shape)
    print("test_Q.shape:",test_Q.shape)
    print("test_content_style.shape:",test_content_style.shape)

    # Sequences around XZ = 0
    xzs_train = np.mean(train_X[:, :, 0, ::2], axis=1, keepdims=True)
    train_X[:, :, 0, 0] = train_X[:, :, 0, 0] - xzs_train[..., 0]
    train_X[:, :, 0, 2] = train_X[:, :, 0, 2] - xzs_train[..., 1]

    xzs_test = np.mean(test_X[:, :, 0, ::2], axis=1, keepdims=True)
    test_X[:, :, 0, 0] = test_X[:, :, 0, 0] - xzs_test[..., 0]
    test_X[:, :, 0, 2] = test_X[:, :, 0, 2] - xzs_test[..., 1]

    # Unify facing on last seed frame
    train_X, train_Q = utils_func.rotate_at_frame(train_X, train_Q, anim.parents, n_past=npast)
    test_X, test_Q = utils_func.rotate_at_frame(test_X, test_Q, anim.parents, n_past=npast)

    return train_X, train_Q, test_X, test_Q, test_content_style, anim.parents




def generate_database_bfa(bvh_path):
    npast = 10

    bvh_files = get_bvh_files(bvh_path) # 获取所有bvh文件的路径名

    test_X = []  # 测试集的位置
    test_Q = []  # 测试集的旋转
    test_style = []  # 测试集的风格
    train_X = []  # 训练集的位置
    train_Q = []  # 训练集的旋转
    train_window = 50
    train_offset = 20
    test_window = 50
    test_offset = 20

    group_size = 10  # pick the last clip from every group_size clips for test  对于每个bvh文件，每10个片段就挑选最后一个片段（第10个）作为测试片段
    group_length = test_window * group_size  # 一组片段（10个）的总帧数
    print("group_length", group_length)
    for i, item in enumerate(bvh_files):
        filename = item.split('/')[-1]
        style, _ = filename.split('_')
        anim = extract.read_bvh(item)
        print('Processing %i of %i (%s) %s.' % (i, len(bvh_files), item, filename))
        i = 0
        count = 0
        while i + train_window < anim.pos.shape[0]:
            count += 1
            if count %10 == 0 and i + test_window < anim.pos.shape[0]:  # test
                test_X.append(anim.pos[i: i + test_window])
                test_Q.append(anim.quats[i: i + test_window])
                test_style.append(style)
                i += test_offset
            else:   # train
                train_X.append(anim.pos[i: i + train_window])
                train_Q.append(anim.quats[i: i + train_window])
                i += train_offset

    train_X = np.asarray(train_X)  # 训练集位置 [B, F, J, 3]
    train_Q = np.asarray(train_Q)  # 训练集旋转 [B, F, J, 4]
    test_X = np.asarray(test_X)  # 测试集位置 [B, F, J, 3]
    test_Q = np.asarray(test_Q)  # 测试集旋转 [B, F, J, 4]
    test_style = np.asarray(test_style)  # 测试集风格信息

    # 输出信息：
    print("train_X.shape:", train_X.shape)
    print("train_Q.shape:", train_Q.shape)
    print("test_X.shape:", test_X.shape)
    print("test_Q.shape:", test_Q.shape)
    print("test_style.shape:", test_style.shape)

    # Sequences around XZ = 0
    xzs_train = np.mean(train_X[:, :, 0, ::2], axis=1, keepdims=True)
    train_X[:, :, 0, 0] = train_X[:, :, 0, 0] - xzs_train[..., 0]
    train_X[:, :, 0, 2] = train_X[:, :, 0, 2] - xzs_train[..., 1]

    xzs_test = np.mean(test_X[:, :, 0, ::2], axis=1, keepdims=True)
    test_X[:, :, 0, 0] = test_X[:, :, 0, 0] - xzs_test[..., 0]
    test_X[:, :, 0, 2] = test_X[:, :, 0, 2] - xzs_test[..., 1]

    # Unify facing on last seed frame
    train_X, train_Q = utils_func.rotate_at_frame(train_X, train_Q, anim.parents, n_past=npast)
    test_X, test_Q = utils_func.rotate_at_frame(test_X, test_Q, anim.parents, n_past=npast)

    return train_X, train_Q, test_X, test_Q, test_style, anim.parents


def parse_args():
    parser = argparse.ArgumentParser("export_train_mocap")
    parser.add_argument("--xia_bvh_path", type=str, default="./data/mocap_xia")    # 数据集xia路径
    parser.add_argument("--bfa_bvh_path", type=str, default="./data/mocap_bfa")    # 数据集bfa路径
    parser.add_argument("--output_path", type=str, default="./data/")               # 数据预处理结果路径
    parser.add_argument("--xia_dataset_config", type=str, default='./config/xia_dataset.yml') # 数据集配置文件
    parser.add_argument("--bfa_dataset_config", type=str, default='./config/bfa_dataset.yml') # 数据集配置文件
    return parser.parse_args()


def main(args):
    train_X_xia, train_Q_xia, test_X_xia, test_Q_xia, test_content_style_xia, parents_xia = generate_database_xia(bvh_path=args.xia_bvh_path,
                          dataset_config=args.xia_dataset_config)
    train_X_bfa, train_Q_bfa, test_X_bfa, test_Q_bfa, test_style_bfa, parents_bfa = generate_database_bfa(bvh_path=args.bfa_bvh_path)

    test_X =  np.concatenate((test_X_xia, test_X_bfa),axis=0)  # 测试集的位置
    test_Q = np.concatenate((test_Q_xia, test_Q_bfa),axis=0)  # 测试集的旋转
    test_content_style = np.concatenate((test_content_style_xia, test_style_bfa),axis=0)  # 测试集的风格
    train_X = np.concatenate((train_X_xia, train_X_bfa), axis=0)  # 训练集的位置
    train_Q = np.concatenate((train_Q_xia, train_Q_bfa), axis=0)  # 训练集的旋转
    print("train_X.shape:", train_X.shape)
    print("train_Q.shape:", train_Q.shape)
    print("test_X.shape:", test_X.shape)    # B, F, J, 3
    print("test_Q.shape:", test_Q.shape)    # B, F, J, 4
    print("test_content_style.shape:", test_content_style.shape)

    train_q_glbl, train_x_glbl = utils_func.quat_fk(train_Q, train_X, parents_xia)
    train_x_mean = np.mean(train_x_glbl.reshape([train_x_glbl.shape[0], train_x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2),
                     keepdims=True)
    train_x_std = np.std(train_x_glbl.reshape([train_x_glbl.shape[0], train_x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2),
                   keepdims=True)

    test_q_glbl, test_x_glbl = utils_func.quat_fk(test_Q, test_X, parents_xia)
    test_x_mean = np.mean(test_x_glbl.reshape([test_x_glbl.shape[0], test_x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2),
                    keepdims=True)
    test_x_std = np.std(test_x_glbl.reshape([test_x_glbl.shape[0], test_x_glbl.shape[1], -1]).transpose([0, 2, 1]),
                         axis=(0, 2),
                         keepdims=True)

    np.savez_compressed(args.output_path + 'mocap_train.npz', X=train_X, Q=train_Q, x_mean = train_x_mean, x_std = train_x_std)
    np.savez_compressed(args.output_path + 'mocap_test.npz', X=test_X, Q=test_Q, style=test_content_style, x_mean = test_x_mean, x_std = test_x_std)


if __name__ == '__main__':
    args = parse_args()
    main(args)      # 数据预处理

