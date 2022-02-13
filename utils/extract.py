import re, os, ntpath
import numpy as np
from . import utils_func

channelmap = {
    'Xrotation': 'x',
    'Yrotation': 'y',
    'Zrotation': 'z'
}

channelmap_inv = {
    'x': 'Xrotation',
    'y': 'Yrotation',
    'z': 'Zrotation',
}

ordermap = {
    'x': 0,
    'y': 1,
    'z': 2,
}


class Anim(object):
    """
    A very basic animation object
    """
    def __init__(self, quats, pos, offsets, parents, bones):
        """
        :param quats: local quaternions tensor
        :param pos: local positions tensor
        :param offsets: local joint offsets
        :param parents: bone hierarchy
        :param bones: bone names
        """
        self.quats = quats
        self.pos = pos
        self.offsets = offsets
        self.parents = parents
        self.bones = bones


def read_bvh(filename, start=None, end=None, order=None):
    """
    Reads a BVH file and extracts animation information.
    解析bvh文件

    :param filename: BVh filename
    :param start: start frame
    :param end: end frame
    :param order: order of euler rotations  如xyz / zyx
    :return: A simple Anim object conatining the extracted information.
    """

    f = open(filename, "r")

    i = 0
    active = -1     # 父关节的下标，初始为-1
    end_site = False

    names = []  # 保存每个关节的名字（包括根关节和子关节）
    orients = np.array([]).reshape((0, 4))      # 方向
    offsets = np.array([]).reshape((0, 3))      # 偏移
    parents = np.array([], dtype=int)           # 父关节下标

    # Parse the  file, line by line 按行解析bvh文件
    for line in f:

        if "HIERARCHY" in line: continue
        if "MOTION" in line: continue

        rmatch = re.match(r"ROOT (\w+)", line)
        if rmatch:  # 匹配到ROOT行
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)     # 父关节值为0
            continue

        if "{" in line: continue

        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue

        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:    # 匹配到OFFSET行
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue

        chanmatch = re.match(r"\s*CHANNELS\s+(\d+)", line)
        if chanmatch:   # 匹配到CHANNELS
            channels = int(chanmatch.group(1))
            if order is None:   # 根据根关节或子关节的channel数量更新旋转的顺序
                channelis = 0 if channels == 3 else 3
                channelie = 3 if channels == 3 else 6
                parts = line.split()[2 + channelis:2 + channelie]
                if any([p not in channelmap for p in parts]):
                    continue
                order = "".join([channelmap[p] for p in parts])
            continue

        jmatch = re.match("\s*JOINT\s+(\w+)", line)
        if jmatch:  # 匹配JOINT行，表示有新的关节
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0) # 创建并插入新的偏移、旋转和父关节
            orients = np.append(orients, np.array([[1, 0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue

        if "End Site" in line:
            end_site = True
            continue

        # 先匹配到一个JOINT行，表示新的关节，则向offset数组中添加新的初始化位移信息，同理向orients添加旋转信息（四元数），向parents添加父关节下标
        # 然后匹配到OFFSET行，如果该OFFSET对应的关节不是end site，则根据该行给出的偏移信息更新offsets数组中对应行的数据
        # 最后匹配到CHANNELS行，分两种情况：1）匹配到根关节的channel  2）匹配到子关节的channel  根据根关节或子关节的channel数量更新旋转的顺序

        fmatch = re.match("\s*Frames:\s+(\d+)", line)
        if fmatch:
            if start and end:
                fnum = (end - start) - 1
            else:
                fnum = int(fmatch.group(1))     # 帧数
            positions = offsets[np.newaxis].repeat(fnum, axis=0)    # 每一帧的所有关节的位置 帧数 × 关节数 × 3（3d位置信息）
            rotations = np.zeros((fnum, len(orients), 3))   # 旋转信息：帧数 × 关节数 × 3（旋转角） 全0
            continue

        fmatch = re.match("\s*Frame Time:\s+([\d\.]+)", line)
        if fmatch:
            frametime = float(fmatch.group(1))
            continue

        if (start and end) and (i < start or i >= end - 1):
            i += 1
            continue

        dmatch = line.strip().split(' ')
        if dmatch:
            data_block = np.array(list(map(float, dmatch)))
            N = len(parents)    # 关节数
            fi = i - start if start else i
            if channels == 3:
                positions[fi, 0:1] = data_block[0:3]    # 每一帧的根关节的位移
                rotations[fi, :] = data_block[3:].reshape(N, 3)
            elif channels == 6:
                data_block = data_block.reshape(N, 6)
                positions[fi, :] = data_block[:, 0:3]
                rotations[fi, :] = data_block[:, 3:6]
            elif channels == 9:
                positions[fi, 0] = data_block[0:3]
                data_block = data_block[3:].reshape(N - 1, 9)
                rotations[fi, 1:] = data_block[:, 3:6]
                positions[fi, 1:] += data_block[:, 0:3] * data_block[:, 6:9]
            else:
                raise Exception("Too many channels! %i" % channels)

            i += 1  # 更新i

    f.close()

    rotations = utils_func.euler_to_quat(np.radians(rotations), order=order)
    rotations = utils_func.remove_quat_discontinuities(rotations)

    return Anim(rotations, positions, offsets, parents, names)


def get_lafan1_set(bvh_path, actors, window=50, offset=20):
    """
    Extract the same test set as in the article, given the location of the BVH files.

    :param bvh_path: Path to the dataset BVH files
    :param list: actor prefixes to use in set  subject1-4/subject 5
    :param window: width of the sliding windows (in timesteps) 滑动窗口的宽度？？？？
    :param offset: offset between windows (in timesteps)  窗口间的偏移？？？
    :return: tuple:
        X: local positions   局部位置 numpy数组  B, F, J, 3
        Q: local quaternions 局部四元数 numpy数组  B, F, J, 4
        parents: list of parent indices defining the bone hierarchy 每个关节的父关节的下标构成的列表
        contacts_l: binary tensor of left-foot contacts of shape (Batchsize, Timesteps, 2)
        contacts_r: binary tensor of right-foot contacts of shape (Batchsize, Timesteps, 2)
    """
    npast = 10
    subjects = []
    seq_names = []
    X = []
    Q = []
    contacts_l = []
    contacts_r = []

    # Extract
    bvh_files = os.listdir(bvh_path)

    for file in bvh_files:  # 对于单独的一个bvh文件
        if file.endswith('.bvh'):
            seq_name, subject = ntpath.basename(file[:-4]).split('_')

            if subject in actors:   # 当前的运动主体为subject1-4
                print('Processing file {}'.format(file))
                seq_path = os.path.join(bvh_path, file)
                anim = read_bvh(seq_path) # 读取一个给定路径的bvh文件，返回Anim类对象
                # anim:  Anim(rotations, positions, offsets, parents, names) 每一帧旋转信息，位置信息，每个关节的偏移，每个关节的父关节下标
                # anim.rotaions:  F, J, 4
                # anim.positions: F, J, 3
                # Sliding windows
                i = 0
                while i+window < anim.pos.shape[0]:
                    q, x = utils_func.quat_fk(anim.quats[i: i + window], anim.pos[i: i + window], anim.parents)
                    # Extract contacts
                    c_l, c_r = utils_func.extract_feet_contacts(x, [3, 4], [7, 8], velfactor=0.02)
                    X.append(anim.pos[i: i+window])
                    Q.append(anim.quats[i: i+window])
                    seq_names.append(seq_name)
                    subjects.append(subject)
                    contacts_l.append(c_l)
                    contacts_r.append(c_r)

                    i += offset

    X = np.asarray(X)   # B, F, J, 3
    Q = np.asarray(Q)   # B, F, J, 4
    contacts_l = np.asarray(contacts_l)
    contacts_r = np.asarray(contacts_r)

    # Sequences around XZ = 0
    xzs = np.mean(X[:, :, 0, ::2], axis=1, keepdims=True)
    X[:, :, 0, 0] = X[:, :, 0, 0] - xzs[..., 0]
    X[:, :, 0, 2] = X[:, :, 0, 2] - xzs[..., 1]

    # Unify facing on last seed frame
    X, Q = utils_func.rotate_at_frame(X, Q, anim.parents, n_past=npast)

    return X, Q, anim.parents, contacts_l, contacts_r


def get_train_stats(bvh_folder, train_set):
    """
    train_set: subject1-4
    Extract the same training set as in the paper in order to compute the normalizing statistics
    :return: Tuple of (local position mean vector, local position standard deviation vector, local joint offsets tensor)get_lafan1_set
    返回值：（局部位置的均值向量，局部位置的标准差向量，局部关节偏移tensor）
    """
    print('Building the train set...')
    xtrain, qtrain, parents, _, _ = get_lafan1_set(bvh_folder, train_set, window=50, offset=20)

    print('Computing stats...\n')
    # Joint offsets : are constant, so just take the first frame:
    offsets = xtrain[0:1, 0:1, 1:, :]  # Shape : (1, 1, J, 3) 子关节的偏移为常量，每一帧都相同，只取第1帧 不包括根关节的偏移

    # Global representation:
    q_glbl, x_glbl = utils_func.quat_fk(qtrain, xtrain, parents)

    # Global positions stats:
    x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
    x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)

    return x_mean, x_std, offsets
