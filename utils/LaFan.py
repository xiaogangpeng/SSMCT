import torch
from torch.utils.data import Dataset, DataLoader

import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.append("..")

import numpy as np
from utils import extract, utils_func

class LaFan1(Dataset):
    def __init__(self, bvh_path, dataset, train = False, seq_len = 50, offset = 20, debug = False):
        """
        Args:
            bvh_path (string): Path to the bvh files.
            seq_len (int): The max len of the sequence for interpolation.
        """
        self.train = train
        self.seq_len = seq_len
        self.debug = debug
        self.offset = offset
        self.cur_seq_length = 5
        self.dataset = dataset

        if dataset == "mocap":
            self.data = self.load_mocap_data(bvh_path)
        else:
            if self.debug:
                self.actors = ['subject1']
            elif train:
                self.actors = ['subject1', 'subject2', 'subject3', 'subject4']
            else:
                self.actors = ['subject5']
            self.data = self.load_data(bvh_path)


    def load_data(self, bvh_path):
        # Get test-set for windows of 65 frames, offset by 40 frames
        # 根据bvh_path构造数据集（训练集或测试集）
        print('Building the data set...')
        # X: local positions   局部位置 numpy数组  [B, F, J, 3]
        # Q: local quaternions 局部四元数 numpy数组  [B, F, J, 4]
        # parents: list of parent indices defining the bone hierarchy 每个关节的父关节的下标构成的列表
        X, Q, parents, contacts_l, contacts_r = extract.get_lafan1_set(\
                                                bvh_path, self.actors, window=self.seq_len, offset=self.offset)
        # Global representation: 获取全局位置
        q_glbl, x_glbl = utils_func.quat_fk(Q, X, parents)   # x_glbl: B, F, J, 3  q_glbl: B, F, J, 4

        # if self.train:
        # Global positions stats:
        x_mean = np.mean(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        x_std = np.std(x_glbl.reshape([x_glbl.shape[0], x_glbl.shape[1], -1]).transpose([0, 2, 1]), axis=(0, 2), keepdims=True)
        self.x_mean = torch.from_numpy(x_mean)
        self.x_std = torch.from_numpy(x_std)

        input_ = {}
        # The following features are inputs:
        # 1. local quaternion vector (J * 4d)   局部旋转四元数向量
        input_['local_q'] = Q   # [B, F, J, 4]
        
        # # 2. global root velocity vector (3d)
        # input_['root_v'] = x_glbl[:,1:,0,:] - x_glbl[:,:-1,0,:]
        #
        # # 3. contact information vector (4d)
        # input_['contact'] = np.concatenate([contacts_l, contacts_r], -1)
        #
        # # 4. global root position offset (?d)
        # input_['root_p_offset'] = x_glbl[:,-1,0,:]
        #
        # # 5. local quaternion offset (?d)
        # input_['local_q_offset'] = Q[:,-1,:,:]
        #
        # # 6. target
        # input_['target'] = Q[:,-1,:,:]
        #
        # # 7. root pos 全局根关节位移 B, F, 3
        # input_['root_p'] = x_glbl[:,:,0,:]

        # 8. X   全局位移  B, F, J, 3
        input_['X'] = x_glbl[:,:,:,:]   #

        print('Nb of sequences : {}\n'.format(X.shape[0]))

        return input_

    def load_mocap_data(self, bvh_path):
        # Get test-set for windows of 65 frames, offset by 40 frames
        # read train or test data from file .npz
        print('Building the data set...')

        data = np.load(bvh_path)
        input_ = {}
        input_['local_q'] = data['Q']  # local quaternion vector  [B, F, J, 4]
        input_['X'] = data['X']  # global positions [B, F, J, 3]
        self.x_mean = data['x_mean']
        self.x_std = data['x_std']
        self.x_mean = torch.from_numpy(self.x_mean)
        self.x_std = torch.from_numpy(self.x_std)
        if self.train == False:
            input_['style'] = data['style']
        return input_

    def __len__(self):
        return len(self.data['local_q'])

    def __getitem__(self, idx):
        idx_ = None
        if self.debug:
            idx_ = 0
        else:
            idx_ = idx
        sample = {}
        sample['local_q'] = self.data['local_q'][idx_].astype(np.float32)
        # sample['root_v'] = self.data['root_v'][idx_].astype(np.float32)
        # sample['contact'] = self.data['contact'][idx_].astype(np.float32)
        # sample['root_p_offset'] = self.data['root_p_offset'][idx_].astype(np.float32)
        # sample['local_q_offset'] = self.data['local_q_offset'][idx_].astype(np.float32)
        # sample['target'] = self.data['target'][idx_].astype(np.float32)
        # sample['root_p'] = self.data['root_p'][idx_].astype(np.float32)
        sample['X'] = self.data['X'][idx_].astype(np.float32)
        if self.dataset == "mocap" and self.train == False:
            sample['style'] = self.data['style'][idx_].astype(np.str_)
        
        # sample['local_q_aug'] = self.data['local_q'][idx_].astype(np.float32)
        # sample['root_v_aug'] = self.data['root_v'][idx_].astype(np.float32)
        # sample['contact_aug'] = self.data['contact'][idx_].astype(np.float32)
        # ## data aug ##
        # sample['root_p_offset'] = self.data['root_p_offset'][idx_].astype(np.float32)
        # sample['local_q_offset'] = self.data['local_q_offset'][idx_].astype(np.float32)
        # sample['target'] = self.data['target'][idx_].astype(np.float32)
        # sample['root_p'] = self.data['root_p'][idx_].astype(np.float32)
        # sample['X'] = self.data['X'][idx_].astype(np.float32)
        return sample

if __name__=="__main__":
    lafan_data = LaFan1('D:\\ubisoft-laforge-animation-dataset\\lafan1\\lafan1')
    print(lafan_data.data_X.shape, lafan_data.data_Q.shape)
