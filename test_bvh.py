import torch
import sys
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.insert(0, os.path.dirname(__file__))
from os.path import join as pjoin
from torch.utils.data import Dataset, DataLoader
from utils.LaFan import LaFan1
from utils.skeleton import Skeleton
from utils.interpolate import interpolate_local
from utils.functions import write_to_bvhfile
import numpy as np
import yaml
import random
import imageio
import matplotlib.pyplot as plt
from model import Encoder
from PIL import Image
import utils.benchmarks as ben
import utils.utils_func as uf


def interpolation(X, Q, n_past=10, n_future=10, n_trans=30):
    """
    Evaluate naive baselines (zero-velocity and interpolation) for transition generation on given data.
    :param X: Local positions array of shape     numpy(Batchsize, Timesteps, Joints, 3)
    :param Q: Local quaternions array of shape     numpy(B, T, J, 4)
    :param n_past: Number of frames used as past context
    :param n_future: Number of frames used as future context (only the first frame is used as the target)
    :param n_trans:
    :return:  B, curr_window, xxx
    """
    batchsize = X.shape[0]  # B

    # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
    curr_window = n_trans + n_past + n_future
    curr_x = X[:, :curr_window, ...]  # B, curr_window, J, 3
    curr_q = Q[:, :curr_window, ...]  # B, curr_window, J, 4
    gt_pose = np.concatenate(
        [curr_x.reshape((batchsize, curr_window, -1)), curr_q.reshape((batchsize, curr_window, -1))],
        axis=2)  # [B, curr_window, J*3+J*4]

    # Interpolation pos/quats
    x, q = curr_x, curr_q  # x: B,curr_window,J,3       q: B, curr_window, J, 4
    inter_pos, inter_local_quats = interpolate_local(x, q, n_past,
                                                     n_future)  # inter_pos: B, n_trans + 2, J, 3   inter_local_quats: B, n_trans + 2, J, 4
    # inter_pos = inter_pos.numpy()
    # inter_local_quats = inter_local_quats.numpy()
    trans_inter_pos = inter_pos[:, 1:-1, :, :]  # B, n_trans, J, 3
    inter_local_quats = inter_local_quats[:, 1:-1, :, :]  # B, n_trans, J, 4
    total_interp_positions = np.concatenate([curr_x[:, 0: n_past, ...], trans_inter_pos, curr_x[:, -n_future:, ...]],
                                            axis=1)  # B, curr_window, J, 3
    total_interp_rotations = np.concatenate([curr_q[:, 0: n_past, ...], inter_local_quats, curr_q[:, -n_future:, ...]],
                                            axis=1)  # B, curr_window, J, 4
    interp_pose = np.concatenate([total_interp_positions.reshape((batchsize, curr_window, -1)),
                                  total_interp_rotations.reshape((batchsize, curr_window, -1))],
                                 axis=2)  # [B, curr_window, xxx]
    return gt_pose, interp_pose



def plot_pose(pose, cur_frame, prefix):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    parents = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20]
    ax.cla()
    num_joint = pose.shape[0] // 3
    for i, p in enumerate(parents):
        if i > 0:
            ax.plot([pose[i, 0], pose[p, 0]], \
                    [pose[i, 2], pose[p, 2]], \
                    [pose[i, 1], pose[p, 1]], c='r')
            ax.plot([pose[i + num_joint, 0], pose[p + num_joint, 0]], \
                    [pose[i + num_joint, 2], pose[p + num_joint, 2]], \
                    [pose[i + num_joint, 1], pose[p + num_joint, 1]], c='b')
            ax.plot([pose[i + num_joint * 2, 0], pose[p + num_joint * 2, 0]], \
                    [pose[i + num_joint * 2, 2], pose[p + num_joint * 2, 2]], \
                    [pose[i + num_joint * 2, 1], pose[p + num_joint * 2, 1]], c='g')
    # ax.scatter(pose[:num_joint, 0], pose[:num_joint, 2], pose[:num_joint, 1],c='b')
    # ax.scatter(pose[num_joint:num_joint*2, 0], pose[num_joint:num_joint*2, 2], pose[num_joint:num_joint*2, 1],c='b')
    # ax.scatter(pose[num_joint*2:num_joint*3, 0], pose[num_joint*2:num_joint*3, 2], pose[num_joint*2:num_joint*3, 1],c='g')
    xmin = np.min(pose[:, 0])
    ymin = np.min(pose[:, 2])
    zmin = np.min(pose[:, 1])
    xmax = np.max(pose[:, 0])
    ymax = np.max(pose[:, 2])
    zmax = np.max(pose[:, 1])
    scale = np.max([xmax - xmin, ymax - ymin, zmax - zmin])
    xmid = (xmax + xmin) // 2
    ymid = (ymax + ymin) // 2
    zmid = (zmax + zmin) // 2
    ax.set_xlim(xmid - scale // 2, xmid + scale // 2)
    ax.set_ylim(ymid - scale // 2, ymid + scale // 2)
    ax.set_zlim(zmid - scale // 2, zmid + scale // 2)

    plt.draw()
    plt.savefig(prefix + '_' + str(cur_frame) + '.png', dpi=200, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    opt = yaml.load(open('config/test_config_lafan.yaml', 'r').read(), Loader=yaml.FullLoader)
    model_dir = opt['test']['model_dir']

    random.seed(opt['test']['seed'])
    torch.manual_seed(opt['test']['seed'])
    if opt['test']['cuda']:
        torch.cuda.manual_seed(opt['test']['seed'])

    # ===================================放到GPU==================================
    device = torch.device('cuda' if opt['test']['cuda'] else 'cpu')
    print(device)

    ## initilize the skeleton ##
    skeleton_mocap = Skeleton(offsets=opt['data']['offsets'], parents=opt['data']['parents'])
    skeleton_mocap.to(device)
    if opt['data']['data_set'] == "lafan":
        skeleton_mocap.remove_joints(opt['data']['joints_to_remove'])
    parents = skeleton_mocap.parents()
    ## load test data ##
    lafan_data_test = LaFan1(opt['data']['data_dir'], \
                             opt['data']['data_set'], \
                             seq_len=opt['model']['seq_length'], \
                             offset=opt['data']['offset'], \
                             train=False,
                             debug=opt['test']['debug'])
    x_mean = lafan_data_test.x_mean.to(device)
    x_std = lafan_data_test.x_std.to(device)
    x_mean_n = lafan_data_test.x_mean.view(1, 1, opt['model']['num_joints'], 3).to(device)
    x_std_n = lafan_data_test.x_std.view(1, 1, opt['model']['num_joints'], 3).to(device)
    print("test_positions.shape", lafan_data_test.data['X'].shape)
    print("test_rotations.shape", lafan_data_test.data['Q'].shape)

    lafan_loader_test = DataLoader(lafan_data_test, \
                                   batch_size=opt['test']['batch_size'], \
                                   shuffle=False)

    for batch_i, batch_data in enumerate(lafan_loader_test):
        test = 0

    ## initialize model and load parameters ##

    positions = lafan_data_test.data['X']  # [B, F, J, 3]
    rotations = lafan_data_test.data['Q']  # [B, F, J, 4]
    positions = positions[60:61,:,:,:]
    rotations = rotations[60:61,:,:,:]

    gt_pose, interp_pose = interpolation(positions,
                                         rotations,
                                         n_past=opt['model']['n_past'],
                                         n_future=opt['model']['n_future'],
                                         n_trans=30)
    gt_pose = gt_pose.astype(np.float32)
    interp_pose = interp_pose.astype(np.float32)

    interp_glbl_p = interp_pose[:, :, 0:opt['model']['num_joints'] * 3].reshape(interp_pose.shape[0],
                                                                                interp_pose.shape[1], -1, 3)
    interp_glbl_q = interp_pose[:, :, opt['model']['num_joints'] * 3:].reshape(interp_pose.shape[0],
                                                                               interp_pose.shape[1], -1, 4)

    glbl_q_gt = gt_pose[:, :, opt['model']['num_joints'] * 3:]  # B, F, J*4
    glbl_q_gt_ = glbl_q_gt.reshape(glbl_q_gt.shape[0], glbl_q_gt.shape[1], -1, 4)  # ground truth rotation and position data
    glbl_p_gt = gt_pose[:, :, 0:opt['model']['num_joints'] * 3]  # B, F, J*3
    glbl_p_gt_ = glbl_p_gt.reshape(glbl_p_gt.shape[0], glbl_p_gt.shape[1], -1, 3)  # B, F, J, 3




    output_dir = opt['test']['test_output_dir']
    if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    if opt['data']['data_set'] == "mocap":
        style = batch_data['style']
        pd_output_name = ('test_%d_%s_pd.bvh' % (batch_i, style))
        gt_output_name = ('test_%d_%s_gt.bvh' % (batch_i, style))
    else:
        pd_output_name = ('test_%03d_pd.bvh' % batch_i)
        gt_output_name = ('test_%03d_gt.bvh' % batch_i)
    pd_output_path = pjoin(output_dir, pd_output_name)
    gt_output_path = pjoin(output_dir, gt_output_name)


    if opt['test']['save_bvh']:
        local_q_pred, local_p_pred = uf.quat_ik(interp_glbl_q,interp_glbl_p, parents)
        local_q_gt, local_p_gt = uf.quat_ik(glbl_q_gt_, glbl_p_gt_,parents)

        local_q_pred = torch.from_numpy(local_q_pred).reshape((local_p_pred.shape[0], local_q_pred.shape[1], -1))
        local_p_pred = torch.from_numpy(local_p_pred).reshape((local_p_pred.shape[0], local_p_pred.shape[1], -1))
        local_p_gt = torch.from_numpy(local_p_gt).reshape((local_p_gt.shape[0], local_p_gt.shape[1], -1))
        local_q_gt = torch.from_numpy(local_q_gt).reshape((local_q_gt.shape[0], local_q_gt.shape[1], -1))

        pd_bvh_data = torch.cat(
            [local_p_pred[0].view(local_p_pred.size(1), -1), local_q_pred[0].view(local_q_pred.size(1), -1)],
            -1).detach().cpu().numpy()  # F, J*(3+4)
        gt_bvh_data = torch.cat(
            [local_p_gt[0].view(local_p_gt.size(1), -1), local_q_gt[0].view(local_q_gt.size(1), -1)],
            -1).detach().cpu().numpy()  # F, J*(3+4)
                # print('bvh_data:',pd_bvh_data.shape)
                # print('bvh_data:',gt_bvh_data.shape)
        # write_to_bvhfile(pd_bvh_data, pd_output_path, opt['data']['data_set'])
        write_to_bvhfile(pd_bvh_data, pd_output_path, opt['data']['data_set'])
        write_to_bvhfile(gt_bvh_data, gt_output_path, opt['data']['data_set'])



