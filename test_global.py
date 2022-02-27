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


def interpolation(X, Q, n_past=10, n_future=10, n_trans = 30):
    """
    Evaluate naive baselines (zero-velocity and interpolation) for transition generation on given data.
    :param X: Local positions array of shape     numpy(Batchsize, Timesteps, Joints, 3)
    :param Q: Local quaternions array of shape     numpy(B, T, J, 4)
    :param n_past: Number of frames used as past context
    :param n_future: Number of frames used as future context (only the first frame is used as the target)
    :param n_trans:
    :return:  B, curr_window, xxx
    """
    batchsize = X.shape[0]  #  B

    # Format the data for the current transition lengths. The number of samples and the offset stays unchanged.
    curr_window = n_trans + n_past + n_future
    curr_x = X[:, :curr_window, ...]    # B, curr_window, J, 3
    curr_q = Q[:, :curr_window, ...]    # B, curr_window, J, 4
    gt_pose = np.concatenate([X.reshape((batchsize, X.shape[1], -1)), Q.reshape((batchsize, Q.shape[1], -1))], axis=2)  # [B, curr_window, J*3+J*4]

    # Interpolation pos/quats
    x, q = curr_x, curr_q    # x: B,curr_window,J,3       q: B, curr_window, J, 4
    inter_pos, inter_local_quats = interpolate_local(x.numpy(), q.numpy(), n_past, n_future)  # inter_pos: B, n_trans + 2, J, 3   inter_local_quats: B, n_trans + 2, J, 4

    trans_inter_pos = inter_pos[:, 1:-1, :, :]    #  B, n_trans, J, 3  把头尾2帧去掉
    inter_local_quats = inter_local_quats[:, 1:-1, :, :]  #  B, n_trans, J, 4
    total_interp_positions = np.concatenate([X[:, 0: n_past, ...], trans_inter_pos, X[:, n_past+n_trans:: , ...]], axis = 1)    # B, curr_window, J, 3  重新拼接
    total_interp_rotations = np.concatenate([Q[:, 0: n_past, ...], inter_local_quats, Q[:, n_past+n_trans: , ...]], axis = 1)  # B, curr_window, J, 4
    interp_pose = np.concatenate([total_interp_positions.reshape((batchsize, X.shape[1], -1)), total_interp_rotations.reshape((batchsize, Q.shape[1], -1))], axis=2)  # [B, curr_window, xxx]
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

    trans_lengths = [5, 15, 30]
    n_joints = 22
    res = {}

    for n_trans in trans_lengths:
        L2Q_zero_list = []
        L2P_zero_list = []
        NPSS_zero_list = []
        L2Q_interp_list = []
        L2P_interp_list = []
        NPSS_interp_list = []
        L2Q_our_list = []
        L2P_our_list = []
        NPSS_our_list = []

        ## initialize model and load parameters ##
        model = Encoder(device=device,
                        seq_len=opt['model']['seq_length'],
                        input_dim=opt['model']['input_dim'],
                        n_layers=opt['model']['n_layers'],
                        n_head=opt['model']['n_head'],
                        d_k=opt['model']['d_k'],
                        d_v=opt['model']['d_v'],
                        d_model=opt['model']['d_model'],
                        d_inner=opt['model']['d_inner'],
                        dropout=opt['test']['dropout'],
                        n_past=opt['model']['n_past'],
                        n_future=opt['model']['n_future'],
                        n_trans=n_trans)

        checkpoint = torch.load(model_dir)
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model loaded.')

        model = model.to(device)
        model.eval()
        print('Computing errors for transition length = {}...'.format(n_trans))

        for batch_i, batch_data in enumerate(lafan_loader_test):
            loss_pose = 0
            loss_quat = 0
            loss_position = 0

            pred_img_list = []
            gt_img_list = []
            with torch.no_grad():
                positions = batch_data['X']  # [B, F, J, 3]
                rotations = batch_data['Q']  # [B, F, J, 4]

                # gt_pose numpy [B, F, dim] interp_pose numpy[B, F, dim]
                gt_pose, interp_pose = interpolation(positions,
                                                     rotations,
                                                     n_past=opt['model']['n_past'],
                                                     n_future=opt['model']['n_future'],
                                                     n_trans=n_trans)
                gt_pose = gt_pose.astype(np.float32)
                interp_pose = interp_pose.astype(np.float32)
                input = torch.from_numpy(interp_pose).to(device)
                target_output = torch.from_numpy(gt_pose).to(device)





                output = model(input)
                # ------------------------global data-----------------------------------
                # interp data processing---------------------
                interp_glbl_p = interp_pose[:, :, 0:opt['model']['num_joints'] * 3].reshape(interp_pose.shape[0],interp_pose.shape[1], -1, 3)
                interp_glbl_q = interp_pose[:, :, opt['model']['num_joints'] * 3:].reshape(interp_pose.shape[0],interp_pose.shape[1], -1, 4)

                trans_interp_glbl_quats = interp_glbl_q[:, opt['model']['n_past']:opt['model']['n_past']+n_trans, :, :]
                trans_interp_glbl_poses = interp_glbl_p[:, opt['model']['n_past']:opt['model']['n_past']+n_trans, :, :]
                # # # Normalize
                trans_interp_glbl_poses = (torch.from_numpy(trans_interp_glbl_poses) - x_mean_n.cpu()) / x_std_n.cpu()



                # prediction and gt data processing------------------
                # rotations
                glbl_q_gt = target_output[:, :, opt['model']['num_joints'] * 3:]  # B, F, J*4
                glbl_q_gt_ = glbl_q_gt.view(glbl_q_gt.size(0), glbl_q_gt.size(1), -1, 4)  # ground truth rotation and position data
                glbl_q_pred = output[:, :, opt['model']['num_joints'] * 3:]  # B, F, J*4            局部四元数
                glbl_q_pred_ = glbl_q_pred.view(glbl_q_pred.shape[0], glbl_q_pred.shape[1], -1, 4)  # pred r and p data

                # positions
                glbl_p_pred = output[:, :, 0:opt['model']['num_joints'] * 3]  # B, F, J*3
                glbl_p_pred = glbl_p_pred.view(glbl_p_pred.shape[0], glbl_p_pred.shape[1], -1, 3)
                glbl_p_pred_ = glbl_p_pred / torch.norm(glbl_p_pred, dim=-1, keepdim=True)
                glbl_p_gt = target_output[:, :, 0:opt['model']['num_joints'] * 3]  # B, F, J*3
                glbl_p_gt = glbl_p_gt.view(glbl_p_gt.size(0), glbl_p_gt.size(1), -1, 3)  # B, F, J, 3
                # glbl_p_gt_ = glbl_p_gt / torch.norm(glbl_p_gt, dim=-1, keepdim=True)


                root_pred = glbl_p_pred[:, :, 0:3]  # B, F, 3   根节点预测值
                root_gt = glbl_p_gt[:, :, 0:3]  # B, F, 3

                # error computing---------------------------------------------

                # global info in prediction area
                trans_glbl_p_pred = (glbl_p_pred[:, opt['model']['n_past']: opt['model']['n_past']+n_trans,
                                     ...] - x_mean_n) / x_std_n  # Normalization
                trans_glbl_q_pred = glbl_q_pred_[:, opt['model']['n_past']:opt['model']['n_past']+n_trans,
                                    ...]  # 过渡区间的局部旋转真实值  B, n_trans, J, 4

                trans_glbl_p_gt = (glbl_p_gt[:, opt['model']['n_past']: opt['model']['n_past']+n_trans,
                                   ...] - x_mean_n) / x_std_n
                trans_glbl_q_gt = glbl_q_gt_[:, opt['model']['n_past']: opt['model']['n_past']+n_trans, ...]

                # zero-v data processing-----------------------------
                zerov_trans_glbl_quats, zerov_trans_glbl_poses = np.zeros_like(trans_glbl_q_gt.detach().cpu().numpy()), np.zeros_like(
                    trans_glbl_p_gt.detach().cpu().numpy())
                zerov_trans_glbl_quats[:, :, :, :] = glbl_q_gt_[:, opt['model']['n_past'] - 1:opt['model']['n_past'], :, :].detach().cpu().numpy()
                trans_zerov_glbl_quats = zerov_trans_glbl_quats
                zerov_trans_glbl_poses[:, :, :, :] = glbl_p_gt[:,opt['model']['n_past'] - 1:opt['model']['n_past'], :, :].detach().cpu().numpy()
                # # Normalize
                trans_zerov_glbl_poses = (torch.from_numpy(zerov_trans_glbl_poses) - x_mean_n.cpu()) / x_std_n.cpu()


                # 评估指标： L2Q、L2P和NPSS
                # global quaternion loss L2Q
                # print(f"trans_local_q_pred:{trans_local_q_pred.shape}trans_local_q_gt: {trans_local_q_gt.shape}")
                l2q_error_zerov = np.mean(
                    np.sqrt(np.sum((trans_zerov_glbl_quats - trans_glbl_q_gt.detach().cpu().numpy()) ** 2.0,axis=(2, 3))))
                l2q_error_interp = np.mean(
                    np.sqrt(np.sum((trans_interp_glbl_quats - trans_glbl_q_gt.detach().cpu().numpy()) ** 2.0, axis=(2, 3))))
                l2q_error_our = np.mean(
                    np.sqrt(np.sum((trans_glbl_q_pred.detach().cpu().numpy() - trans_glbl_q_gt.detach().cpu().numpy()) ** 2.0,axis=(2, 3))))
                # 插值的全局旋转四元数 - 真实的全局旋转四元数 二范数 最后求均值
                # Global positions loss L2P
                l2p_error_zerov = np.mean(np.sqrt(
                    np.sum((trans_zerov_glbl_poses.numpy() - trans_glbl_p_gt.detach().cpu().numpy()) ** 2.0,axis=(2, 3))))
                l2p_error_interp = np.mean(np.sqrt(
                    np.sum((trans_interp_glbl_poses.numpy() - trans_glbl_p_gt.detach().cpu().numpy()) ** 2.0,axis=(2, 3))))
                l2p_error_our = np.mean(np.sqrt(
                    np.sum((trans_glbl_p_pred.detach().cpu().numpy() - trans_glbl_p_gt.detach().cpu().numpy()) ** 2.0, axis=(2,3))))

                # NPSS loss on global quaternions
                npss_error_zerov = ben.fast_npss(ben.flatjoints(trans_glbl_q_gt.detach().cpu().numpy()),
                                               ben.flatjoints(trans_zerov_glbl_quats))

                npss_error_interp = ben.fast_npss(ben.flatjoints(trans_glbl_q_gt.detach().cpu().numpy()),
                                               ben.flatjoints(trans_interp_glbl_quats))

                npss_error_our = ben.fast_npss(ben.flatjoints(trans_glbl_q_gt.detach().cpu().numpy()),
                                             ben.flatjoints(trans_glbl_q_pred.detach().cpu().numpy()))


                L2Q_our_list.append(l2q_error_our.item())
                L2P_our_list.append(l2p_error_our.item())
                NPSS_our_list.append(npss_error_our.item())

                L2Q_interp_list.append(l2q_error_interp.item())
                L2P_interp_list.append(l2p_error_interp.item())
                NPSS_interp_list.append(npss_error_interp.item())

                L2Q_zero_list.append(l2q_error_zerov.item())
                L2P_zero_list.append(l2p_error_zerov.item())
                NPSS_zero_list.append(npss_error_zerov.item())

                # output_dir = opt['test']['test_output_dir']
                if n_trans == 5:
                    output_dir = opt[ 'test']['output_dir1']
                if n_trans == 15:
                    output_dir = opt['test']['output_dir2']
                if n_trans == 30:
                    output_dir = opt['test']['output_dir3']
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

                # interp_output_name = ('test_%03d_interp.bvh'% batch_i)
                # interp_output_path = pjoin(output_dir, interp_output_name)

                if opt['test']['save_bvh']:

                    local_q_pred, local_p_pred = uf.quat_ik(glbl_q_pred_.detach().cpu().numpy(), glbl_p_pred.detach().cpu().numpy(), parents)
                    local_q_gt, local_p_gt = uf.quat_ik(glbl_q_gt_.detach().cpu().numpy(), glbl_p_gt.detach().cpu().numpy(), parents)
                    local_q_pred = torch.from_numpy(local_q_pred).reshape((local_p_pred.shape[0],local_q_pred.shape[1],-1))
                    local_p_pred = torch.from_numpy(local_p_pred).reshape((local_p_pred.shape[0], local_p_pred.shape[1], -1))
                    local_p_gt = torch.from_numpy(local_p_gt).reshape((local_p_gt.shape[0], local_p_gt.shape[1], -1))
                    local_q_gt = torch.from_numpy(local_q_gt).reshape((local_q_gt.shape[0], local_q_gt.shape[1], -1))

                    pd_bvh_data = torch.cat([local_p_pred[0].view(local_p_pred.size(1), -1), local_q_pred[0].view(local_q_pred.size(1), -1)],
                                            -1).detach().cpu().numpy()  # F, J*(3+4)
                    gt_bvh_data = torch.cat([local_p_gt[0].view(local_p_gt.size(1), -1), local_q_gt[0].view(local_q_gt.size(1), -1)],
                                            -1).detach().cpu().numpy()  # F, J*(3+4)
                    # print('bvh_data:',pd_bvh_data.shape)
                    # print('bvh_data:',gt_bvh_data.shape)
                    write_to_bvhfile(pd_bvh_data, pd_output_path, opt['data']['data_set'])
                    write_to_bvhfile(gt_bvh_data, gt_output_path, opt['data']['data_set'])
                    print(f"{n_trans} frames completion files have been written in outputdir ")

                # save_img & save_gif 还有问题
                if opt['test']['save_img'] and opt['test']['save_gif'] and batch_i < 50:
                    gif_dir = opt['test']['gif_dir']
                    if not os.path.exists(gif_dir):
                        os.mkdir(gif_dir)
                    img_dir = opt['test']['img_dir']
                    if not os.path.exists(img_dir):
                        os.mkdir(img_dir)
                    num_joints = opt['model']['num_joints']
                    position_0 = glbl_p_gt[0, 0, ...].detach().to('cpu').numpy()
                    position_1 = glbl_p_gt[0, -1, ...].detach().to('cpu').numpy()
                    for t in range(opt['model']['seq_length']):
                        # print(type(position_0))
                        # print(position_0.device)
                        plot_pose(
                            np.concatenate([position_0, glbl_p_pred_[0, t].detach().to('cpu').numpy(), position_1], axis=0),t,
                            img_dir + '/pred_batch_' + str(batch_i))
                        plot_pose(np.concatenate([position_0,  # .detach().cpu().numpy(), \
                                                  glbl_p_gt[0, t].detach().to('cpu').numpy(),
                                                  position_1], 0), \
                                  t,
                                  img_dir + '/gt_batch_' + str(batch_i))
                        pred_img = Image.open(img_dir + '/pred_batch_' + str(batch_i) + '_' + str(t) + '.png', 'r')
                        gt_img = Image.open(img_dir + '/gt_batch_' + str(batch_i) + '_' + str(t) + '.png', 'r')
                        pred_img_list.append(pred_img)
                        gt_img_list.append(gt_img)
                    imageio.mimsave((gif_dir + '/pred_img_%03d.gif' % batch_i), pred_img_list, duration=0.1)
                    imageio.mimsave((gif_dir + '/gt_img_%03d.gif' % batch_i), gt_img_list, duration=0.1)
        res[('our_L2Q_error', n_trans)] = np.mean(L2Q_our_list)
        res[('our_L2P_error', n_trans)] = np.mean(L2P_our_list)
        res[('our_NPSS_error', n_trans)] = np.mean(NPSS_our_list)

        res[('interp_L2Q_error', n_trans)] = np.mean(L2Q_interp_list)
        res[('interp_L2P_error', n_trans)] = np.mean(L2P_interp_list)
        res[('interp_NPSS_error', n_trans)] = np.mean(NPSS_interp_list)

        res[('zerov_L2Q_error', n_trans)] = np.mean(L2Q_zero_list)
        res[('zerov_L2P_error', n_trans)] = np.mean(L2P_zero_list)
        res[('zerov_NPSS_error', n_trans)] = np.mean(NPSS_zero_list)

    # print("batch:%5d, avg test loss:%.6f"% (batch_i, np.mean(loss_total_list)))
    # print("batch:%5d, avg test npss:%.6f"% (batch_i, np.mean(npss_total_list)))

    print()
    avg_our_L2Q_error = [res[('our_L2Q_error', n)] for n in trans_lengths]
    avg_our_L2P_error = [res[('our_L2P_error', n)] for n in trans_lengths]
    avg_our_NPSS_error = [res[('our_NPSS_error', n)] for n in trans_lengths]

    avg_interp_L2Q_error = [res[('interp_L2Q_error', n)] for n in trans_lengths]
    avg_interp_L2P_error = [res[('interp_L2P_error', n)] for n in trans_lengths]
    avg_interp_NPSS_error = [res[('interp_NPSS_error', n)] for n in trans_lengths]

    avg_zerov_L2Q_error = [res[('zerov_L2Q_error', n)] for n in trans_lengths]
    avg_zerov_L2P_error = [res[('zerov_L2P_error', n)] for n in trans_lengths]
    avg_zerov_NPSS_error = [res[('zerov_NPSS_error', n)] for n in trans_lengths]

    print("=== L2Q losses ===")
    print("{0: <16} | {1:6d} | {2:6d} | {3:6d}".format("Lengths", 5, 15, 30))
    print("{0: <16} | {1:6.4f} | {2:6.4f} | {3:6.4f}".format("zerov", *avg_zerov_L2Q_error))
    print("{0: <16} | {1:6.4f} | {2:6.4f} | {3:6.4f}".format("interp", *avg_interp_L2Q_error))
    print("{0: <16} | {1:6.4f} | {2:6.4f} | {3:6.4f}".format("ours", *avg_our_L2Q_error))
    print()
    print("=== L2P losses ===")
    print("{0: <16} | {1:6d} | {2:6d} | {3:6d}".format("Lengths", 5, 15, 30))
    print("{0: <16} | {1:6.4f} | {2:6.4f} | {3:6.4f}".format("zerov", *avg_zerov_L2P_error))
    print("{0: <16} | {1:6.4f} | {2:6.4f} | {3:6.4f}".format("interp", *avg_interp_L2P_error))
    print("{0: <16} | {1:6.4f} | {2:6.4f} | {3:6.4f}".format("ours", *avg_our_L2P_error))
    print()
    print("=== NPSS losses ===")
    print("{0: <16} | {1:5d} | {2:5d} | {3:5d}".format("Lengths", 5, 15, 30))
    print("{0: <16} | {1:5.4f} | {2:5.4f} | {3:5.4f}".format("zerov", *avg_zerov_NPSS_error))
    print("{0: <16} | {1:5.4f} | {2:5.4f} | {3:5.4f}".format("interp", *avg_interp_NPSS_error))
    print("{0: <16} | {1:5.4f} | {2:5.4f} | {3:5.4f}".format("ours", *avg_our_NPSS_error))
    print()