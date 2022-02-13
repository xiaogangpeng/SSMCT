import torch
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.insert(0, os.path.dirname(__file__))
from utils.LaFan import LaFan1
from torch.utils.data import Dataset, DataLoader
from utils.skeleton import Skeleton
from utils.interpolate import interpolate_local
import torch.optim as optim
import numpy as np
import yaml
import time
import random
from model import Encoder
from visdom import Visdom
import utils.benchmarks as bench
import utils.utils_func as utilsf
from tqdm import tqdm

def flatjoints(x):
    """
    Shorthand for a common reshape pattern. Collapses all but the two first dimensions of a tensor.
    :param x: Data tensor of at least 3 dimensions.
    :return: The flattened tensor.
    """
    return x.reshape((x.shape[0], x.shape[1], -1))

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
    gt_pose = np.concatenate([curr_x.reshape((batchsize, curr_window, -1)), curr_q.reshape((batchsize, curr_window, -1))], axis=2)  # [B, curr_window, J*3+J*4]

    # Interpolation pos/quats
    x, q = curr_x, curr_q    # x: B,curr_window,J,3       q: B, curr_window, J, 4
    inter_pos, inter_local_quats = interpolate_local(x, q, n_past, n_future)  # inter_pos: B, n_trans + 2, J, 3   inter_local_quats: B, n_trans + 2, J, 4
    inter_pos = inter_pos.numpy()
    inter_local_quats = inter_local_quats.numpy()
    trans_inter_pos = inter_pos[:, 1:-1, :, :]    #  B, n_trans, J, 3  把头尾2帧去掉
    inter_local_quats = inter_local_quats[:, 1:-1, :, :]  #  B, n_trans, J, 4
    total_interp_positions = np.concatenate([curr_x[:, 0: n_past, ...], trans_inter_pos, curr_x[:, -n_future: , ...]], axis = 1)    # B, curr_window, J, 3  重新拼接
    total_interp_rotations = np.concatenate([curr_q[:, 0: n_past, ...], inter_local_quats, curr_q[:, -n_future: , ...]], axis = 1)  # B, curr_window, J, 4
    interp_pose = np.concatenate([total_interp_positions.reshape((batchsize, curr_window, -1)), total_interp_rotations.reshape((batchsize, curr_window, -1))], axis=2)  # [B, curr_window, xxx]
    return gt_pose, interp_pose


if __name__ == '__main__':
    # ===========================================读取配置信息===============================================
    opt = yaml.load(open('./config/train_config_lafan.yaml', 'r').read(), Loader=yaml.FullLoader)      # 用mocap_bfa, mocap_xia数据集训练
    # opt = yaml.load(open('./config/train_config_lafan.yaml', 'r').read())     # 用lafan数据集训练
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    print(stamp)
    # assert 0

    output_dir = opt['train']['output_dir']     # 模型输出路径
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    random.seed(opt['train']['seed'])
    torch.manual_seed(opt['train']['seed'])
    if opt['train']['cuda']:
        torch.cuda.manual_seed(opt['train']['seed'])

    # ===================================使用GPU==================================
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    print(device)


    #==========================初始化Skel和数据========================================
    ## initilize the skeleton ##

    skeleton_mocap = Skeleton(offsets=opt['data']['offsets'], parents=opt['data']['parents'])
    # print(skeleton_mocap.offsets())
    skeleton_mocap.to(device)
    if opt['data']['data_set'] == "lafan":
        skeleton_mocap.remove_joints(opt['data']['joints_to_remove'])

    ## load train data ##
    lafan_data_train = LaFan1(opt['data']['data_dir'],
                              opt['data']['data_set'],
                              seq_len = opt['model']['seq_length'],
                              offset = opt['data']['offset'],
                              train = True,
                              debug=opt['train']['debug'])
    x_mean = lafan_data_train.x_mean.to(device)
    x_std = lafan_data_train.x_std.to(device).view(1, 1, opt['model']['num_joints'], 3)
    print("train_positions.shape", lafan_data_train.data['X'].shape)
    print("train_rotations.shape", lafan_data_train.data['local_q'].shape)

    lafan_loader_train = DataLoader(lafan_data_train,
                                    batch_size=opt['train']['batch_size'],
                                    shuffle=True,
                                    num_workers=opt['data']['num_workers'])

    #===============================初始化模型=======================================
    ## initialize model ##
    model = Encoder(device = device,
                    seq_len=opt['model']['seq_length'],
                    input_dim=opt['model']['input_dim'],
                    n_layers=opt['model']['n_layers'],
                    n_head=opt['model']['n_head'],
                    d_k=opt['model']['d_k'],
                    d_v=opt['model']['d_v'],
                    d_model=opt['model']['d_model'],
                    d_inner=opt['model']['d_inner'],
                    dropout=opt['train']['dropout'],
                    n_past=opt['model']['n_past'],
                    n_future=opt['model']['n_future'],
                    n_trans=opt['model']['n_trans'])
    print(model)
    model.to(device)
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=opt['train']['lr'])

    #============================================= train ===================================================
    viz = Visdom()
    x_cor = 0
    viz.line([0.], [x_cor], win=opt['train']['picture_name'], opts=dict(title=opt['train']['picture_name']))
    loss_total_min = 10000000.0

    curr_window = opt['model']['n_past'] + opt['model']['n_trans'] + opt['model']['n_future']
    print(f"curr_window: {curr_window}")
    for epoch_i in range(1, opt['train']['num_epoch']+1):  # 每个epoch轮完一遍所有的训练数据
        model.train()
        loss_total_list = []
        L2Q_error_list = []
        L2P_error_list = []
        NPSS_error_list = []

        # 每个batch训练一批数据
        for batch_i, batch_data in tqdm(enumerate(lafan_loader_train)):  # mini_batch
            loss_pose = 0
            loss_quat = 0
            loss_position = 0
            positions = batch_data['X'] # B, F, J, 3
            rotations = batch_data['local_q']
            # 插值 求ground truth 和 插值结果
            # gt_pose numpy [B, F, dim] interp_pose numpy[B, F, dim]
            gt_pose, interp_pose = interpolation(positions,
                                                 rotations,
                                                 n_past = opt['model']['n_past'],
                                                 n_future = opt['model']['n_future'],
                                                 n_trans = opt['model']['n_trans'])

            # 数据放到GPU to_device
            gt_pose = gt_pose.astype(np.float32)
            interp_pose = interp_pose.astype(np.float32)
            input = torch.from_numpy(interp_pose).to(device)
            target_output = torch.from_numpy(gt_pose).to(device)

            # print("gt_pose.shape",gt_pose.shape)
            # print("interp_pose.shape",interp_pose.shape)
            # print("gt_pose.type", type(gt_pose))            # class 'numpy.ndarray'
            # print("input.type", input.dtype)    # class 'numpy.ndarray'

            # 训练
            optimizer.zero_grad()

            output = model(input)

            local_q_pred = output[:,:, opt['model']['num_joints']*3:]       # B, F, J*4
            local_q_gt = target_output[:,:, opt['model']['num_joints']*3:]  # B, F, J*4
            glbl_p_pred = output[:,:, 0:opt['model']['num_joints']*3]       # B, F, J*3
            glbl_p_gt = target_output[:,:, 0:opt['model']['num_joints']*3]  # B, F, J*3
            root_pred = glbl_p_pred[:,:,0:3]         # B, F, 3   根节点预测值

            # root_gt = glbl_p_gt[:,:, 0:3]           # B, F, 3

            local_q_pred_ = local_q_pred.view(local_q_pred.size(0), local_q_pred.size(1), -1, 4)
            local_q_pred_ = local_q_pred_ / torch.norm(local_q_pred_, dim=-1, keepdim=True)
            pos_pred = skeleton_mocap.forward_kinematics(local_q_pred_, root_pred)
            glbl_p_gt_ = glbl_p_gt.view(glbl_p_gt.size(0), glbl_p_gt.size(1), -1, 3)  # B, F, J, 3

            loss_pose += torch.mean(torch.abs(pos_pred - glbl_p_gt_) / x_std)   # 运动学损失
            loss_quat += torch.mean(torch.abs(local_q_pred - local_q_gt))       # 旋转四元数损失
            loss_position += torch.mean(torch.abs(glbl_p_pred - glbl_p_gt))     # 位移损失

            # 计算损失函数
            loss_total = opt['train']['loss_quat_weight'] * loss_quat + \
                         opt['train']['loss_position_weight'] * loss_position  + \
                         opt['train']['loss_pose_weight'] * loss_pose
            loss_total.backward()

            # update parameters
            optimizer.step()
            loss_total_list.append(loss_total.item())

            # error computing---------------------------------------------
            gt_local_quats = rotations  # 局部旋转四元数真实值 B, curr_window, J, 4
            gt_roots = positions[:, :, 0:1, :]  # 根关节位移真实值 B, curr_window, 1, 3
            offsets = skeleton_mocap.offsets().detach().cpu().numpy()
            # print(f"offsets:{offsets.shape}")
            parents = skeleton_mocap.parents()
            gt_offsets = np.tile(offsets, [gt_local_quats.size(0), curr_window, 1, 1])  # 子关节的偏移量真实值 B, curr_window, J, 3
            # print(f"gt_offsets:{gt_offsets.shape}")
            # print(f"gt_offsets:{gt_offsets.shape}")
            # gt_local_poses = np.concatenate([gt_roots, gt_offsets], axis=2)  # 所有关节的局部位置真实值 B, curr_window, J, 3
            # print(f"!!!!!!  {np.tile(gt_roots, [1, 1, opt['model']['num_joints'], 1]).shape}")
            # gt_local_poses = np.tile(gt_roots, [1, 1, opt['model']['num_joints'], 1]) + gt_offsets             #保持尺寸一致
            gt_local_poses = gt_roots + gt_offsets
            # print(f"gt_roots:{np.tile(gt_roots, [1, 1, opt['model']['num_joints'], 1]).shape},--gt_offsets  :{gt_offsets.shape}")
            trans_gt_local_poses = gt_local_poses[:, opt['model']['n_past']: -opt['model']['n_future'], ...] # 过渡区间的局部位置真实值  B, n_trans, J, 3
            trans_gt_local_quats = gt_local_quats[:, opt['model']['n_past']: -opt['model']['n_future'], ...]  # 过渡区间的局部旋转真实值  B, n_trans, J, 4
            # print(
            #     f"trans_gt_local_poses: {type(trans_gt_local_poses)} trans_gt_local_quats:{type(trans_gt_local_quats)}")
            # Local to global with Forward Kinematics (FK) 局部位置和旋转转换为全局位置和旋转
            trans_gt_global_quats, trans_gt_global_poses = utilsf.quat_fk(trans_gt_local_quats, trans_gt_local_poses, parents)
            # print(
            #      f"trans_gt_global_poses: {type(trans_gt_global_poses)} trans_gt_global_quats:{trans_gt_global_quats.shape}")
            # print(f"trans_gt_global_quats:{trans_gt_global_quats.shape}")
            # trans_gt_global_poses = trans_gt_global_poses.reshape((trans_gt_global_poses.shape[0], -1, 22 * 3)).transpose([0, 2, 1])
            # print(f"x_std:{x_std.reshape(-1, -1, opt['model']['num_joints']*3).transpose([0, 2, 1]).shape},--xmean:{x_mean.shape}")
            # Normalize 全局位置 正则化处理
            # trans_gt_global_poses = torch.from_numpy(trans_gt_global_poses).to(device)
            # print(f"trans_gt_global_poses:{trans_gt_global_poses.shape}")
            # trans_gt_global_poses = (trans_gt_global_poses - x_mean) / x_std.reshape(-1, -1, opt['model']['num_joints']*3).transpose([0, 2, 1])

            # 预测值！！！！！-------------------------------------------------------------------------
            trans_root_pred = root_pred[:,opt['model']['n_past']: -opt['model']['n_future'],:]
            pred_offsets = np.tile(offsets, [gt_local_quats.size(0), opt['model']['n_trans'], 1, 1])
            # print(f"trans_root_pred:{ trans_root_pred.view(trans_root_pred.size(0), trans_root_pred.size(1), -1, 3).shape} + pred_offsets:{pred_offsets.shape}")
            trans_pred_local_poses = trans_root_pred.view(trans_root_pred.size(0), trans_root_pred.size(1), -1, 3).detach().cpu() + torch.from_numpy(pred_offsets).detach().cpu() # 所有关节的局部位置真实值 B, curr_window, J, 3
            # trans_pred_global_poses = pos_pred[:, opt['model']['n_past']: -opt['model']['n_future'],...]  # 过渡区间的局部位置真实值  B, n_trans, J, 3
            trans_pred_local_quats = local_q_pred_[:, opt['model']['n_past']: -opt['model']['n_future'],...]  # 过渡区间的局部旋转真实值  B, n_trans, J, 4
            #  这里把两个tensor 转成numpy 再喂入quat_fk
            trans_interp_global_quats, trans_interp_global_poses = utilsf.quat_fk(trans_pred_local_quats.detach().cpu().numpy(),
                                                                                 trans_pred_local_poses.detach().cpu().numpy(), parents)
            # trans_pred_global_poses = trans_pred_global_poses.reshape((trans_pred_global_poses.shape[0], -1, 22 * 3)).transpose([0, 2, 1])
            # Normalize
            # trans_pred_global_poses = torch.from_numpy(trans_pred_global_poses).to(device)
            # trans_pred_global_poses = (trans_pred_global_poses - x_mean) / x_std.reshape(-1, -1, opt['model']['num_joints']*3).transpose([0, 2, 1])
            npss_error = bench.fast_npss(flatjoints(trans_gt_global_quats), flatjoints(trans_interp_global_quats))
            NPSS_error_list.append(npss_error.item())

        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch_i
        }

        loss_total_cur = np.mean(loss_total_list)
        npss_erroe_cur = np.mean(NPSS_error_list)
        if loss_total_cur < loss_total_min:
            loss_total_min = loss_total_cur
        print('[train epoch: %5d] cur total loss: %.6f, cur best loss:%.6f, cur NPSS: %.4f' % (epoch_i, loss_total_cur, loss_total_min, npss_erroe_cur))
        viz.line([loss_total_cur], [x_cor], win=opt['train']['picture_name'], update='append')
        x_cor += 10

        if epoch_i % opt['train']['save_per_epochs'] == 0 or epoch_i == 1:
            filename = os.path.join(opt['train']['output_dir'], f'epoch_{epoch_i}.pt')
            torch.save(checkpoint, filename)


