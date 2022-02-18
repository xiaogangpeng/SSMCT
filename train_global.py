import torch
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
sys.path.insert(0, os.path.dirname(__file__))
from utils.LaFan import LaFan1
from torch.utils.data import Dataset, DataLoader
from utils.skeleton import Skeleton
from utils.interpolate import interpolate_local
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from modules.warmup import GradualWarmupScheduler
import torch.optim as optim
import numpy as np
import yaml
import time
import random
from model import Encoder
from visdom import Visdom
import utils.benchmarks as bench
import utils.utils_func as uf
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
    parents = skeleton_mocap.parents()
    ## load train data ##
    lafan_data_train = LaFan1(opt['data']['data_dir'],
                              opt['data']['data_set'],
                              seq_len = opt['model']['seq_length'],
                              offset = opt['data']['offset'],
                              train = True,
                              debug=opt['train']['debug'])
    x_mean = lafan_data_train.x_mean.to(device)
    x_std = lafan_data_train.x_std.to(device)
    x_mean_n = lafan_data_train.x_mean.view(1, 1, opt['model']['num_joints'], 3).to(device)
    x_std_n = lafan_data_train.x_std.view(1, 1, opt['model']['num_joints'], 3).to(device)
    # print("train_positions.shape", lafan_data_train.data['X'].shape)
    # print("train_rotations.shape", lafan_data_train.data['local_q'].shape)

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
    optimizer = optim.Adam(filter(lambda x: x.requires_grad,  model.parameters()),
                           lr=opt['train']['lr'],)
    scheduler_steplr = StepLR(optimizer, step_size=200, gamma=opt['train']['weight_decay'])
    scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=50, after_scheduler=scheduler_steplr)

    #============================================= train ===================================================
    viz = Visdom()
    x_cor = 0
    viz.line([0.], [x_cor], win=opt['train']['picture_name'], opts=dict(title=opt['train']['picture_name']))
    # viz.line([0.], [x_cor], win='loss_fk_cur', opts=dict(title='loss_fk_cur'))
    viz.line([0.], [x_cor], win='loss_root_cur', opts=dict(title='loss_root_cur'))
    viz.line([0.], [x_cor], win='loss_pos_cur', opts=dict(title='loss_pos_cur'))
    viz.line([0.], [x_cor], win='loss_quat_cur', opts=dict(title='loss_quat_cur'))
    viz.line([0.], [x_cor], win='L2Q', opts=dict(title='L2Q'))
    viz.line([0.], [x_cor], win='L2P', opts=dict(title='L2P'))
    viz.line([0.], [x_cor], win='NPSS', opts=dict(title='NPSS'))
    loss_total_min = 10000000.0
    npss_total_min = 10000000.0
    l2q_total_min = 10000000.0
    l2p_total_min = 10000000.0


    curr_window = opt['model']['n_past'] + opt['model']['n_trans'] + opt['model']['n_future']
    print(f"curr_window: {curr_window}")
    for epoch_i in range(1, opt['train']['num_epoch']+1):  # 每个epoch轮完一遍所有的训练数据
        model.train()
        loss_total_list = []
        loss_root_list = []
        loss_fk_list = []
        loss_quat_list = []
        loss_pos_list = []
        L2Q_error_list = []
        L2P_error_list = []
        NPSS_error_list = []
        scheduler_warmup.step(epoch_i)
        print("epoch: ",epoch_i, "lr: {:.10f} ".format(optimizer.param_groups[0]['lr']))

        # 每个batch训练一批数据
        for batch_i, batch_data in tqdm(enumerate(lafan_loader_train)):  # mini_batch
            loss_ik = 0
            loss_quat = 0
            loss_position = 0
            loss_root = 0
            positions = batch_data['X'] # B, F, J, 3
            rotations = batch_data['Q']
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

            # Training
            output = model(input)
            optimizer.zero_grad()

            # Results output
            glbl_q_pred = output[:, :, opt['model']['num_joints']*3:]       # B, F, J*4            局部四元数
            glbl_q_gt = target_output[:, :, opt['model']['num_joints']*3:]  # B, F, J*4
            glbl_p_pred = output[:, :, 0:opt['model']['num_joints']*3]       # B, F, J*3            局部位置坐标
            glbl_p_gt = target_output[:, :, 0:opt['model']['num_joints']*3]  # B, F, J*3

            #------------------------global or local data-----------------------------------
            glbl_q_pred = glbl_q_pred.view(glbl_q_pred.shape[0], glbl_q_pred.shape[1], -1, 4)          #pred r and p data
            glbl_p_pred = glbl_p_pred.view(glbl_p_pred.shape[0], glbl_p_pred.shape[1], -1, 3)
            glbl_p_pred_ = glbl_p_pred / torch.norm(glbl_p_pred, dim=-1, keepdim=True)

            glbl_q_gt = glbl_q_gt.view(glbl_q_gt.size(0), glbl_q_gt.size(1), -1, 4)  # ground truth rotation and position data
            glbl_p_gt = glbl_p_gt.view(glbl_p_gt.size(0), glbl_p_gt.size(1), -1, 3)  # B, F, J, 3
            glbl_p_gt_ = glbl_p_gt/ torch.norm(glbl_p_gt, dim=-1, keepdim=True)

            root_pred = glbl_p_pred_[:, :, 0:1, :]         # B, F, 1, 3   根节点预测值
            root_gt = glbl_p_gt_[:, :, 0:1, :]           # B, F, 1, 3

            #----------------------------local data-------------------------------------
            # local_q_pred, local_p_pred = uf.quat_ik(glbl_q_pred.detach().cpu().numpy(), glbl_p_pred_.detach().cpu().numpy(), parents)
            # local_q_gt, local_p_gt = uf.quat_ik(glbl_q_gt.detach().cpu().numpy(), glbl_p_gt_.detach().cpu().numpy(), parents)
            # local_p_gt


            # loss --------------------------------------
            # loss_ik += torch.mean(torch.abs(glbl_p_pred_ - glbl_p_gt_) / x_std_n)   # ik运动学损失                                                            # Lik反运动学损失
            loss_quat += torch.mean(torch.abs(glbl_q_pred - glbl_q_gt))       # 旋转四元数损失
            loss_position += torch.mean(torch.abs(glbl_p_pred - glbl_p_gt) / x_std_n)     # 位移损失
            loss_root += torch.mean(torch.abs(root_pred - root_gt) / x_std_n)


            # 计算损失函数
            loss_total = opt['train']['loss_quat_weight'] * loss_quat + \
                         opt['train']['loss_fk_weight'] * loss_position + \
                         opt['train']['loss_root_weight'] * loss_root
                         # opt['train']['loss_fk_weight'] * loss_fk

            # update parameters
            loss_total.backward()
            optimizer.step()
            # loss_fk = opt['train']['loss_fk_weight'] * loss_fk
            loss_root = opt['train']['loss_root_weight'] * loss_root
            loss_quat = opt['train']['loss_quat_weight'] * loss_quat
            loss_pos = opt['train']['loss_position_weight'] * loss_position
            # loss_fk_list.append(loss_fk.item())
            loss_quat_list.append(loss_quat.item())
            loss_pos_list.append(loss_pos.item())
            loss_root_list.append(loss_root.item())
            loss_total_list.append(loss_total.item())

            # error computing---------------------------------------------

            # global info in prediction area
            trans_glbl_p_pred = (glbl_p_pred[:, opt['model']['n_past']: opt['model']['n_past']+opt['model']['n_trans'], ...] - x_mean_n) / x_std_n  #  Normalization
            trans_glbl_q_pred = glbl_q_pred[:, opt['model']['n_past']: opt['model']['n_past']+opt['model']['n_trans'], ...]  # 过渡区间的局部旋转真实值  B, n_trans, J, 4

            trans_glbl_p_gt = (glbl_p_gt[:, opt['model']['n_past']: opt['model']['n_past']+opt['model']['n_trans'], ...] - x_mean_n ) /x_std_n
            trans_glbl_q_gt = glbl_q_gt[:, opt['model']['n_past']: opt['model']['n_past']+opt['model']['n_trans'], ...]


            # 评估指标： L2Q、L2P和NPSS
            # global quaternion loss L2Q
            # print(f"trans_local_q_pred:{trans_local_q_pred.shape}trans_local_q_gt: {trans_local_q_gt.shape}")
            l2q_error = np.mean(np.sqrt(np.sum((trans_glbl_q_pred.detach().cpu().numpy() - trans_glbl_q_gt.detach().cpu().numpy()) ** 2.0, axis=(2, 3))))
             # 插值的全局旋转四元数 - 真实的全局旋转四元数 二范数 最后求均值
            # Global positions loss L2P
            l2p_error = np.mean(np.sqrt(np.sum((trans_glbl_p_pred.detach().cpu().numpy() - trans_glbl_p_gt.detach().cpu().numpy()) ** 2.0, axis=(2, 3))))

            # NPSS loss on global quaternions
            npss_error = bench.fast_npss(flatjoints(trans_glbl_q_gt.detach().cpu().numpy()), flatjoints(trans_glbl_q_pred.detach().cpu().numpy()))

            L2Q_error_list.append(l2q_error.item())
            L2P_error_list.append(l2p_error.item())
            NPSS_error_list.append(npss_error.item())

        checkpoint = {
            'model': model.state_dict(),
            'epoch': epoch_i
        }
        #  calculate mean error
        loss_total_cur = np.mean(loss_total_list)
        loss_root_cur = np.mean(loss_root_list)
        loss_fk_cur = np.mean(loss_fk_list)
        loss_pos_cur = np.mean(loss_pos_list)
        loss_quat_cur = np.mean(loss_quat_list)
        l2q_error_cur = np.mean(L2Q_error_list)
        l2p_error_cur = np.mean(L2P_error_list)
        npss_error_cur = np.mean(NPSS_error_list)
        # record the lowest error
        if loss_total_cur < loss_total_min:
            loss_total_min = loss_total_cur
        if npss_error_cur < npss_total_min:
            npss_total_min = npss_error_cur
        if l2q_error_cur < l2q_total_min:
            l2q_total_min = l2q_error_cur
        if l2p_error_cur < l2p_total_min:
            l2p_total_min = l2p_error_cur

        print('[train epoch: %5d] cur total loss: %.6f, '
              'cur best loss:%.6f, cur best NPSS: %.4f, '
              'cur best L2Q: %.4f, cur best L2P: %.4f '
              % (epoch_i, loss_total_cur, loss_total_min, npss_total_min, l2q_total_min, l2p_total_min))
        viz.line([loss_total_cur], [x_cor], win=opt['train']['picture_name'], update='append')
        # viz.line([loss_fk_cur], [x_cor], win='loss_fk_cur', update='append')
        viz.line([loss_root_cur], [x_cor], win='loss_root_cur', update='append')
        viz.line([loss_pos_cur], [x_cor], win='loss_pos_cur', update='append')
        viz.line([loss_quat_cur], [x_cor], win='loss_quat_cur', update='append')
        viz.line([l2q_error_cur], [x_cor], win='L2Q', update='append')
        viz.line([l2p_error_cur], [x_cor], win='L2P', update='append')
        viz.line([npss_error_cur], [x_cor], win='NPSS', update='append')
        x_cor += 10

        if epoch_i % opt['train']['save_per_epochs'] == 0 or epoch_i == 1:
            filename = os.path.join(opt['train']['output_dir'], f'epoch_{epoch_i}.pt')
            torch.save(checkpoint, filename)


