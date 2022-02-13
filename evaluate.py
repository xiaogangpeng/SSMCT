import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from utils.LaFan import LaFan1
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml
import time
import utils.benchmarks as ben
from utils.skeleton import Skeleton
import torch

if __name__ == '__main__':
    opt = yaml.load(open('config/train_config_lafan.yaml', 'r').read(), Loader=yaml.FullLoader)
    stamp = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime(time.time()))
    stamp = stamp + '-' + opt['train']['method']
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if opt['train']['debug']:
        stamp = 'debug'
    # log_dir = os.path.join('log', stamp)
    # model_dir = os.path.join('model', stamp)
    # if not os.path.exists(log_dir):
    #     os.mkdir(log_dir)
    # if not os.path.exists(model_dir):
    #     os.mkdir(model_dir)

    # def copydirs(from_file, to_file):
    #     if not os.path.exists(to_file):
    #         os.makedirs(to_file)
    #     files = os.listdir(from_file)
    #     for f in files:
    #         if os.path.isdir(from_file + '/' + f):
    #             copydirs(from_file + '/' + f, to_file + '/' + f)
    #         else:
    #             if '.git' not in from_file:
    #                 shutil.copy(from_file + '/' + f, to_file + '/' + f)
    # copydirs('./', log_dir + '/src')

    lafan_data_train = LaFan1(opt['data']['data_dir'],
                              opt['data']['data_set'],
                              seq_len = opt['model']['seq_length'],
                              offset = opt['data']['offset'],
                              train = True,
                              debug=opt['train']['debug'])
    print("train_positions.shape", lafan_data_train.data['X'].shape)
    print("train_rotations.shape", lafan_data_train.data['local_q'].shape)
    lafan_loader_train = DataLoader(lafan_data_train,
                                    batch_size=opt['train']['batch_size'],
                                    shuffle=True,
                                    num_workers=opt['data']['num_workers'])

    x_mean = lafan_data_train.x_mean.to(device)
    x_std = lafan_data_train.x_std.to(device).view(1, 1, opt['model']['num_joints'], 3)

    skeleton_mocap = Skeleton(offsets=opt['data']['offsets'], parents=opt['data']['parents'])
    skeleton_mocap.to(device)

    if opt['data']['data_set'] == "lafan":
        skeleton_mocap.remove_joints(opt['data']['joints_to_remove'])

    offsets = skeleton_mocap.offsets().detach().cpu().numpy()

    print(f"Start training for {opt['train']['num_epoch']} epochs")
    for epoch in range(opt['train']['num_epoch']):
        for batch_i, batch_data in tqdm(enumerate(lafan_loader_train)):
            positions = batch_data['X']  # B, F, J, 3
            rotations = batch_data['local_q']
            print(f"x.shape: {positions.shape}")
            ben.benchmark_interpolation(X=positions, Q=rotations, x_mean=x_mean, x_std = x_std,
                                        offsets=offsets,parents=opt['data']['parents'],
                                        n_future=opt['model']['n_future'], n_past=opt['model']['n_past'])
