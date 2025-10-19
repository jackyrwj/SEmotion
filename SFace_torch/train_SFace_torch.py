# train_SFace_torch备份
import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter

from config import get_config
from image_iter_rec import FaceDataset
from backbone.model_irse import IR_50, IR_101
from backbone.model_mobilefacenet import MobileFaceNet
from head.metrics import SFaceLoss

from util.utils import separate_irse_bn_paras, separate_resnet_bn_paras, separate_mobilefacenet_bn_paras
from util.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy
import math
import time
from IPython import embed


# from torcheeg.model_selection import train_test_split_per_subject_cross_trial, LeaveOneSubjectOut,KFold, KFoldPerSubjectCrossTrial, LeaveOneSubjectOut,KFoldCrossTrial,KFoldPerSubject,KFoldPerSubjectCrossTrial,KFoldGroupbyTrial
from torcheeg.model_selection import LeaveOneSubjectOut,KFold, KFoldPerSubjectCrossTrial, LeaveOneSubjectOut,KFoldCrossTrial,KFoldPerSubject,KFoldPerSubjectCrossTrial,KFoldGroupbyTrial
from torcheeg.datasets import DEAPDataset, SEEDDataset,MAHNOBDataset,SEEDIVDataset
import argparse
import time
import pretty_errors
from split_session import train_test_split_per_subject_cross_trial


from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT,DEAP_CHANNEL_LIST
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LOCATION_DICT,SEED_STANDARD_ADJACENCY_MATRIX
from torcheeg.datasets.constants.emotion_recognition.seed_iv import \
    SEED_IV_STANDARD_ADJACENCY_MATRIX
from torcheeg import transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
from torcheeg.models import CCNN, FBCCNN,TSCeption,DGCNN
from torcheeg.models.pyg import RGNN
import numpy as np
from torch.utils.data.sampler import Sampler
from braindecode.models import EEGConformer
from datetime import datetime
from loss_v2 import grad_cam_loss_v2
from loss_v1 import grad_cam_loss_v1
import sys
from sklearn.cluster import KMeans
from torcheeg.transforms.pyg import ToG
import torch_geometric


def xavier_normal_(tensor, gain=1., mode='avg'):
    fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(tensor)
    if mode == 'avg':
        fan = fan_in + fan_out
    elif mode == 'in':
        fan = fan_in
    elif mode == 'out':
        fan = fan_out
    else:
        raise Exception('wrong mode')
    std = gain * math.sqrt(2.0 / float(fan))

    return nn.init._no_grad_normal_(tensor, 0., std)


def weight_init(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d):
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.fill_(1)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()
        if hasattr(m, 'running_mean') and m.running_mean is not None:
            m.running_mean.data.zero_()
        if hasattr(m, 'running_var') and m.running_var is not None:
            m.running_var.data.fill_(1)
    elif isinstance(m, nn.PReLU):
        m.weight.data.fill_(1)
    else:
        if hasattr(m, 'weight') and m.weight is not None:
            xavier_normal_(m.weight.data, gain=2, mode='out')
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.zero_()

def schedule_lr(optimizer):
    for params in optimizer.param_groups:
        params['lr'] /= 10.



def valid(dataloader, model, HEAD, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    loss, correct = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            X = batch[0].to(DEVICE)
            y = batch[1].to(DEVICE)

            pred = model(X)

            # outputs = HEAD(pred, y)

            outputs = pred

            # loss += loss_fn(outputs[0], y).item()
            # correct += (outputs[0].argmax(1) == y).type(torch.float).sum().item()
            loss += loss_fn(outputs, y).item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()

    loss /= num_batches
    correct /= size
    print(f"Valid Error: \n Accuracy: {(correct):.8%}%, Avg loss: {loss}")

    return correct, loss


high_quality_sample = []
low_quality_sample = []


def test(dataloader, model, HEAD):
    model.eval()
    with torch.no_grad():
        idx = 0
        for batch in dataloader:
            X = batch[0].to(DEVICE)
            y = batch[1].to(DEVICE)

            pred = model(X)
            outputs = HEAD(pred, y)

            high_quality_sample.extend(idx + torch.nonzero((outputs[0].argmax(1) == y) == True).flatten().cpu().numpy())
            low_quality_sample.extend(idx + torch.nonzero((outputs[0].argmax(1) == y) == False).flatten().cpu().numpy())
            idx += 256

    np.save('./results/high.npy', high_quality_sample)
    np.save('./results/low.npy', low_quality_sample)

    
    

def need_save(acc, highest_acc):
    do_save = False
    save_cnt = 0
    if acc[0] > 0.98:
        do_save = True
    for i, accuracy in enumerate(acc):
        if accuracy > highest_acc[i]:
            highest_acc[i] = accuracy
            do_save = True
        if i > 0 and accuracy >= highest_acc[i]-0.002:
            save_cnt += 1
    if save_cnt >= len(acc)*3/4 and acc[0]>0.99:
        do_save = True
    print("highest_acc:", highest_acc)
    return do_save

class Logger(object):
    def __init__(self, filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.terminal = sys.stdout
        self.log = open(filename, "a")
    def write(self, message):
        self.log.write(message)
        self.terminal.write(message)
        self.log.flush()    #缓冲区的内容及时更新到log文件中
    def flush(self):
        pass
# sys.stdout = Logger(f'./logs/fbccnn_fold0_classical1.txt')
# sys.stdout = Logger(f'./logs/fbccnn_fold1_classical1.txt')
# sys.stdout = Logger(f'./logs/fbccnn_fold2_classical1.txt')
# sys.stdout = Logger(f'./logs/fbccnn_fold3_classical1.txt')
# sys.stdout = Logger(f'./logs/fbccnn_fold4_classical1.txt')

# sys.stdout = Logger(f'./logs/tscption_fold0_classical1.txt')
# sys.stdout = Logger(f'./logs/tscption_fold1_classical1.txt')
# sys.stdout = Logger(f'./logs/tscption_fold2_classical1.txt')
# sys.stdout = Logger(f'./logs/tscption_fold3_classical1.txt')
# sys.stdout = Logger(f'./logs/tscption_fold4_classical1.txt')



class CustomDataset(DEAPDataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class.
    """

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.info = dataset.info.iloc[indices,:]
        self.root_path = dataset.root_path
        self.chunk_size = dataset.chunk_size
        self.overlap = dataset.overlap
        self.num_channel = dataset.num_channel
        self.num_baseline = dataset.num_baseline
        self.baseline_chunk_size = dataset.baseline_chunk_size
        self.online_transform = dataset.online_transform
        self.offline_transform = dataset.offline_transform
        self.label_transform = dataset.label_transform
        self.before_trial = dataset.before_trial
        self.after_trial = dataset.after_trial
        self.num_worker = dataset.num_worker
        self.verbose = dataset.verbose
        self.io_path=dataset.io_path
        self.io_size=dataset.io_size
        self.io_mode=dataset.io_mode
        self.in_memory=dataset.in_memory
        self.eeg_io = dataset.eeg_io

    # def __len__(self):
    #     return len(self.indices)

    def __getitem__(self, index):
        # x, y = self.dataset[self.indices[index]]
        # return x, y

        info = self.read_info(index)

        eeg_index = str(info['clip_id'])
        eeg = self.read_eeg(eeg_index)

        baseline_index = str(info['baseline_id'])
        baseline = self.read_eeg(baseline_index)

        signal = eeg
        label = info

        if self.online_transform:
            signal = self.online_transform(eeg=eeg, baseline=baseline)['eeg']

        if self.label_transform:
            label = self.label_transform(y=info)['y']

        return signal, label
        # info = self.read_info(index)
        # label = self.label_transform(y=info)['y']
        # return self.info.iloc[index], label


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='1', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='4,5', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='3,4', type=str)

    parser.add_argument('--workers_id', help="gpu ids or cpu", default='7', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='6', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='5', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='4', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='3', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='2', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='1', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='0', type=str)
    # parser.add_argument('--epochs', help="training epochs", default=1, type=int)
    parser.add_argument('--epochs', help="training epochs", default=300, type=int)
    # parser.add_argument('--epochs', help="training epochs", default=200, type=int)

    parser.add_argument('--stages', help="training stages", default='200,250', type=str)
    # parser.add_argument('--stages', help="training stages", default='100,150', type=str)
    # parser.add_argument('--stages', help="training stages", default='80,90', type=str)
    parser.add_argument('--lr',help='learning rate',default=1e-2, type=float)
    # parser.add_argument('--lr',help='learning rate',default=1e-1, type=float)
    # parser.add_argument('--batch_size', help="batch_size", default=16, type=int)
    # parser.add_argument('--batch_size', help="batch_size", default=32, type=int)
    parser.add_argument('--batch_size', help="batch_size", default=64, type=int)
    # parser.add_argument('--batch_size', help="batch_size", default=128, type=int)
    # parser.add_argument('--batch_size', help="batch_size", default=256, type=int)
    parser.add_argument('--data_mode', help="use which database, [casia, vgg, ms1m, retina, ms1mr]",default='casia', type=str)
    parser.add_argument('--net', help="which network, ['IR_50', 'IR_101', 'MobileFaceNet']",default='IR_50', type=str)
    parser.add_argument('--head', help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']", default='ArcFace', type=str)
    parser.add_argument('--target', help="verification targets", default='lfw', type=str)
    parser.add_argument('--resume_backbone', help="resume backbone model", default='./results/IR_50-sface-casia/Backbone_IR_50_Epoch_50_Batch_12000_Time_2023-11-20-11-01_checkpoint.pth', type=str)
    parser.add_argument('--resume_head', help="resume head model", default='./results/IR_50-sface-casia/Head_ArcFace_Epoch_50_Batch_12000_Time_2023-11-20-11-01_checkpoint.pth', type=str)

    parser.add_argument('--outdir', help="output dir", default='output_model', type=str)
    parser.add_argument('--param_s', default=64.0, type=float)
    parser.add_argument('--param_k', default=80.0, type=float)
    parser.add_argument('--param_a', default=0.8, type=float)
    parser.add_argument('--param_b', default=1.23, type=float)
    args = parser.parse_args()

    current_time = datetime.now()
    print(current_time)

    #======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg['SEED'] # random seed for reproduce results
    # torch.manual_seed(SEED)
    torch.manual_seed(3407)
    # torch.manual_seed(114514)

    DATA_ROOT = cfg['DATA_ROOT'] # the parent root where your train data are stored
    EVAL_PATH = cfg['EVAL_PATH'] # the parent root where your val data are stored
    WORK_PATH = cfg['WORK_PATH'] # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT'] # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['IR_50', 'IR_101']
    HEAD_NAME = cfg['HEAD_NAME']

    INPUT_SIZE = cfg['INPUT_SIZE']
    EMBEDDING_SIZE = cfg['EMBEDDING_SIZE'] # feature dimension
    BATCH_SIZE = cfg['BATCH_SIZE']
    DROP_LAST = cfg['DROP_LAST'] # whether drop the last batch to ensure consistent batch_norm statistics
    LR = cfg['LR'] # initial LR
    NUM_EPOCH = cfg['NUM_EPOCH']
    WEIGHT_DECAY = cfg['WEIGHT_DECAY']
    MOMENTUM = cfg['MOMENTUM']
    STAGES = cfg['STAGES'] # epoch stages to decay learning rate

    DEVICE = cfg['DEVICE']
    MULTI_GPU = cfg['MULTI_GPU'] # flag to use multiple GPUs
    GPU_ID = cfg['GPU_ID'] # specify your GPU ids
    print('GPU_ID', GPU_ID)
    TARGET = cfg['TARGET']
    print("=" * 60)
    print("Overall Configurations:")
    print(cfg)
    # with open(os.path.join(WORK_PATH, 'config.txt'), 'w') as f:
    #     f.write(str(cfg))
    print("=" * 60)

    # writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True
    # NUM_CLASS = 4
    # NUM_CLASS = 3
    NUM_CLASS = 2



    
    # FBCCNN
    dataset = DEAPDataset(
        # io_path=f'./tmp_out/examples_fbccnn/deap_gaussian',
        # io_path=f'./tmp_out/examples_fbccnn/deap_salt',
        # io_path=f'./tmp_out/examples_fbccnn/deap_for_conformer',
        io_path=f'./tmp_out/examples_fbccnn/deap',
        root_path='./tmp_in/data_preprocessed_python',
        offline_transform=transforms.Compose([
            #-----noise------
            # transforms.AddGaussianNoise(0,1,1),
            # transforms.AddSaltPepperNoise(0.2),
            #---------------
            transforms.BandDifferentialEntropy(apply_to_baseline=True),
            transforms.BaselineRemoval(), 
            # transforms.MeanStdNormalize(),
            transforms.ToGrid(DEAP_CHANNEL_LOCATION_DICT)
        ]),
        online_transform=transforms.ToTensor(),
        label_transform=transforms.Compose([
            transforms.Select('valence'),
            transforms.Binary(5.0),
        ]),
        num_worker=20
    )
    print(dataset[0])


    # 导入TSception
    # dataset = DEAPDataset(
    # io_path=f'./tmp_out/examples_tsception/deap',
    # # io_path=f'./tmp_out/examples_tsception/arousal',
    # root_path='./tmp_in/data_preprocessed_python',
    # chunk_size=512,
    # num_baseline=1,
    # baseline_chunk_size=512,
    # offline_transform=transforms.Compose([
    #     transforms.PickElectrode(
    #         transforms.PickElectrode.to_index_list([
    #             'FP1', 'AF3', 'F3', 'F7', 'FC5', 'FC1', 'C3', 'T7', 'CP5', 'CP1', 'P3', 'P7', 'PO3', 'O1', 'FP2',
    #             'AF4', 'F4', 'F8', 'FC6', 'FC2', 'C4', 'T8', 'CP6', 'CP2', 'P4', 'P8', 'PO4', 'O2'
    #         ], DEAP_CHANNEL_LIST)),
    #     transforms.To2d()
    # ]),
    # online_transform=transforms.ToTensor(),
    # label_transform=transforms.Compose([
    #     transforms.Select('valence'),
    #     transforms.Binary(5.0),
    # ]))

    # 导入EEGConformer
    # dataset = SEEDDataset(io_path=f'./tmp_out/examples_fbccnn/for_EEGConformer',
    # dataset = SEEDDataset(io_path=f'./tmp_out/examples_confomer/seed',
    #                 # root_path=f'./tmp_in/Preprocessed_EEG',
    #                 root_path=f'/data/dataset/SEED/Preprocessed_EEG',
    #                 online_transform=transforms.Compose([
    #                     transforms.ToTensor(),
    #                     transforms.To2d(),
    #                     transforms.Lambda(lambda x: x.squeeze())
    #                 ]),
    #                 label_transform=transforms.Compose([
    #                     transforms.Select('emotion'),
    #                     transforms.Lambda(lambda x: x + 1)
    #                 ]),
    #     num_worker=48)
    # print(dataset[0])

    # 导入RGNN_SEED
    # dataset = SEEDDataset(
    #                     io_path=f'./tmp_out/examples_rgnn/seed',
    #                     root_path=f'/data/dataset/SEED/Preprocessed_EEG',
    #                     offline_transform=transforms.BandDifferentialEntropy(),
    #                     online_transform=ToG(SEED_STANDARD_ADJACENCY_MATRIX),
    #                     label_transform=transforms.Compose([
    #                         transforms.Select('emotion'),
    #                         transforms.Lambda(lambda x: int(x) + 1),
    #                     ]),
    #                     num_worker=8)
    # print(dataset[0])

    # 导入RGNN_SEED_IV
    # dataset = SEEDIVDataset(
    #                     io_path=f'./tmp_out/examples_rgnn/seed_iv',
    #                     root_path='/data/dataset/SEED-IV/SEED_IV/eeg_raw_data',
    #                     offline_transform=transforms.BandDifferentialEntropy(),
    #                     online_transform=transforms.Compose([
    #                         ToG(SEED_IV_STANDARD_ADJACENCY_MATRIX)
    #                     ]),
    #                     label_transform=transforms.Select('emotion'),
    #                     num_worker=8)
    # print(dataset[0])


    #  导入DGCNN_SEED
    # dataset = SEEDDataset(
    #                     io_path=f'./tmp_out/examples_dgcnn/seed',
    #                 root_path=f'/data/dataset/SEED/Preprocessed_EEG',
    #                   offline_transform=transforms.BandDifferentialEntropy(band_dict={
    #                       "delta": [1, 4],
    #                       "theta": [4, 8],
    #                       "alpha": [8, 14],
    #                       "beta": [14, 31],
    #                       "gamma": [31, 49]
    #                   }),
    #                   online_transform=transforms.Compose([
    #                       transforms.ToTensor()
    #                   ]),
    #                   label_transform=transforms.Compose([
    #                       transforms.Select('emotion'),
    #                       transforms.Lambda(lambda x: x + 1)
    #                   ]))
    
    #  导入DGCNN_SEED_IV
    # dataset = SEEDIVDataset(
    #                     io_path=f'./tmp_out/examples_dgcnn/seed_iv',
    #                     root_path='/data/dataset/SEED-IV/SEED_IV/eeg_raw_data',
    #                   offline_transform=transforms.BandDifferentialEntropy(band_dict={
    #                       "delta": [1, 4],
    #                       "theta": [4, 8],
    #                       "alpha": [8, 14],
    #                       "beta": [14, 31],
    #                       "gamma": [31, 49]
    #                   }),
    #                   online_transform=transforms.Compose([
    #                       transforms.ToTensor()
    #                   ]),
    #                 label_transform=transforms.Select('emotion'),
    #                 num_worker=8)



    k_fold = KFold(n_splits=5, split_path=f'./tmp_out/examples_fbccnn/kfold', shuffle=True)
    # k_fold = KFold(n_splits=5, split_path=f'./tmp_out/examples_conformer/kfold', shuffle=True)
    # k_fold = KFoldCrossTrial(n_splits=10, split_path=f'./tmp_out/examples_tsception/KFoldCrossTrial', shuffle=True)
    # k_fold = KFoldPerSubjectCrossTrial(n_splits=10, split_path=f'./tmp_out/examples_tsception/KFoldPerSubjectCrossTrial10', shuffle=True)
    
    # k_fold = KFoldCrossTrial(n_splits=10, split_path=f'./tmp_out/examples_rgnn/KFoldCrossTrial', shuffle=True)
    # k_fold = KFoldGroupbyTrial(n_splits=5, split_path=f'./tmp_out/examples_rgnn/KFoldGroupbyTrial', shuffle=True)

    # k_fold = KFold(n_splits=5, split_path=f'./tmp_out/examples_rgnn/kfold', shuffle=True)
    # k_fold = KFold(n_splits=5, split_path=f'./tmp_out/examples_rgnn/kfold', shuffle=False)
    # k_fold = KFoldGroupbyTrial(n_splits=5, split_path=f'./tmp_out/examples_rgnn/KFoldGroupbyTrial', shuffle=True)
    # k_fold = KFoldCrossTrial(n_splits=5, split_path=f'./tmp_out/examples_rgnn/KFoldCrossTrial', shuffle=True)
    # k_fold = LeaveOneSubjectOut(split_path=f'./tmp_out/examples_rgnn/LeaveOneSubjectOut')
    # k_fold = KFoldPerSubject(n_splits=5, split_path=f'./tmp_out/examples_rgnn/KFoldPerSubject',shuffle=True)
    # k_fold = KFoldPerSubjectCrossTrial(n_splits=2, split_path=f'./tmp_out/examples_rgnn/KFoldPerSubjectCrossTrial2', shuffle=True)
    # k_fold = KFoldPerSubjectCrossTrial(n_splits=3, split_path=f'./tmp_out/examples_rgnn/KFoldPerSubjectCrossTrial3', shuffle=True)
    # k_fold = KFoldPerSubjectCrossTrial(n_splits=4, split_path=f'./tmp_out/examples_rgnn/KFoldPerSubjectCrossTrial4', shuffle=True)
    # k_fold = KFoldPerSubjectCrossTrial(n_splits=5, split_path=f'./tmp_out/examples_rgnn/KFoldPerSubjectCrossTrial5', shuffle=True)
    # k_fold = KFoldPerSubjectCrossTrial(n_splits=15, split_path=f'./tmp_out/examples_rgnn/KFoldPerSubjectCrossTrial15', shuffle=True)
    
    # k_fold = KFold(n_splits=5, split_path=f'./tmp_out/examples_rgnn/kfold_seed_iv', shuffle=False)
    # k_fold = KFoldGroupbyTrial(n_splits=5, split_path=f'./tmp_out/examples_rgnn/KFoldGroupbyTrial_iv', shuffle=True)
    # k_fold = KFoldCrossTrial(n_splits=5, split_path=f'./tmp_out/examples_rgnn/KFoldCrossTrial_iv', shuffle=True)
    
    
    # *********************数据划分*************************************************************************
    # for i in range(2, 16):
    #     # if i == 1:
    #     if 2 <= i <= 3:
    #     # if 4 <= i <= 5:
    #     # if 6 <= i <= 7:
    #     # if 8 <= i <= 9:
    #     # if 10 <= i <= 11:
    #     # if 12 <= i <= 13:
    #     # if 14 <= i <= 15:

    #         print(f"*****************sub: {i}*****************")
    #         train_dataset, val_dataset = train_test_split_per_subject_cross_trial(
    #             dataset=dataset, 
    #             shuffle = False,
    #             # test_size = 0.1,
    #             # test_size = 0.2,
    #             # test_size = 0.3,
    #             test_size = 0.4,
    #             subject=i,
    #             # split_path=f"./dataset/seed/{i}/0.1"
    #             # split_path=f"./dataset/seed/{i}/0.2"
    #             # split_path=f"./dataset/seed/{i}/0.3"
    #             split_path=f"./dataset/seed/{i}/0.4"
                
    #             # split_path=f"./dataset/seed_iv/{i}/0.1"
    #             # split_path=f"./dataset/seed_iv/{i}/0.2"
    #             # split_path=f"./dataset/seed_iv/{i}/0.3"
    #             # split_path=f"./dataset/seed_iv/{i}/0.4"

    #             # ---------------------dgcnn------------------------
    #             # split_path=f"./dataset/dgcnn/seed/{i}/0.1"
    #             # split_path=f"./dataset/dgcnn/seed/{i}/0.2"
    #             # split_path=f"./dataset/dgcnn/seed/{i}/0.3"
    #             # split_path=f"./dataset/dgcnn/seed/{i}/0.4"
                
    #             # split_path=f"./dataset/dgcnn/seed_iv/{i}/0.1"
    #             # split_path=f"./dataset/dgcnn/seed_iv/{i}/0.2"
    #             # split_path=f"./dataset/dgcnn/seed_iv/{i}/0.3"
    #             # split_path=f"./dataset/dgcnn/seed_iv/{i}/0.4"
    #             )
    #         print('ok')
    # *********************************************************************************************************************************************************************************************************************
        
        
        
    total_sum_higest = 0
    total_sum_normal = 0
    higest_acc = 0
    count = 0
    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        if i == 0:
        # if 1 <= i <= 2:
        # if 3 <= i <= 4:
        # if 5 <= i <= 6:
        # if 7 <= i <= 8:
        # if 9 <= i <= 10:
        # if 11 <= i <= 12:
        # if 13 <= i <= 14:
            print(f"*****************fold: {i}*****************")
            # 手动cuda                                               
            # t1 = time.time()
            # train_dataset_cuda = []
            # val_dataset_cuda = []
            # for x in range(len(train_dataset)):
            #     train_dataset_cuda.append((train_dataset[x][0].to(DEVICE), train_dataset[x][1]))
            # for y in range(len(val_dataset)):
            #     val_dataset_cuda.append((val_dataset[y][0].to(DEVICE), val_dataset[y][1]))
            # t2 = time.time()
            # print(t2 - t1)

            # 普通卷积
            # allloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
            # trainloader = torch.utils.data.DataLoader(train_dataset_cuda, batch_size=BATCH_SIZE, shuffle=True)
            # validloader = torch.utils.data.DataLoader(val_dataset_cuda, batch_size=BATCH_SIZE, shuffle=True)
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            validloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
            
            
            # 图卷积
            # trainloader = torch_geometric.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            # validloader = torch_geometric.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
            # trainloader = torch_geometric.data.DataLoader(train_dataset_cuda, batch_size=BATCH_SIZE, shuffle=True)
            # validloader = torch_geometric.data.DataLoader(val_dataset_cuda, batch_size=BATCH_SIZE, shuffle=True)


            #======= model & loss & optimizer =======#
            # BACKBONE = CCNN(num_classes=EMBEDDING_SIZE, in_channels=4, grid_size=(9, 9))
            BACKBONE = FBCCNN(num_classes=EMBEDDING_SIZE, in_channels=4, grid_size=(9, 9))
            # BACKBONE = TSCeption(num_classes=EMBEDDING_SIZE,
            #                      num_electrodes=28,
            #                      sampling_rate=128,
            #                      num_T=15,
            #                      num_S=15,
            #                      hid_channels=32,
            #                      dropout=0.5)
            # BACKBONE = RGNN(adj=torch.Tensor(SEED_STANDARD_ADJACENCY_MATRIX),
            #                     # in_channels=5,
            #                     in_channels=4,
            #                     num_electrodes=62,
            #                     hid_channels=32,
            #                     num_layers=2,
            #                     # num_classes=3,
            #                     num_classes=EMBEDDING_SIZE,
            #                     dropout=0.7,
            #                     # dropout=0.5,
            #                     learn_edge_weights=True)

            
            # BACKBONE = DGCNN(in_channels=5, 
            #                  num_electrodes=62, 
            #                  hid_channels=32, 
            #                  num_layers=2, 
            #                  num_classes=EMBEDDING_SIZE)
            # BACKBONE = EEGConformer(n_chans=62,n_outputs=EMBEDDING_SIZE,n_times=200,final_fc_length='auto')


            # --------------------------导入模型--------------------------------------------------------------------------------------------------------------------------
            # # # checkpoint = torch.load(f'./results/FBCCNN_model/model0.pt')
            # # # checkpoint = torch.load(f'./results/FBCCNN_model/model1.pt')

            # # checkpoint = torch.load(f'./results/TSCeption_model/model0.pt')
            # # checkpoint = torch.load(f'./results/TSCeption_model/model1.pt')
            # checkpoint = torch.load(f'./results/best_model/Backbone_Epoch_1_Batch_421_Time_2024-02-27-16-44_checkpoint.pth')
            # param = checkpoint['model']
            # BACKBONE.load_state_dict(param)
            # -------------------------------------------------------------------------------------------------------------------------------------------------------------

            print("=" * 60)
            print(BACKBONE)
            print("{} Backbone Generated".format(BACKBONE_NAME))
            print("=" * 60)

            HEAD = SFaceLoss(in_features = EMBEDDING_SIZE, 
                             out_features=NUM_CLASS, 
                             device_id=GPU_ID,
                            s = args.param_s, 
                            k = args.param_k, 
                            a = args.param_a, 
                            b = args.param_b)
            print("=" * 60)
            print(HEAD)        
            print("{} Head Generated".format(HEAD_NAME))
            print("=" * 60)
            LOSS = nn.CrossEntropyLoss()

            # OPTIMIZER=torch.optim.SGD(BACKBONE.parameters(), lr = 0.1) 
            # OPTIMIZER=torch.optim.SGD(BACKBONE.parameters(), lr = 0.01) 
            # OPTIMIZER=torch.optim.SGD(BACKBONE.parameters(), lr = 0.001) 
            # OPTIMIZER=torch.optim.SGD(BACKBONE.parameters(), lr = 0.0002) 

            # OPTIMIZER=torch.optim.Adam(BACKBONE.parameters(), lr = 0.1) 
            # OPTIMIZER=torch.optim.Adam(BACKBONE.parameters(), lr = 0.01) 
            # OPTIMIZER=torch.optim.Adam(BACKBONE.parameters(), lr = 0.001) 
            # OPTIMIZER=torch.optim.Adam(BACKBONE.parameters(), lr = 0.0005) 


            ## good
            OPTIMIZER=torch.optim.Adam(BACKBONE.parameters(), lr = 0.0002, betas=(0.5,0.999)) 
            print("=" * 60)
            print(OPTIMIZER)
            print("Optimizer Generated")
            print("=" * 60)

            ## ----------------聚类 计算和典型中心的距离-------------------------------------------------------------------------------------------------------------------# 

            # 不同的fold   5kfold
            array = np.load('./results/FBCCNN_classical/classical.npy')
            # array = np.load('./results/FBCCNN_classical/classical1.npy')
            # array = np.load('./results/FBCCNN_classical/classical2.npy')
            # array = np.load('./results/FBCCNN_classical/classical3.npy')
            # array = np.load('./results/FBCCNN_classical/classical4.npy')

            # array = np.load('./results/TSCeption_classical/classical.npy')
            # array = np.load('./results/TSCeption_classical/classical1.npy')
            # array = np.load('./results/TSCeption_classical/classical2.npy')
            # array = np.load('./results/TSCeption_classical/classical3.npy')
            # array = np.load('./results/TSCeption_classical/classical4.npy')


            # 计算要置为1的数量
            top_percentage = 0.1
            top_10_percent = int(len(array) * top_percentage)

            # 获取元素大小前10%的元素下标
            indices = np.argsort(array)[:top_10_percent]

            # 将下标保存到 .npy 文件
            # np.save("top_indices.npy", indices)

            # 从DEAP数据集中按照下标取出数据
            classical_dataset = CustomDataset(dataset, indices)
            classicalloader = torch.utils.data.DataLoader(classical_dataset, 
                                                          batch_size=BATCH_SIZE, 
                                                          shuffle=False, 
                                                          num_workers=len(GPU_ID), 
                                                          drop_last=True)
            

            # 创建空数组用于存储每次的 inputs 和 labels
            inputs_array = []
            # labels_array = []


            # 使用迭代器遍历每个输入和标签
            for inputs, labels in iter(classicalloader):
                # 将每次的 inputs 存入 inputs_array 数组
                inputs_array.append(inputs)
                # # 将每次的 labels 存入 labels_array 数组
                # labels_array.append(labels)

            # 将 inputs_array 转换为一个张量
            inputs_tensor = torch.cat(inputs_array, dim=0)


            # ## 按照平均的方法求中心
            # # 计算样本的中心
            # center = torch.mean(inputs_tensor, dim=0)



            ## 按照KMeans的方法求中心
            # 将样本重新组织为二维数组
            reshaped_samples = inputs_tensor.reshape(-1, np.prod(inputs_tensor.shape[1:]))

            # 设置聚类的数量
            n_clusters = 1

            # 创建 KMeans 对象并进行聚类
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(reshaped_samples)

            # 获取聚类中心
            cluster_centers = kmeans.cluster_centers_

            # 将聚类中心重新恢复为原来的形状
            # center = cluster_centers.reshape(n_clusters, *inputs_tensor.shape[1:])
            center = torch.from_numpy(cluster_centers).reshape(n_clusters, *inputs_tensor.shape[1:])

            # 输出中心的形状
            print("Center shape:", center.shape)
            HEAD.setKnoWeight(center)


            #---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------# 
            

            # --------------------------导入最佳--------------------------------------------------------------------------------------------  
            # print("=" * 60)
            # # path = 'results/best_model/Backbone_Epoch_1_Batch_421_Time_2024-02-27-16-44_checkpoint.pth'

            # # path = './results/best_model/KFold_best' 
            # # path = './results/best_model/KFoldGroupbyTrial_best'
            # # path = './results/best_model/KFoldCrossTrial_best'
            # # path = './results/best_model/KFoldPerSubjectCrossTrial_best'
            # # path = './results/best_model/LeaveOneSubjectOut_best'
            # path = './results/best_model/KFoldPerSubject_best'
            # print("Loading Backbone Checkpoint '{}'".format(path))
            # BACKBONE.load_state_dict(torch.load(path))
            # print("=" * 60)
            # ----------------------------------------------------------------------------------------------------------------------------------
            
            #======= init & resume & MULTI_GPU=======#
            if MULTI_GPU:
                # multi-GPU setting
                BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
                BACKBONE = BACKBONE.to(DEVICE)
            else:
                # single-GPU setting
                BACKBONE = BACKBONE.to(DEVICE)

            #======= train & validation & save checkpoint =======#
            # DISP_FREQ = 800 # frequency to display training loss & acc
            DISP_FREQ = 100 # frequency to display training loss & acc
            # VER_FREQ = 100
            batch = 0  # batch index

            losses = AverageMeter()
            # intra_losses = AverageMeter()
            # inter_losses = AverageMeter()
            # Wyi_mean = AverageMeter()
            # Wj_mean = AverageMeter()
            # theta_yis = AverageMeter()
            # theta_js = AverageMeter()
            top1 = AverageMeter()
            # ------------------------------------------------------------------------
            BACKBONE.train()  # set to training mode
            # HEAD.train()
            for epoch in range(NUM_EPOCH):

                if epoch in STAGES:
                    schedule_lr(OPTIMIZER)

                last_time = time.time()

                catch_array = []
                counter = 0

                for inputs, labels in iter(trainloader):
                # for inputs, labels in iter(allloader):

                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE).long()
                    features = BACKBONE(inputs)
                    outputs = features
                    
                    # outputs, loss, intra_loss, inter_loss, WyiX, WjX, theta_yi, theta_j= HEAD(features, labels)
                    outputs, loss, intra_loss, inter_loss, WyiX, WjX= HEAD(features, labels)

                    # lossv1 = LOSS(outputs, labels) - torch.sigmoid(grad_cam_loss_v1(BACKBONE, inputs, labels, mode = 'SEED_ALL_FRONT', weight_nroi=0, layer = 'pp'))
                    # loss = sface_loss - torch.sigmoid(grad_cam_loss_v1(BACKBONE, inputs, labels, mode = 'SEED_ALL_FRONT', weight_nroi=0, layer = 'pp'))
                    # loss1 = LOSS(outputs, labels) - torch.sigmoid(grad_cam_loss_v1(BACKBONE, inputs, labels, mode = 'NEW_ALL_FRONT', weight_nroi=0, layer = 'BN_s'))
                    # loss1 = LOSS(outputs, labels) - torch.sigmoid(grad_cam_loss_v1(BACKBONE, inputs, labels, mode = 'ALL_FRONT', weight_nroi=0, layer = 'block2'))
                    # loss1 = loss - torch.sigmoid(grad_cam_loss_v1(BACKBONE, inputs, labels, mode = 'ALL_FRONT', weight_nroi=0, layer = 'block2'))
                    # outputs, loss2, intra_loss, inter_loss, WyiX, WjX, theta_yi, theta_j =  grad_cam_loss_v2(BACKBONE, HEAD, inputs, labels, mode = 'ALL', layer = 'pp')
                    # outputs, loss2, intra_loss, inter_loss, WyiX, WjX, theta_yi, theta_j =  grad_cam_loss_v2(BACKBONE, HEAD, inputs, labels, mode = 'NEW_ALL_FRONT', layer = 'BN_s')
                    
                    # FBCCNN
                    catch_array, loss, counter =  grad_cam_loss_v2(BACKBONE, HEAD, inputs, labels, mode = 'ALL_FRONT', layer = 'block2', array=catch_array, counter = counter)
                    
                    # TSCeption
                    # catch_array, loss, counter =  grad_cam_loss_v2(BACKBONE, HEAD, inputs, labels, mode = 'NEW_ALL_FRONT', layer = 'BN_s', array=catch_array, counter = counter)
                    

                    # loss = loss1
                    # loss = loss2
                    # loss = sface_loss + lossv1


                    # loss = LOSS(features, labels)
                    # loss = loss_cross + sface_loss

                    # loss = loss1 + sface_loss
                    

                ## --------------------------------------------保存典型---------------------------------------------------------------------------------------
                # # FBCCNN
                # # np.save('./results/FBCCNN_classical/classical.npy', catch_array)
                # # np.save('./results/FBCCNN_classical/classical1.npy', catch_array)
                # # np.save('./results/FBCCNN_classical/classical2.npy', catch_array)
                # # np.save('./results/FBCCNN_classical/classical3.npy', catch_array)
                # np.save('./results/FBCCNN_classical/classical4.npy', catch_array)
                    
                # TSCetpion
                # np.save('./results/TSCeption_classical/classical.npy', catch_array)
                # np.save('./results/TSCeption_classical/classical1.npy', catch_array)
                # np.save('./results/TSCeption_classical/classical2.npy', catch_array)
                # np.save('./results/TSCeption_classical/classical3.npy', catch_array)
                # np.save('./results/TSCeption_classical/classical4.npy', catch_array)
                ## ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

                    prec1 = train_accuracy(outputs.data, labels, topk=(1,))
                    losses.update(loss.data.item(), inputs.size(0))
                    # intra_losses.update(intra_loss.data.item(), inputs.size(0))
                    # inter_losses.update(inter_loss.data.item(), inputs.size(0))
                    # Wyi_mean.update(WyiX.data.item(), inputs.size(0))
                    # Wj_mean.update(WjX.data.item(), inputs.size(0))
                    # theta_yis.update(theta_yi.data.item(), inputs.size(0))
                    # theta_js.update(theta_j.data.item(), inputs.size(0))
                    top1.update(prec1.data.item(), inputs.size(0))

                    OPTIMIZER.zero_grad()
                    loss.backward()
                    OPTIMIZER.step()
                    batch += 1  # batch index

                    if ((batch + 1) % DISP_FREQ == 0) and batch != 0:

                        batch_time = time.time() - last_time
                        last_time = time.time()

                        print('Epoch {} Batch {}\t'
                            # 'Speed: {speed:.2f} samples/s\t'
                            # 'intra_Loss {loss1.val:.4f} ({loss1.avg:.4f})\t'
                            # 'inter_Loss {loss2.val:.4f} ({loss2.avg:.4f})\t'
                            # 'Wyi {Wyi.val:.4f} ({Wyi.avg:.4f})\t'
                            # 'Wj {Wj.val:.4f} ({Wj.avg:.4f})\t'
                            # 'theta_yi {theta_yi.val:.4f} ({theta_yi.avg:.4f})\t'
                            # 'theta_j {theta_j.val:.4f} ({theta_j.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch + 1, batch + 1, 
                            # speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                            # loss1 = intra_losses, loss2 = inter_losses, 
                            # Wyi=Wyi_mean, Wj=Wj_mean, 
                            # theta_yi = theta_yis, theta_j = theta_js, 
                            top1=top1)
                            )
                            
                        losses = AverageMeter()
                        # intra_losses = AverageMeter()
                        # inter_losses = AverageMeter()
                        # Wyi_mean = AverageMeter()
                        # Wj_mean = AverageMeter()
                        # theta_yis = AverageMeter()
                        # theta_js = AverageMeter()
                        top1 = AverageMeter()



                    # if ((batch + 1) % VER_FREQ == 0) and batch != 0: #perform validation & save checkpoints (buffer for visualization)
                    # if ((batch + 1) % 400 == 0) and batch != 0: #perform validation & save checkpoints (buffer for visualization)
                    # if (epoch + 1) % NUM_EPOCH == 0 and epoch != 0:
                    # if (epoch + 1) % 25 == 0 and epoch != 0:
                            
                val_acc, val_loss= valid(validloader, BACKBONE, HEAD, nn.CrossEntropyLoss())
                if val_acc > higest_acc:
                    print('find hightest')
                    higest_acc = val_acc
                    ## --------------------------------------------保存最佳---------------------------------------------------------------------------------------
                    # torch.save(
                    #     BACKBONE.state_dict(), 
                    #         #    os.path.join(
                    #         #        "./results/best_model/Backbone_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                    #         #            epoch + 1, batch + 1,get_time()))
                            
                    #         # './results/best_model/KFold_best' 
                    #         # './results/best_model/KFoldGroupbyTrial_best'
                    #         # './results/best_model/KFoldCrossTrial_best'
                    #         # './results/best_model/KFoldPerSubjectCrossTrial_best'
                    #         # './results/best_model/LeaveOneSubjectOut_best'
                    #         './results/best_model/KFoldPerSubject_best'
                    #     )
                    ## -----------------------------------------------------------------------------------------------------------------------------------

                                

                    
                    # save checkpoints per epoch
                    # if MULTI_GPU:
                    #     torch.save(BACKBONE.module.state_dict(), os.path.join(WORK_PATH,
                    #                                                         "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                    #                                                             BACKBONE_NAME, epoch + 1, batch + 1,
                    #                                                             get_time())))
                    #     torch.save(HEAD.state_dict(), os.path.join(WORK_PATH,
                    #                                             "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                    #                                                 HEAD_NAME, epoch + 1, batch + 1, get_time())))
                    # else:
                    # torch.save(BACKBONE.state_dict(), os.path.join(WORK_PATH,
                    #                                             "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                    #                                                 BACKBONE_NAME, epoch + 1, batch + 1,
                    #                                                 get_time())))
                    # torch.save(HEAD.state_dict(), os.path.join(WORK_PATH,
                    #                                         "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(
                    #                                             HEAD_NAME, epoch + 1, batch + 1, get_time())))
                    

                BACKBONE.train()  # set to training mode
            
                
            current_time = datetime.now()
            print(current_time)
            # count += 1
            # total_sum_higest += higest_acc
            print(higest_acc)


            # higest_acc = 0
            # total_sum_normal += val_acc
        # print('%.5f' % (total_sum_higest / count))  # 取1位小数
        # print('%.5f' % (total_sum_normal / count))  # 取1位小数

            # test(allloader, BACKBONE, HEAD)
            # test(trainloader, BACKBONE, HEAD)


