import os, argparse, sklearn
import torch
import torch.nn as nn
import torch.optim as optim
# from tensorboardX import SummaryWriter

from config import get_config
from image_iter_rec import FaceDataset
from backbone.model_irse import IR_50, IR_101
from backbone.model_mobilefacenet import MobileFaceNet
from head.metrics import Softmax, ArcFace, CosFace, SphereFace, Am_softmax
from util.utils import separate_irse_bn_paras, separate_resnet_bn_paras, separate_mobilefacenet_bn_paras
from util.utils import get_val_data, perform_val, get_time, buffer_val, AverageMeter, train_accuracy
from braindecode.models import EEGConformer

import math
import time


from torcheeg.model_selection import KFold, LeaveOneSubjectOut,KFoldCrossTrial,KFoldPerSubject,KFoldPerSubjectCrossTrial,\
KFoldPerSubjectGroupbyTrial
from torcheeg.datasets import DEAPDataset, SEEDDataset,MAHNOBDataset
import argparse

from torcheeg.datasets.constants.emotion_recognition.deap import \
    DEAP_CHANNEL_LOCATION_DICT,DEAP_CHANNEL_LIST
from torcheeg.datasets.constants.emotion_recognition.seed import \
    SEED_CHANNEL_LOCATION_DICT
from torcheeg.datasets.constants.emotion_recognition.mahnob import \
    MAHNOB_CHANNEL_LOCATION_DICT



from torcheeg import transforms
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch
from torcheeg.models import CCNN, FBCCNN,TSCeption
import numpy as np
from datetime import datetime
from loss_v2 import grad_cam_loss_v2
from loss_v1 import grad_cam_loss_v1
import pretty_errors
import sys


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

    # print(optimizer)


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
# sys.stdout = Logger(f'./output/exp0/{args.CV}_seed_{args.session}.txt')
# sys.stdout = Logger(f'./logs/softmax.txt')

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

            outputs = model(X)

            loss += loss_fn(outputs, y).item()
            # loss += loss_fn(outputs[0], y).item()
            correct += (outputs.argmax(1) == y).type(torch.float).sum().item()

    loss /= num_batches
    correct /= size
    print(f"Valid Error: \n Accuracy: {(correct):.5%}, Avg loss: {loss}")

    return correct, loss




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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='6,7', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='4,5', type=str)
    parser.add_argument('--workers_id', help="gpu ids or cpu", default='3,4', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='7', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='6', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='5', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='4', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='3', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='2', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='3', type=str)
    # parser.add_argument('--workers_id', help="gpu ids or cpu", default='0', type=str)
    parser.add_argument('--epochs', help="training epochs", default=300, type=int)
    # parser.add_argument('--epochs', help="training epochs", default=100, type=int)
    parser.add_argument('--stages', help="training stages", default='200,250', type=str)
    # parser.add_argument('--stages', help="training stages", default='25,30,40', type=str)
    parser.add_argument('--lr',help='learning rate',default=5e-4, type=float)
    # parser.add_argument('--lr',help='learning rate',default=5e-4, type=float)
    parser.add_argument('--batch_size', help="batch_size", default=256, type=int)
    parser.add_argument('--data_mode', help="use which database, [casia, vgg, ms1m, retina, ms1mr]",default='casia', type=str)
    parser.add_argument('--net', help="which network, ['ResNet_50', 'ResNet_101', 'ResNet_152', 'IR_50', 'IR_101', 'IR_152', 'IR_SE_50', 'IR_SE_101', 'IR_SE_152', 'MobileFaceNet','HRNet']",default='MobileFaceNet', type=str)
    parser.add_argument('--head', help="head type, ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']", default='Softmax', type=str)
    parser.add_argument('--target', help="verification targets", default='lfw', type=str)
    parser.add_argument('--resume_backbone', help="resume backbone model", default='results/IR_50-arc-casia/Backbone_IR_50_Epoch_117_Batch_28000_Time_2023-11-16-17-29_checkpoint.pth', type=str)
    parser.add_argument('--resume_head', help="resume head model", default='results/IR_50-arc-casia/Head_Softmax_Epoch_117_Batch_28000_Time_2023-11-16-17-29_checkpoint.pth', type=str)
    parser.add_argument('--outdir', help="output dir", default='./results/IR_50-arc-casia', type=str)
    args = parser.parse_args()

    #======= hyperparameters & data loaders =======#
    cfg = get_config(args)

    SEED = cfg['SEED'] # random seed for reproduce results
    torch.manual_seed(SEED)

    DATA_ROOT = cfg['DATA_ROOT']  # the parent root where your train data are stored
    EVAL_PATH = cfg['EVAL_PATH']  # the parent root where your val data are stored
    WORK_PATH = cfg['WORK_PATH']  # the root to buffer your checkpoints and to log your train/val status
    BACKBONE_RESUME_ROOT = cfg['BACKBONE_RESUME_ROOT']  # the root to resume training from a saved checkpoint
    HEAD_RESUME_ROOT = cfg['HEAD_RESUME_ROOT']  # the root to resume training from a saved checkpoint

    BACKBONE_NAME = cfg['BACKBONE_NAME'] # support: ['IR_50', 'IR_101']
    HEAD_NAME = cfg['HEAD_NAME'] # support:  ['Softmax', 'ArcFace', 'CosFace', 'SphereFace', 'Am_softmax']


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
    with open(os.path.join(WORK_PATH, 'config.txt'), 'w') as f:
        f.write(str(cfg))
    print("=" * 60)

    # writer = SummaryWriter(WORK_PATH) # writer for buffering intermedium results
    torch.backends.cudnn.benchmark = True

    # with open(os.path.join(DATA_ROOT, 'property'), 'r') as f:
    #     NUM_CLASS, h, w = [int(i) for i in f.read().split(',')]
    #     print('NUM_CLASS%d' % NUM_CLASS)
    #     print()
    # assert h == INPUT_SIZE[0] and w == INPUT_SIZE[1]
    # NUM_CLASS = 2
    NUM_CLASS = 3

    current_time = datetime.now()
    print(current_time)

    # 导入FBCCNN
    dataset = DEAPDataset(
        io_path=f'./tmp_out/examples_fbccnn/deap',
        root_path='./tmp_in/data_preprocessed_python',
        offline_transform=transforms.Compose([
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

    
    # 导入 Confomer
    dataset = SEEDDataset(io_path=f'./tmp_out/examples_confomer/seed',
                    # root_path=f'./tmp_in/Preprocessed_EEG',
                    root_path=f'/data/dataset/SEED/Preprocessed_EEG',
                    online_transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.To2d(),
                        transforms.Lambda(lambda x: x.squeeze())
                    ]),
                    label_transform=transforms.Compose([
                        transforms.Select('emotion'),
                        transforms.Lambda(lambda x: x + 1)
                    ]),
        num_worker=48)
    
   

    
    k_fold = KFold(n_splits=5, split_path=f'./tmp_out/examples_conformer/kfold', shuffle=True)


    
    # allloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    total_sum_higest = 0
    total_sum_normal = 0
    higest_acc = 0
    count = 0
    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        if i == 0:
            print(f"*****************fold: {i}*****************")
            #======= good & bad sample =======#
            # high_array = np.load('./results/high_quality_sample_softmax.npy')
            # low_array = np.load('./results/low_quality_sample_softmax.npy')

            # high_dataset = CustomDataset(dataset, high_array)
            # low_dataset = CustomDataset(dataset, low_array)

            # 手动cuda                                               
            t1 = time.time()
            train_dataset_cuda = []
            val_dataset_cuda = []
            for x in range(len(train_dataset)):
                train_dataset_cuda.append((train_dataset[x][0].to(DEVICE), train_dataset[x][1]))
            for y in range(len(val_dataset)):
                val_dataset_cuda.append((val_dataset[y][0].to(DEVICE), val_dataset[y][1]))
            t2 = time.time()
            print(t2 - t1)

            # trainloader = torch.utils.data.DataLoader(high_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=len(GPU_ID), drop_last=True)
            # trainloader = torch.utils.data.DataLoader(low_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=len(GPU_ID), drop_last=True)
            # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=len(GPU_ID), drop_last=True)
            # validloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

            # trainloader = torch.utils.data.DataLoader(train_dataset_cuda, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True,num_workers=len(GPU_ID), drop_last=True)
            # validloader = torch.utils.data.DataLoader(val_dataset_cuda, batch_size=BATCH_SIZE, shuffle=True,  pin_memory=True, drop_last=True)

            trainloader = torch.utils.data.DataLoader(train_dataset_cuda, batch_size=BATCH_SIZE, shuffle=True)
            validloader = torch.utils.data.DataLoader(val_dataset_cuda, batch_size=BATCH_SIZE, shuffle=True)
            # trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            # validloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)

            
            



            #======= model & loss & optimizer =======#
            # BACKBONE = CCNN(num_classes=EMBEDDING_SIZE, in_channels=4, grid_size=(9, 9))
            # BACKBONE = FBCCNN(num_classes=EMBEDDING_SIZE, in_channels=4, grid_size=(9, 9))    
            # BACKBONE = TSCeption(num_classes=EMBEDDING_SIZE,num_electrodes=28,sampling_rate=128,num_T=15,num_S=15,hid_channels=32,dropout=0.5)
            # BACKBONE = TSCeption(num_classes=EMBEDDING_SIZE,
            #                     num_electrodes=28,
            #                     sampling_rate=128,
            #                     num_T=15,num_S=15,
            #                     hid_channels=32,
            #                     dropout=0.5)
            # BACKBONE = EEGConformer(n_chans=32,n_outputs=EMBEDDING_SIZE,n_times=128,final_fc_length='auto')
            BACKBONE = EEGConformer(n_chans=62,n_outputs=EMBEDDING_SIZE,n_times=200,final_fc_length='auto')

            print("=" * 60)
            print(BACKBONE)
            print("{} Backbone Generated".format(BACKBONE_NAME))
            print("=" * 60)
            

            HEAD =  Softmax(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)
            # HEAD =  ArcFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)
            # HEAD =  CosFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)
            # HEAD =  SphereFace(in_features = EMBEDDING_SIZE, out_features = NUM_CLASS, device_id = GPU_ID)
            print("=" * 60)
            print(HEAD)
            print("{} Head Generated".format(HEAD_NAME))
            print("=" * 60)

            LOSS = nn.CrossEntropyLoss()

            # OPTIMIZER=torch.optim.SGD(BACKBONE.parameters(), lr = LR) 
            # OPTIMIZER=torch.optim.Adam(BACKBONE.parameters(), lr = LR) 
            OPTIMIZER=torch.optim.Adam(BACKBONE.parameters(), lr = 0.0002, betas=(0.5,0.999)) 
            print("=" * 60)
            print(OPTIMIZER)
            print("Optimizer Generated")
            print("=" * 60)

            #======= init & resume & MULTI_GPU=======#

            if MULTI_GPU:
                # multi-GPU setting
                BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
                BACKBONE = BACKBONE.to(DEVICE)
            else:
                # single-GPU setting
                BACKBONE = BACKBONE.to(DEVICE)

            #======= train & validation & save checkpoint =======#
            DISP_FREQ = 50 # frequency to display training loss & acc
            VER_FREQ = 2000
            batch = 0  # batch index

            losses = AverageMeter()
            top1 = AverageMeter()
            theta_yis = AverageMeter()
            theta_js = AverageMeter()


            BACKBONE.train()  # set to training mode
            HEAD.train()
            for epoch in range(NUM_EPOCH):
                
                if epoch in STAGES:
                    schedule_lr(OPTIMIZER)

                last_time = time.time()

                for inputs, labels in iter(trainloader):
                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE).long()
                    features = BACKBONE(inputs)
                    outputs = HEAD(features, labels)
                    loss = LOSS(outputs, labels)


                    # outputs = BACKBONE(inputs)
                    # outputs, theta_yi, theta_j= HEAD(features, labels)


                    # loss1 = LOSS(outputs, labels) - torch.sigmoid(grad_cam_loss_v1(BACKBONE, inputs, labels, mode = 'ALL', weight_nroi=0, layer = 'pp'))
                    # loss1 = LOSS(outputs, labels) - torch.sigmoid(grad_cam_loss_v1(BACKBONE, inputs, labels, mode = 'NEW_ALL_FRONT', weight_nroi=0, layer = 'BN_s'))
                    # loss1 = LOSS(outputs, labels) - torch.sigmoid(grad_cam_loss_v1(BACKBONE, inputs, labels, mode = 'FRONT_LEFT', weight_nroi=0, layer = 'block3'))

                    
                    # loss2 =  grad_cam_loss_v2(BACKBONE,HEAD, inputs, labels, mode = 'ALL', layer = 'pp')
                    # loss2 =  grad_cam_loss_v2(BACKBONE, HEAD, inputs, labels, mode = 'NEW_ALL_FRONT', layer = 'BN_s')
                    # loss2 =  grad_cam_loss_v2(BACKBONE, HEAD, inputs, labels, mode = 'FRONT_LEFT', layer = 'block3')

                    # loss = loss1
                    # loss = loss2
                    
                    # prec1 = train_accuracy(outputs.data, labels, topk = (1,))
                    prec1 = train_accuracy(features.data, labels, topk = (1,))

                    losses.update(loss.data.item(), inputs.size(0))
                    top1.update(prec1.data.item(), inputs.size(0))
                    # theta_yis.update(theta_yi.data.item(), inputs.size(0))
                    # theta_js.update(theta_j.data.item(), inputs.size(0))

                    # compute gradient and do SGD step
                    OPTIMIZER.zero_grad()
                    loss.backward()
                    OPTIMIZER.step()
                    
                    batch += 1 # batch index //**************
                    # print(batch)
                    
                    # dispaly training loss & acc every DISP_FREQ (buffer for visualization)
                    if ((batch + 1) % DISP_FREQ == 0) and batch != 0:
                        batch_time = time.time() - last_time
                        last_time = time.time()

                        print('Epoch {} Batch {}\t'
                            # 'Speed: {speed:.2f} samples/s\t'
                            'Training Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Training Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            # 'theta_yi {theta_yi.val:.4f} ({theta_yi.avg:.4f})\t'
                            # 'theta_j {theta_j.val:.4f} ({theta_j.avg:.4f})\t'
                            .format(
                                epoch + 1, batch + 1, 
                                # speed=inputs.size(0) * DISP_FREQ / float(batch_time),
                                loss=losses, 
                                top1=top1,
                                # theta_yi = theta_yis, theta_j = theta_js, 
                            ))
                        losses = AverageMeter()
                        top1 = AverageMeter()
                        # theta_yis = AverageMeter()
                        # theta_js = AverageMeter()
                

                # if ((batch + 1) % VER_FREQ == 0) and batch != 0: #perform validation & save checkpoints (buffer for visualization)
                val_acc, val_loss=valid(validloader, BACKBONE, HEAD, nn.CrossEntropyLoss())
                if val_acc > higest_acc:
                    higest_acc = val_acc
                    
                BACKBONE.train()  # set to training mode
                
                

                # for params in OPTIMIZER.param_groups:
                #     lr = params['lr']
                #     break
                # print("Learning rate %f"%lr)
                # print("Perform Evaluation on", TARGET, ", and Save Checkpoints...")
                # acc = []
                # for ver in vers:
                #     name, data_set, issame = ver
                #     accuracy, std, xnorm, best_threshold, roc_curve = perform_val(MULTI_GPU, DEVICE, EMBEDDING_SIZE, BATCH_SIZE, BACKBONE, data_set, issame)
                #     # buffer_val(writer, name, accuracy, std, xnorm, best_threshold, roc_curve, batch + 1)
                #     print('[%s][%d]XNorm: %1.5f' % (name, batch+1, xnorm))
                #     print('[%s][%d]Accuracy-Flip: %1.5f+-%1.5f' % (name, batch+1, accuracy, std))
                #     print('[%s][%d]Best-Threshold: %1.5f' % (name, batch+1, best_threshold))
                #     acc.append(accuracy)

                # save checkpoints per epoch
                # if need_save(acc, highest_acc):

                # if MULTI_GPU:
                # torch.save(BACKBONE.module.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                # torch.save(HEAD.state_dict(), os.path.join(WORK_PATH, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch + 1, get_time())))
                # else:
                #     torch.save(BACKBONE.state_dict(), os.path.join(WORK_PATH, "Backbone_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(BACKBONE_NAME, epoch + 1, batch + 1, get_time())))
                #     torch.save(HEAD.state_dict(), os.path.join(WORK_PATH, "Head_{}_Epoch_{}_Batch_{}_Time_{}_checkpoint.pth".format(HEAD_NAME, epoch + 1, batch + 1, get_time())))
                
                
            current_time = datetime.now()
            print(current_time)
            # count += 1
            # total_sum_higest += higest_acc
            print(higest_acc)
            # total_sum_normal += val_acc
        # print('%.5f' % (total_sum_higest / count))  # 取1位小数
        # print('%.5f' % (total_sum_normal / count))  # 取1位小数

            # test(allloader, BACKBONE, HEAD)
            # test(trainloader, BACKBONE, HEAD)
                

