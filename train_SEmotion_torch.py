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


from torcheeg.model_selection import LeaveOneSubjectOut,KFold,
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
            outputs = pred

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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')

    parser.add_argument('--workers_id', help="gpu ids or cpu", default='7', type=str)
    parser.add_argument('--epochs', help="training epochs", default=300, type=int)

    parser.add_argument('--stages', help="training stages", default='200,250', type=str)
    parser.add_argument('--lr',help='learning rate',default=1e-2, type=float)
    parser.add_argument('--batch_size', help="batch_size", default=64, type=int)
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
    print("=" * 60)

    torch.backends.cudnn.benchmark = True
    NUM_CLASS = 2

    dataset = DEAPDataset(
        io_path=f'./tmp_out/examples_fbccnn/deap',
        root_path='./tmp_in/data_preprocessed_python',
        offline_transform=transforms.Compose([
            transforms.BandDifferentialEntropy(apply_to_baseline=True),
            transforms.BaselineRemoval(), 
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





    k_fold = KFold(n_splits=5, split_path=f'./tmp_out/examples_fbccnn/kfold', shuffle=True)
    
    
        
    total_sum_higest = 0
    total_sum_normal = 0
    higest_acc = 0
    count = 0
    for i, (train_dataset, val_dataset) in enumerate(k_fold.split(dataset)):
        if i == 0:
            print(f"*****************fold: {i}*****************")
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            validloader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
            


            #======= model & loss & optimizer =======#
            BACKBONE = FBCCNN(num_classes=EMBEDDING_SIZE, in_channels=4, grid_size=(9, 9))
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

            ## good
            OPTIMIZER=torch.optim.Adam(BACKBONE.parameters(), lr = 0.0002, betas=(0.5,0.999)) 
            print("=" * 60)
            print(OPTIMIZER)
            print("Optimizer Generated")
            print("=" * 60)

            array = np.load('./results/FBCCNN_classical/classical.npy')


            top_percentage = 0.1
            top_10_percent = int(len(array) * top_percentage)
            indices = np.argsort(array)[:top_10_percent]
            classical_dataset = CustomDataset(dataset, indices)
            classicalloader = torch.utils.data.DataLoader(classical_dataset, 
                                                          batch_size=BATCH_SIZE, 
                                                          shuffle=False, 
                                                          num_workers=len(GPU_ID), 
                                                          drop_last=True)
            

            inputs_array = []
            for inputs, labels in iter(classicalloader):
                inputs_array.append(inputs)

            inputs_tensor = torch.cat(inputs_array, dim=0)
            reshaped_samples = inputs_tensor.reshape(-1, np.prod(inputs_tensor.shape[1:]))
            n_clusters = 1
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(reshaped_samples)
            cluster_centers = kmeans.cluster_centers_
            center = torch.from_numpy(cluster_centers).reshape(n_clusters, *inputs_tensor.shape[1:])

            print("Center shape:", center.shape)
            HEAD.setKnoWeight(center)

--------------------------------------------
            
            #======= init & resume & MULTI_GPU=======#
            if MULTI_GPU:
                # multi-GPU setting
                BACKBONE = nn.DataParallel(BACKBONE, device_ids = GPU_ID)
                BACKBONE = BACKBONE.to(DEVICE)
            else:
                # single-GPU setting
                BACKBONE = BACKBONE.to(DEVICE)

            #======= train & validation & save checkpoint =======#
            DISP_FREQ = 100 # frequency to display training loss & acc
            batch = 0  # batch index

            losses = AverageMeter()
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

                    inputs = inputs.to(DEVICE)
                    labels = labels.to(DEVICE).long()
                    features = BACKBONE(inputs)
                    outputs = features
                    
                    outputs, loss, intra_loss, inter_loss, WyiX, WjX= HEAD(features, labels)
                    catch_array, loss, counter =  grad_cam_loss_v2(BACKBONE, HEAD, inputs, labels, mode = 'ALL_FRONT', layer = 'block2', array=catch_array, counter = counter)
                    prec1 = train_accuracy(outputs.data, labels, topk=(1,))
                    losses.update(loss.data.item(), inputs.size(0))
                    top1.update(prec1.data.item(), inputs.size(0))

                    OPTIMIZER.zero_grad()
                    loss.backward()
                    OPTIMIZER.step()
                    batch += 1  # batch index

                    if ((batch + 1) % DISP_FREQ == 0) and batch != 0:

                        batch_time = time.time() - last_time
                        last_time = time.time()

                        print('Epoch {} Batch {}\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                            epoch + 1, batch + 1, 
                            top1=top1)
                            )
                            
                        losses = AverageMeter()
                        top1 = AverageMeter()


                val_acc, val_loss= valid(validloader, BACKBONE, HEAD, nn.CrossEntropyLoss())
                if val_acc > higest_acc:
                    print('find hightest')
                    higest_acc = val_acc

                BACKBONE.train()  # set to training mode
            
                
            current_time = datetime.now()
            print(current_time)
            print(higest_acc)


