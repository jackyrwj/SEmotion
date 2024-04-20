
# standard
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import copy
import random
import time
import math
# from torch.utils.tensorboard import SummaryWriter

#
import utils
import models
import pretty_errors
import logging
from torch.autograd import Variable
# random seed


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# setup_seed(20)
setup_seed(3407)





# writer = SummaryWriter()
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")



epsilons = [0, .05, .1, .15, .2, .25, .3]

def fgsm_attack(image, epsilon, data_grad):
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image




class MSMDAER():
    def __init__(self, model=models.MSMDAERNet(), source_loaders=0, target_loader=0, batch_size=64, iteration=10000, lr=0.001, momentum=0.9, log_interval=10):
        self.model = model
        self.model.to(device)
        self.source_loaders = source_loaders
        self.target_loader = target_loader
        self.batch_size = batch_size
        self.iteration = iteration
        self.lr = lr
        self.momentum = momentum
        self.log_interval = log_interval

    def __getModel__(self):
        return self.model
    


    def deepfool(self, image, net, num_classes=4, overshoot=0.002, max_iter=100):
        is_cuda = torch.cuda.is_available()
        if is_cuda:
            # print("Using GPU")
            # image = image.cuda()
            # net = net.cuda()
            image = image.to(device)
            net = net.to(device)
            
        f_image = net.forward(Variable(image[None, :], requires_grad=True),len(self.source_loaders))

        f_image = torch.mean(torch.cat(f_image, dim=0),dim=0,keepdim=True).data.cpu().numpy().flatten()

        
        I = (np.array(f_image)).flatten().argsort()[::-1]

        I = I[0:num_classes]
        label = I[0]

        input_shape = image.cpu().numpy().shape
        pert_image = copy.deepcopy(image)
        w = np.zeros(input_shape)
        r_tot = np.zeros(input_shape)

        loop_i = 0

        x = Variable(pert_image[None, :], requires_grad=True)
        fs = net.forward(x,len(self.source_loaders))
        fs = torch.mean(torch.cat(fs, dim=0),dim=0,keepdim=True)

        k_i = label

        while k_i == label and loop_i < max_iter:

            pert = np.inf
            fs[0, I[0]].backward(retain_graph=True)
            # ----
            # for i in range(len(fs)):
            #     fs[i][0, I[0]].backward(retain_graph=True)
            grad_orig = x.grad.data.cpu().numpy().copy()

            for k in range(1, num_classes):
                if x.grad is not None:
                    x.grad.zero_()

                fs[0, I[k]].backward(retain_graph=True)
                cur_grad = x.grad.data.cpu().numpy().copy()

                # set new w_k and new f_k
                w_k = cur_grad - grad_orig
                f_k = (fs[0, I[k]] - fs[0, I[0]]).data.cpu().numpy()

                pert_k = abs(f_k)/np.linalg.norm(w_k.flatten())

                # determine which w_k to use
                if pert_k < pert:
                    pert = pert_k
                    w = w_k

            # compute r_i and r_tot
            # Added 1e-4 for numerical stability
            r_i =  (pert+1e-4) * w / np.linalg.norm(w)
            r_tot = np.float32(r_tot + r_i)

            if is_cuda:
                # pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).cuda()
                pert_image = image + (1+overshoot)*torch.from_numpy(r_tot).to(device)
            else:
                pert_image = image + (1+overshoot)*torch.from_numpy(r_tot)

            x = Variable(pert_image, requires_grad=True)

            # fs = net.forward(x)
            fs = net.forward(x,len(self.source_loaders))
            fs = torch.mean(torch.cat(fs, dim=0),dim=0,keepdim=True)


            k_i = np.argmax(fs.data.cpu().numpy().flatten())
            loop_i += 1
        r_tot = (1+overshoot)*r_tot
        return r_tot, loop_i, label, k_i, pert_image

    def train(self):
        # best_model_wts = copy.deepcopy(model.state_dict())
        source_iters = []
        for i in range(len(self.source_loaders)):
            source_iters.append(iter(self.source_loaders[i]))
        target_iter = iter(self.target_loader)
        correct = 0
        correct_attack = 0
        # correct_deep_attack = 0

        for i in range(1, self.iteration+1):
        # for i in range(1, 50):
        # for i in range(1, 2):
            self.model.train()
            # LEARNING_RATE = self.lr / math.pow((1 + 10 * (i - 1) / (self.iteration)), 0.75)
            LEARNING_RATE = self.lr
            # if (i - 1) % 100 == 0:
                # print("Learning rate: ", LEARNING_RATE)
            # optimizer = torch.optim.SGD(self.model.parameters(), lr=LEARNING_RATE, momentum=self.momentum)
            optimizer = torch.optim.Adam(
                self.model.parameters(), lr=LEARNING_RATE)

            for j in range(len(source_iters)):
                try:
                    source_data, source_label = next(source_iters[j])
                except Exception as err:
                    source_iters[j] = iter(self.source_loaders[j])
                    source_data, source_label = next(source_iters[j])
                try:
                    target_data, _ = next(target_iter)
                except Exception as err:
                    target_iter = iter(self.target_loader)
                    target_data, _ = next(target_iter)
                source_data, source_label = source_data.to(
                    device), source_label.to(device)
                target_data = target_data.to(device)


                optimizer.zero_grad()
                # cls_loss, mmd_loss, l1_loss = self.model(source_data, 
                cls_loss, mmd_loss, l1_loss, intra_inter_loss = self.model(source_data, 
                # mmd_loss, l1_loss, intra_inter_loss = self.model(source_data, 
                                                         number_of_source=len(source_iters), 
                                                         data_tgt=target_data, 
                                                         label_src=source_label, 
                                                         mark=j)
                gamma = 2 / (1 + math.exp(-10 * (i) / (self.iteration))) - 1
                beta = gamma/100

                # loss = cls_loss + gamma * mmd_loss + beta * l1_loss
                loss = cls_loss + gamma * mmd_loss + beta * l1_loss + intra_inter_loss
                # loss =  gamma * mmd_loss + beta * l1_loss + intra_inter_loss
                loss.backward()
                optimizer.step()
                if i % log_interval == 0:
                    print('Train source' + str(j) + ', iter: {} [({:.0f}%)]\t' \
                        #   'Loss: {:.6f}\t' \
                            'soft_loss: {:.6f}\tmmd_loss {:.6f}\tl1_loss: {:.6f}'
                          .format(
                        i, 100.*i/self.iteration, 
                        loss.item(), 
                        # cls_loss.item(), 
                        mmd_loss.item(), 
                        l1_loss.item()
                    ))

            if i % (log_interval * 20) == 0:
                t_correct = self.test(i)
                if t_correct > correct:
                    correct = t_correct
                
                # t_correct_attack = self.test_attack(i)
                # if t_correct_attack > correct_attack:
                #     correct_attack = t_correct_attack
                    
                
                # t_correct_attack = self.test_DEEPFOOL(i)
                # if t_correct_attack > correct_attack:
                #     correct_attack = t_correct_attack

        # return 100. * correct / len(self.target_loader.dataset)
        return 100. * correct / len(self.target_loader.dataset), 100. * correct_attack / len(self.target_loader.dataset)
    def test(self, i):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []
        for i in range(len(self.source_loaders)):
            corrects.append(0)
        with torch.no_grad():
            for data, target in self.target_loader:
                data = data.to(device)
                target = target.to(device)
                preds = self.model(data, len(self.source_loaders))
                for i in range(len(preds)):
                    preds[i] = F.softmax(preds[i], dim=1)
                pred = sum(preds)/len(preds)
                test_loss += F.nll_loss(F.log_softmax(pred,dim=1), target.squeeze()).item()
                pred = pred.data.max(1)[1]
                correct += pred.eq(target.data.squeeze()).cpu().sum()
                for j in range(len(self.source_loaders)):
                    pred = preds[j].data.max(1)[1]
                    corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()

            test_loss /= len(self.target_loader.dataset)
            # writer.add_scalar("Test/Test loss", test_loss, i)

            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #     test_loss, correct, len(self.target_loader.dataset),

            #     100. * correct / len(self.target_loader.dataset)
            # ))
            # for n in range(len(corrects)):
            #     print('Source' + str(n) + 'accnum {}'.format(corrects[n]))
        return correct
    

    def test_DEEPFOOL(self, i):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []
        for i in range(len(self.source_loaders)):
            corrects.append(0)
        
        # adver_example_by_FOOL = torch.zeros((batch_size,1,28,28)).to(device)
        # for data, target in self.target_loader:
        for i,(data,target) in enumerate(self.target_loader):
            data = data.to(device)
            target = target.to(device)
            cur_adver_example_by_FOOL = torch.zeros_like(data).to(device)

            for j in range(batch_size):
                if dataset_name == 'seed3':
                    r_rot,loop_i,label,k_i,pert_image = self.deepfool(data[j],self.model, 3)
                else:
                    r_rot,loop_i,label,k_i,pert_image = self.deepfool(data[j],self.model, 4)
                cur_adver_example_by_FOOL[j] = pert_image

            preds = self.model(cur_adver_example_by_FOOL, len(self.source_loaders))

            for i in range(len(preds)):
                preds[i] = F.softmax(preds[i], dim=1)
            pred = sum(preds)/len(preds)

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.squeeze()).cpu().sum()
            for j in range(len(self.source_loaders)):
                pred = preds[j].data.max(1)[1]
                corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()

        return correct

    def test_attack(self, i):
        self.model.eval()
        test_loss = 0
        correct = 0
        corrects = []
        for i in range(len(self.source_loaders)):
            corrects.append(0)
        # with torch.no_grad():
        for data, target in self.target_loader:
            data = data.to(device)
            target = target.to(device)

            data.requires_grad = True

            preds = self.model(data, len(self.source_loaders))
            for i in range(len(preds)):
                preds[i] = F.softmax(preds[i], dim=1)
            pred = sum(preds)/len(preds)

            # init_pred = pred.max(1, keepdim=True)[1]  # get the index of the max log-probability
            # if init_pred.item() != target.item():
            # if init_pred != target:
                # continue
            
            # test_loss = F.nll_loss(F.log_softmax(pred,dim=1), target.squeeze()).item()
            test_loss = F.nll_loss(F.log_softmax(pred,dim=1), target.squeeze())

            self.model.zero_grad()

            test_loss.backward()

            # if i == self.iteration and j == (len(source_iters) - 1):

            data_grad = data.grad.data
            # perturbed_data = fgsm_attack(data, epsilon, data_grad)
            perturbed_data = fgsm_attack(data, args.epsilon, data_grad)
            
            preds = self.model(perturbed_data, len(self.source_loaders))
            # ------------------------------------------

            for i in range(len(preds)):
                preds[i] = F.softmax(preds[i], dim=1)
            pred = sum(preds)/len(preds)
            # attack_loss += F.nll_loss(F.log_softmax(pred,dim=1), target.squeeze()).item()

            pred = pred.data.max(1)[1]
            correct += pred.eq(target.data.squeeze()).cpu().sum()
            for j in range(len(self.source_loaders)):
                pred = preds[j].data.max(1)[1]
                corrects[j] += pred.eq(target.data.squeeze()).cpu().sum()
            # with torch.no_grad()缩进
            # attack_loss /= len(self.target_loader.dataset)
        return correct
    



    
    

def cross_subject(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum, log_interval):
    # # one_session_data, one_session_label = copy.deepcopy(data_tmp[session_id]), copy.deepcopy(label[session_id])
    # # target_data, target_label = one_session_data.pop(), one_session_label.pop()
    # # source_data, source_label = copy.deepcopy(one_session_data[0:source_number]), copy.deepcopy(one_session_label[0:source_number])
    # # print("Source number: ", len(source_data))
    
    # # LOSO
    # # print(len(data))
    # # print(len(data[session_id]))
    # one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    # train_idxs = list(range(15))
    # del train_idxs[subject_id]
    # test_idx = subject_id
    # target_data, target_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(one_session_label[test_idx])
    # source_data, source_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(one_session_label[train_idxs])
    # # print('Target_subject_id: ', test_idx)
    # # print('Source_subject_id: ', train_idxs)

    one_session_data, one_session_label = copy.deepcopy(data[session_id]), copy.deepcopy(label[session_id])
    # one_session_data, one_session_label = copy.deepcopy(
    #     data), copy.deepcopy(label)
    train_idxs = list(range(15))
    del train_idxs[subject_id]
    test_idx = subject_id
    # target_data, target_label = one_session_data[test_idx,:], one_session_label[test_idx, :]
    # source_data, source_label = copy.deepcopy(one_session_data[train_idxs, :]), copy.deepcopy(one_session_label[train_idxs, :])

    target_data, target_label = copy.deepcopy(one_session_data[test_idx]), copy.deepcopy(one_session_label[test_idx])
    source_data, source_label = copy.deepcopy(one_session_data[train_idxs]), copy.deepcopy(one_session_label[train_idxs])


    del one_session_label
    del one_session_data

    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = MSMDAER(
        model=models.MSMDAERNet(
                    pretrained=False, 
                    number_of_source=len(source_loaders), 
                    number_of_category=category_number,
                    
                    # in_features = 3, 
                    # out_features= 3, 
                    in_features = EMBEDDING, 
                    out_features= NUM_CLASS, 
                    s = args.param_s, 
                    k = args.param_k, 
                    a = args.param_a, 
                    b = args.param_b,
                    device = device),

                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval,)
    # print(model.__getModel__())
    # acc = model.train()
    # acc, acc_attack = model.train()
    acc, acc_attack = model.train()
    print('Target_subject_id: {}, current_session_id: {}, acc: {}'.format(test_idx, session_id, acc))
    print('Target_subject_id: {}, current_session_id: {}, acc_attack: {}'.format(test_idx, session_id, acc_attack))
    logging.info('Target_subject_id: {}, current_session_id: {}, acc: {}'.format(test_idx, session_id, acc))
    logging.info('Target_subject_id: {}, current_session_id: {}, acc_attack: {}'.format(test_idx, session_id, acc_attack))
    return acc,acc_attack

# def cross_session(data, label, session_id, subject_id, category_number, batch_size, iteration, lr, momentum, log_interval):
def cross_session(data, label, subject_id, category_number, batch_size, iteration, lr, momentum, log_interval):
    # # target_data, target_label = copy.deepcopy(data[2][subject_id]), copy.deepcopy(label[2][subject_id])
    # # source_data, source_label = [copy.deepcopy(data[0][subject_id]), copy.deepcopy(data[1][subject_id])], [copy.deepcopy(label[0][subject_id]), copy.deepcopy(label[1][subject_id])]

    # ## LOSO
    # train_idxs = list(range(3))
    # del train_idxs[session_id]
    # test_idx = session_id
    
    # target_data, target_label = copy.deepcopy(data[test_idx][subject_id]), copy.deepcopy(label[test_idx][subject_id])
    # source_data, source_label = copy.deepcopy(data[train_idxs][:, subject_id]), copy.deepcopy(label[train_idxs][:, subject_id])

    # no LOSO
    target_data, target_label = copy.deepcopy(data_tmp[2][subject_id]), copy.deepcopy(label[2][subject_id])
    source_data, source_label = [copy.deepcopy(data_tmp[0][subject_id]), copy.deepcopy(data_tmp[1][subject_id])], [copy.deepcopy(label[0][subject_id]), copy.deepcopy(label[1][subject_id])]
    # one_sub_data, one_sub_label = data[i], label[i]
    # target_data, target_label = one_session_data.pop(), one_session_label.pop()
    # source_data, source_label = one_session_data.copy(), one_session_label.copy()
    # print(len(source_data))
    test_idx = 2

    source_loaders = []
    for j in range(len(source_data)):
        source_loaders.append(torch.utils.data.DataLoader(dataset=utils.CustomDataset(source_data[j], source_label[j]),
                                                          batch_size=batch_size,
                                                          shuffle=True,
                                                          drop_last=True))
    target_loader = torch.utils.data.DataLoader(dataset=utils.CustomDataset(target_data, target_label),
                                                batch_size=batch_size,
                                                shuffle=True,
                                                drop_last=True)
    model = MSMDAER(
                model=models.MSMDAERNet(
                    pretrained=False, 
                    number_of_source=len(source_loaders), 
                    number_of_category=category_number,
                    in_features = EMBEDDING, 
                    out_features= NUM_CLASS, 
                    s = args.param_s, 
                    k = args.param_k, 
                    a = args.param_a, 
                    b = args.param_b,
                    device = device),

                    source_loaders=source_loaders,
                    target_loader=target_loader,
                    batch_size=batch_size,
                    iteration=iteration,
                    lr=lr,
                    momentum=momentum,
                    log_interval=log_interval)
    # print(model.__getModel__())
    # acc = model.train()
    acc, acc_attack = model.train()
    print('Target_session_id: {}, current_subject_id: {}, acc: {}'.format(test_idx, subject_id, acc))
    print('Target_session_id: {}, current_subject_id: {}, acc_attack: {}'.format(test_idx, subject_id, acc_attack))
    logging.info('Target_session_id: {}, current_subject_id: {}, acc: {}'.format(test_idx, subject_id, acc))
    logging.info('Target_session_id: {}, current_subject_id: {}, acc_attack: {}'.format(test_idx, subject_id, acc_attack))


    return acc,acc_attack


if __name__ == '__main__':
    
   
    parser = argparse.ArgumentParser(description='MS-MDAER parameters')

    parser.add_argument(
        # '--dataset', type=str, default='seed3',
        '--dataset', type=str, default='seed4',
                        help='the dataset used for MS-MDAER, "seed3" or "seed4"')
    
    parser.add_argument('--norm_type', type=str, default='ele',
                        help='the normalization type used for data, "ele", "sample", "global" or "none"')
    
    parser.add_argument(
        # '--batch_size', type=int, default=16,
        # '--batch_size', type=int, default=32,
        # '--batch_size', type=int, default=64,
        # '--batch_size', type=int, default=128,
        '--batch_size', type=int, default=256,
                        help='size for one batch, integer')
    
    parser.add_argument(
        # '--epsilon', type=int, default=0,
        # '--epsilon', type=int, default=.01,
        '--epsilon', type=int, default=.02,
        # '--epsilon', type=int, default=.05,
        # '--epsilon', type=int, default=.1,
        # '--epsilon', type=int, default=.15,
        # '--epsilon', type=int, default=.2,
        # '--epsilon', type=int, default=.25,
        # '--epsilon', type=int, default=.3,
                        help='training epoch, integer')
    
    parser.add_argument(
        '--epoch', type=int, default=200,
        # '--epoch', type=int, default=150,
        # '--epoch', type=int, default=100,
                        help='training epoch, integer')
    
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')

    
    parser.add_argument('--param_s', default=64.0, type=float)
    parser.add_argument('--param_k', default=80.0, type=float)
    



    
    # args = parser.parse_args()
    args = parser.parse_args(args=[])
    dataset_name = args.dataset
    bn = args.norm_type

    # data preparation
    print('Model name: MS-MDAER. Dataset name: ', dataset_name)
    data, label = utils.load_data(dataset_name)
    print('Normalization type: ', bn)
    if bn == 'ele':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminy(data_tmp[i][j])
    elif bn == 'sample':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.norminx(data_tmp[i][j])
    elif bn == 'global':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
        for i in range(len(data_tmp)):
            for j in range(len(data_tmp[0])):
                data_tmp[i][j] = utils.normalization(data_tmp[i][j])
    elif bn == 'none':
        data_tmp = copy.deepcopy(data)
        label_tmp = copy.deepcopy(label)
    else:
        pass
    trial_total, category_number, _ = utils.get_number_of_label_n_trial(
        dataset_name)

    # training settings
    batch_size = args.batch_size
    epoch = args.epoch
    lr = args.lr
    print('BS: {}, epoch: {}'.format(batch_size, epoch))
    momentum = 0.9
    log_interval = 10
    iteration = 0
    if dataset_name == 'seed3':
        iteration = math.ceil(epoch*3394/batch_size)
        EMBEDDING = int(3)
        NUM_CLASS = int(3)
    elif dataset_name == 'seed4':
        iteration = math.ceil(epoch*820/batch_size)
        EMBEDDING = int(4)  
        NUM_CLASS = int(4)
    else:
        iteration = 5000
    print('Iteration: {}'.format(iteration))

    logging.basicConfig(filename='outputs/output.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # store the results
    csub = []
    csub_attack = []
    csub_deep_attack = []
    csesn = []
    csesn_attack = []
    csesn_deep_attack = []
 
    # # cross-validation, LOSO
    # # cross-subject 
    # for session_id_main in range(3):
    # # for session_id_main in range(2,3):
    # # for session_id_main in range(1,3):
    #     for subject_id_main in range(15):
    #     # for subject_id_main in range(6,15):
    #         csub.append(cross_subject(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
    #                                 batch_size, iteration, lr, momentum, log_interval))
    # print("Cross-subject: ", csub)
    # print("Cross-subject mean: ", np.mean(csub), "std: ", np.std(csub))
    # logging.info("Cross-subject: %s" % csub)
    # logging.info("Cross-subject mean: %s, std: %s" % (np.mean(csub),  np.std(csub)))

    # cross-session                             
    # for subject_id_main in range(15):
    #     for session_id_main in range(3):
    # # for subject_id_main in range(1):
    # #     for session_id_main in range(0):
    #         csesn.append(cross_session(data_tmp, label_tmp, session_id_main, subject_id_main, category_number,
    #                                 batch_size, iteration, lr, momentum, log_interval))
    # print("Cross-session: ", csesn)
    # print("Cross-session mean: ", np.mean(csesn), "std: ", np.std(csesn))
    # logging.info("Cross-session: %s" % csesn)
    # logging.info("Cross-session mean: %s, std: %s" % (np.mean(csesn),  np.std(csesn)))

    # -------------------------------------------------------------------------------------
    # no LOSO 14/9
#    cross-subject, for 3 sessions, 1-14 as sources, 15 as target
    # for i in range(3):
    #     # for subject_id in range(15):
    #     subject_id = 14
    #     # csub.append(cross_subject(data_tmp, label_tmp, i, subject_id, category_number,
    #     #                             batch_size, iteration, lr, momentum, log_interval))
    #     csub_re, csub_attack_re = cross_subject(data_tmp, label_tmp, i, subject_id, category_number,
    #                                 batch_size, iteration, lr, momentum, log_interval)
    #     csub.append(csub_re)
    #     csub_attack.append(csub_attack_re)
    # print("Cross-subject: ", csub)
    # print("Cross-subject mean: ", np.mean(csub), "std: ", np.std(csub))
    # print("Cross-subject attack mean: ", np.mean(csub_attack), "std: ", np.std(csub_attack))

    # logging.info("Cross-subject: %s" % csub)
    # logging.info("Cross-subject mean: %s, std: %s" % (np.mean(csub),  np.std(csub)))
    # logging.info("Cross-subject attack mean: %s, std: %s" % (np.mean(csub_attack),  np.std(csub_attack)))


    # cross-session, for 15 subjects, 1-2 as sources, 3 as target
    for i in range(15):
        # csesn.append(cross_session(data_tmp, label_tmp, i, category_number, batch_size, iteration, lr, momentum, log_interval))
        csesn_re, csesn_attack_re = cross_session(data_tmp, label_tmp, i, category_number, batch_size, iteration, lr, momentum, log_interval)
        csesn.append(csesn_re)
        csesn_attack.append(csesn_attack_re)
    print("Cross-session: ", csesn)
    print("Cross-session mean: ", np.mean(csesn), "std: ", np.std(csesn))
    print("Cross-session attack mean: ", np.mean(csesn_attack), "std: ", np.std(csesn_attack))
    logging.info("Cross-session: %s" % csesn)
    logging.info("Cross-session mean: %s, std: %s" % (np.mean(csesn),  np.std(csesn)))
    logging.info("Cross-session attack mean: %s, std: %s" % (np.mean(csesn_attack),  np.std(csesn_attack)))

    
    
    logging.shutdown()
