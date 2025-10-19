'''
Description: 
Author: voicebeer
Date: 2020-09-09 00:06:57
LastEditTime: 2021-03-25 03:27:41
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.nn import Parameter
import math
import pretty_errors

import utils


class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            # nn.BatchNorm1d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            # nn.BatchNorm1d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            # nn.BatchNorm1d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


def pretrained_CFE(pretrained=False):
    model = CFE()
    if pretrained:
        pass
    return model


class pre_trained_MLP(nn.Module):
    def __init__(self):
        super(pre_trained_MLP, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(310, 256),
            nn.BatchNorm1d(256, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.module(x)
        return x


class DSFE(nn.Module):
    def __init__(self):
        super(DSFE, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(64, 32),
            # nn.ReLU(inplace=True),
            nn.BatchNorm1d(32, eps=1e-05, momentum=0.1,
                           affine=True, track_running_stats=True),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            # nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, x):
        x = self.module(x)
        return x


class MSMDAERNet_tsne(nn.Module):
    def __init__(self, pretrained=False, number_of_source=14, number_of_category=4):
        super(MSMDAERNet_tsne, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # for i in range(1, number_of_source):
        #     exec('self.DSFE' + str(i) + '=DSFE()')
        #     exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
            之所以target data每一条线都要过一遍是因为要计算discrepency loss, mmd和cls都只要mark-th那条线就行
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type} 
        '''
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor
            # to extract the domain specific feature of target data
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            # Use the specific feature extractor
            # to extract the source data, and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
            # discrepency loss
            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))
            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(
                pred_src, dim=1), label_src.squeeze())

            return cls_loss, mmd_loss, disc_loss

        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            feature_DSFE = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                feature_DSFE.append(feature_DSFE_i)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred, feature_DSFE


class MSMDAERNet(nn.Module):
    def __init__(self, 
                 pretrained=False, 
                 number_of_source=15, 
                 number_of_category=4,

                 in_features = 3, 
                 out_features= 3, 
                 s = 64.0, 
                 k = 80.0, 
                 a = 0.80, 
                 b = 1.23,
                 device = None,
                 ):
        super(MSMDAERNet, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # for i in range(1, number_of_source):
        #     exec('self.DSFE' + str(i) + '=DSFE()')
        #     exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')
        
        
        # SEmotion----------------------------
        # self.in_features = in_features
        # self.out_features = out_features
        # self.s = s
        # self.k = k
        # self.a = a
        # self.b = b
        # self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # self.device = device

        # Softmax----------------------------
        # self.in_features = in_features
        # self.out_features = out_features
        # self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        # self.bias = Parameter(torch.FloatTensor(out_features))
        # nn.init.xavier_uniform_(self.weight)
        # nn.init.zeros_(self.bias)
        # self.LOSS = nn.CrossEntropyLoss()
        # self.device = device
            
        # Cosface---------------------------
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = 0.35
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.LOSS = nn.CrossEntropyLoss()
        self.device = device




    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        '''
        description: take one source data and the target data in every forward operation.
            the mmd loss is calculated between the source data and the target data (both after the DSFE)
            the discrepency loss is calculated between all the classifiers' results (test on the target data)
            the cls loss is calculated between the ground truth label and the prediction of the mark-th classifier
            之所以target data每一条线都要过一遍是因为要计算discrepency loss, mmd和cls都只要mark-th那条线就行
        param {type}:
            mark: int, the order of the current source
            data_src: take one source data each time
            number_of_source: int
            label_Src: corresponding label
            data_tgt: target data
        return {type} 
        '''
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor
            # to extract the domain specific feature of target data
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            # Use the specific feature extractor
            # to extract the source data, and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
            # discrepency loss
            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))
            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)

            # 分类损失
            pred_src = eval(DSC_name)(data_src_DSFE)

            cls_loss = F.nll_loss(F.log_softmax(pred_src, dim=1), label_src.squeeze())
            # return cls_loss, mmd_loss, disc_loss



  
            # --球面损失------------------------- cos(theta) & phi(theta) ---------------------------
            # # data_src  label_src
            # # if self.device_id == None:
            # cosine = F.linear(F.normalize(pred_src), F.normalize(self.weight))
            # # --------------------------- s*cos(theta) ---------------------------
            # output = cosine * self.s
            # # --------------------------- sface loss ---------------------------

            # one_hot = torch.zeros(cosine.size())
            # one_hot = one_hot.to(self.device)
            # one_hot.scatter_(1, label_src.view(-1, 1), 1)

            # zero_hot = torch.ones(cosine.size())
            # zero_hot = zero_hot.to(self.device)
            # zero_hot.scatter_(1, label_src.view(-1, 1), 0)


            # WyiX = torch.sum(one_hot * output, 1)
            # # with torch.no_grad():
            #     # theta_yi = torch.acos(WyiX / self.s)
            #     # sigmoid
            #     # theta_yi = torch.acos(torch.clamp(WyiX / self.s, min=-1.0, max=1.0))
            #     # weight_yi = 1.0 / (1.0 + torch.exp(-self.k * (theta_yi - self.a)))
                
            #     # steep
            #     # tmpA = theta_yi - self.a
            #     # weight_yi = torch.sign(torch.max(tmpA, torch.zeros_like(tmpA)))
            # # intra_loss = - weight_yi * WyiX
            # # Constant
            # intra_loss = - WyiX

            # Wj = zero_hot * output
            # # with torch.no_grad():
            #     # theta_j = torch.acos(Wj / self.s)
            #     # sigmoid
            #     # theta_j = torch.acos(torch.clamp(Wj / self.s, min=-1.0, max=1.0))
            #     # weight_j = 1.0 / (1.0 + torch.exp(self.k * (theta_j - self.b)))
                
            #     # steep
            #     # tmpB = self.b - theta_j
            #     # weight_j = torch.sign(torch.max(tmpB, torch.zeros_like(tmpB)))
            # # inter_loss = torch.sum(weight_j * Wj, 1)
            # # Constant
            # inter_loss = torch.sum( Wj, 1)

            # intra_inter_loss =  intra_loss.mean() + inter_loss.mean()
            
          
            # # Wyi_s = WyiX / self.s
            # # Wj_s = Wj / self.s
            # return  cls_loss, mmd_loss, disc_loss, intra_inter_loss
            # ----------------------------------------------------------------------

            # --softmax----------------------------------------------------
            # x = input
            # sub_weights = torch.chunk(self.weight, len(self.device_id), dim=0)
            # sub_biases = torch.chunk(self.bias, len(self.device_id), dim=0)
            # temp_x = x.cuda(self.device_id[0])
            # weight = sub_weights[0].cuda(self.device_id[0])
            # bias = sub_biases[0].cuda(self.device_id[0])
            # out = F.linear(temp_x, weight, bias)
            # for i in range(1, len(self.device_id)):
            #     temp_x = x.cuda(self.device_id[i])
            #     weight = sub_weights[i].cuda(self.device_id[i])
            #     bias = sub_biases[i].cuda(self.device_id[i])
            #     out = torch.cat((out, F.linear(temp_x, weight, bias).cuda(self.device_id[0])), dim=1)   
            # out = F.linear(pred_src, self.weight, self.bias)
            # softmax_loss = self.LOSS(out, label_src.squeeze())
            # return cls_loss, mmd_loss, disc_loss, softmax_loss
            # ------------------------------------------------------

            # ---cosface----------------------------------
            cosine = F.linear(F.normalize(pred_src), F.normalize(self.weight))
            phi = cosine - self.m
            # --------------------------- convert label to one-hot ---------------------------
            one_hot = torch.zeros(cosine.size())
            # if self.device_id != None:
            #     one_hot = one_hot.cuda(self.device_id[0])
            # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
            one_hot = one_hot.to(self.device)
            one_hot.scatter_(1, label_src.view(-1, 1).long(), 1)
            # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
            output *= self.s
            # cosface = self.LOSS(output, label_src.squeeze())
            cosface = F.nll_loss(F.log_softmax(output, dim=1), label_src.squeeze())
            return  cls_loss, mmd_loss, disc_loss, cosface
            # --------------------------------------------


        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred



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

class MEERNtmp(nn.Module):
    def __init__(self, pretrained=False, number_of_source=15, number_of_category=4):
        super(MEERNtmp, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # for i in range(1, number_of_source):
        #     exec('self.DSFE' + str(i) + '=DSFE()')
        #     exec('self.cls_fc_DSC' + str(i) + '=nn.Linear(32,' + str(number_of_category) + ')')
        for i in range(number_of_source):
            exec('self.DSFE' + str(i) + '=DSFE()')
            exec('self.cls_fc_DSC' + str(i) +
                 '=nn.Linear(32,' + str(number_of_category) + ')')

    def forward(self, data_src, number_of_source, data_tgt=0, label_src=0, mark=0):
        mmd_loss = 0
        disc_loss = 0
        data_tgt_DSFE = []
        if self.training == True:
            # common feature extractor
            data_src_CFE = self.sharedNet(data_src)
            data_tgt_CFE = self.sharedNet(data_tgt)
            # Each domian specific feature extractor
            # to extract the domain specific feature of target data
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                data_tgt_DSFE_i = eval(DSFE_name)(data_tgt_CFE)
                data_tgt_DSFE.append(data_tgt_DSFE_i)
            # Use the specific feature extractor
            # to extract the source data, and calculate the mmd loss
            DSFE_name = 'self.DSFE' + str(mark)
            data_src_DSFE = eval(DSFE_name)(data_src_CFE)
            # mmd_loss += utils.mmd(data_src_DSFE, data_tgt_DSFE[mark])
            mmd_loss += utils.mmd_linear(data_src_DSFE, data_tgt_DSFE[mark])
            # discrepency loss
            for i in range(len(data_tgt_DSFE)):
                if i != mark:
                    disc_loss += torch.mean(torch.abs(
                        F.softmax(data_tgt_DSFE[mark], dim=1) -
                        F.softmax(data_tgt_DSFE[i], dim=1)
                    ))
            # domain specific classifier and cls_loss
            DSC_name = 'self.cls_fc_DSC' + str(mark)
            pred_src = eval(DSC_name)(data_src_DSFE)
            cls_loss = F.nll_loss(F.log_softmax(
                pred_src, dim=1), label_src.squeeze())

            return cls_loss, mmd_loss, disc_loss, data_src_DSFE

        else:
            data_CFE = self.sharedNet(data_src)
            pred = []
            for i in range(number_of_source):
                DSFE_name = 'self.DSFE' + str(i)
                DSC_name = 'self.cls_fc_DSC' + str(i)
                feature_DSFE_i = eval(DSFE_name)(data_CFE)
                pred.append(eval(DSC_name)(feature_DSFE_i))

            return pred


class DDC(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DDC, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # self.DSFE = DSFE()
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src_feature = self.sharedNet(data_src)
        # data_src_feature = self.DSFE(data_src_feature)
        if self.training == True:
            data_tgt_feature = self.sharedNet(data_tgt)
            # data_tgt_feature = self.DSFE(data_tgt_feature)
            # loss = utils.mmd_linear(data_src_feature, data_tgt_feature)
            loss += utils.mmd_rbf_accelerate(data_src_feature,
                                             data_tgt_feature)

        data_src_cls = self.cls_fc(data_src_feature)
        return data_src_cls, loss


class DAN(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DAN, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # self.DSFE = DSFE()
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src_feature = self.sharedNet(data_src)
        # data_src_feature = self.DSFE(data_src_feature)
        if self.training == True:
            data_tgt_feature = self.sharedNet(data_tgt)
            loss += utils.mmd(data_src_feature, data_tgt_feature)

        data_src_cls = self.cls_fc(data_src_feature)
        return data_src_cls, loss


class DAN_tsne(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DAN_tsne, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # self.DSFE = DSFE()
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src_feature = self.sharedNet(data_src)
        # data_src_feature = self.DSFE(data_src_feature)
        if self.training == True:
            data_tgt_feature = self.sharedNet(data_tgt)
            loss += utils.mmd(data_src_feature, data_tgt_feature)

        data_src_cls = self.cls_fc(data_src_feature)
        return data_src_cls, loss, data_src_feature


class DeepCoraltmp(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DeepCoraltmp, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        # self.DSFE = DSFE()
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src = self.sharedNet(data_src)
        # data_Src_feature = self.DSFE(data_src_feature)
        if self.training == True:
            data_tgt = self.sharedNet(data_tgt)
            loss += utils.CORAL(data_src, data_tgt)

        data_src_cls = self.cls_fc(data_src)
        return data_src_cls, loss, data_src, data_tgt


class DeepCoral(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DeepCoral, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        self.cls_fc = nn.Linear(64, number_of_category)

    def forward(self, data_src, data_tgt=0):
        loss = 0
        data_src_feature = self.sharedNet(data_src)

        if self.training == True:
            data_tgt_feature = self.sharedNet(data_tgt)
            loss += utils.CORAL(data_src_feature, data_tgt_feature)

        data_src_cls = self.cls_fc(data_src_feature)
        return data_src_cls, loss


class DANN(nn.Module):
    def __init__(self, pretrained=False, number_of_category=4):
        super(DANN, self).__init__()
        self.sharedNet = pretrained_CFE(pretrained=pretrained)
        self.cls_fc = nn.Linear(64, number_of_category)
        self.domain_fc = AdversarialNetwork(in_feature=64)

    def forward(self, data):
        data = self.sharedNet(data)
        clabel_pred = self.cls_fc(data)
        dlabel_pred = self.domain_fc(
            AdversarialLayer(high_value=1.0).apply(data))
        return clabel_pred, dlabel_pred


class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, 32)
        self.ad_layer2 = nn.Linear(32, 32)
        self.ad_layer3 = nn.Linear(32, 1)
        self.ad_layer1.weight.data.normal_(0, 0.01)
        self.ad_layer2.weight.data.normal_(0, 0.01)
        self.ad_layer3.weight.data.normal_(0, 0.3)
        self.ad_layer1.bias.data.fill_(0.0)
        self.ad_layer2.bias.data.fill_(0.0)
        self.ad_layer3.bias.data.fill_(0.0)
        self.relu1 = nn.LeakyReLU()
        self.relu2 = nn.LeakyReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.ad_layer3(x)
        x = self.sigmoid(x)
        return x


class AdversarialLayer(torch.autograd.Function):
    def __init__(self, high_value=1.0):
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = high_value
        self.max_iter = 2000.0

    @staticmethod
    def forward(ctx, input):
        iter_num = 0
        iter_num += 1
        output = input * 1.0
        # ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, gradOutput):
        coeff = np.float(2.0 * (high - low) / (1.0 +
                         np.exp(alpha*iter_num / max_iter)) - (high - low) + low)
        return -coeff * gradOutput
