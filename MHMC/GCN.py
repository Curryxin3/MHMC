# coding = utf-8
import torch
import torch.nn as nn
import math
from utils import *
from torch.nn import Parameter
import scipy.io as sio

import torch.nn.functional as F
import numpy as np

class GCNNet(nn.Module):

    def __init__(self,opt, adj, num_classes, t=None, adj_file=None, label_word=None,in_channel=None):
        super(GCNNet, self).__init__()
        #self.H = H

        self.inp = sio.loadmat(label_word)['labelVector']   #标签的词向量,63*300
        self.inp = torch.from_numpy(self.inp).cuda(opt.device)
        self.gc1 = GraphConvolution(in_channel, 406)
        self.gc2 = GraphConvolution(406, 512)
        self.rulu = nn.LeakyReLU(0.2)

        _adj = gen_A(num_classes, t, adj_file)   #生成邻接矩阵A
        _adj = torch.from_numpy(_adj).float()
        # _adj = sio.loadmat(adj)['adj_less']   #标签的词向量,63*300
        # _adj = torch.from_numpy(_adj).float()
        self.A = Parameter(_adj)
        self.classfier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 63)
        )

    def forward(self, feature):    #feature是网络的输入，inp是词嵌入
        #feature = self.H(feature)

        adj = gen_adj(self.A).detach()   #detach是保持值不变，只训练一部分，adj是经过规范化后的A
        x = self.gc1(self.inp, adj)
        x = self.rulu(x)
        x = self.gc2(x, adj)
        # x = self.rulu(x)
        # x = self.gc3(x, adj)
        x = x.transpose(0, 1)
        y_hat = torch.matmul(feature, x)
        # y_hat = self.classfier(feature)


        return y_hat


class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias = False):
        super(GraphConvolution,self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data_uniform(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output