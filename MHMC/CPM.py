# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F

class CPMNet(nn.Module):

    def __init__(self, lsd_dim, hidden_dim1, hidden_dim2, out_dim, numclass):
        """
        :param lsd_dim:
        :param hidden_dim:
        :param out_dim:
        """
        super(CPMNet, self).__init__()
        self.lsd_dim = lsd_dim
        self.out_dim = out_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.numclass = numclass


        #定义3个视角的网络结构，两层网络
        self.fc_sce = nn.Sequential(
            nn.Linear(self.lsd_dim, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            # nn.Dropout(p = 0.4),
            nn.Sigmoid(),  ## inplace = True可以节省运行内存，但会覆盖原来的值
            # nn.Dropout(p=0.3),
            # nn.Linear(self.hidden_dim1, self.hidden_dim2),
            # nn.BatchNorm1d(self.hidden_dim2),
            # nn.Tanh(), ## inplace = True可以节省运行内存，但会覆盖原来的值
            # nn.Dropout(p=0.3),
            nn.Linear(self.hidden_dim1, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            # nn.ReLU()  ## inplace = True可以节省运行内存，但会覆盖原来的值
        )

        self.fc_tra = nn.Sequential(
            nn.Linear(self.lsd_dim, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            # nn.Dropout(p = 0.4),
            nn.Sigmoid(),  ## inplace = True可以节省运行内存，但会覆盖原来的值
            # nn.Linear(self.hidden_dim1, self.hidden_dim2),
            # nn.BatchNorm1d(self.hidden_dim2),
            # # nn.Dropout(p=0.3),
            # nn.Tanh(),  ## inplace = True可以节省运行内存，但会覆盖原来的值
            nn.Linear(self.hidden_dim1, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            # nn.ReLU()  ## inplace = True可以节省运行内存，但会覆盖原来的值
        )
        self.fc_mfcc = nn.Sequential(
            nn.Linear(self.lsd_dim, self.hidden_dim1),
            nn.BatchNorm1d(self.hidden_dim1),
            # nn.Dropout(p = 0.4),
            nn.Sigmoid(),  ## inplace = True可以节省运行内存，但会覆盖原来的值
            # nn.Linear(self.hidden_dim1, self.hidden_dim2),
            # nn.BatchNorm1d(self.hidden_dim2),
            # # nn.Dropout(p=0.3),
            # nn.Tanh(),  ## inplace = True可以节省运行内存，但会覆盖原来的值
            nn.Linear(self.hidden_dim1, self.out_dim),
            nn.BatchNorm1d(self.out_dim),
            # nn.ReLU()  ## inplace = True可以节省运行内存，但会覆盖原来的值
        )
        self.classfier = nn.Sequential(
            nn.Linear(self.lsd_dim, self.numclass)
        )


        self.weights = self.init_weights()

    #初始化参数
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, H):
        """
        :param h:
        :return: x_hat
        """
        x_scene = self.fc_sce(H)
        x_tra = self.fc_tra(H)
        # x_mfcc = self.fc_tra(H)
        # y_hat = self.classfier(H)
        return x_scene, x_tra, H

class AutoEncoder(nn.Module):
    """
    :得到视觉模态的公共表示
    """

    def __init__(self,dim_encoder,out_dim_common):   ## out_dim是公共表示的维数
        super(AutoEncoder,self).__init__()
        self.dim = dim_encoder
        self.out_dim = out_dim_common
        self.hidden = (self.out_dim + self.dim)//2
        # self.encode_mlp = AE_MLP(name='encode_mlp', dims=[self.dim, self.out_dim], activations=[torch.nn.Sigmoid])
        # self.decode_mlp = AE_MLP(name='decode_mlp', dims=[self.out_dim, (self.out_dim + self.dim)/2, self.dim], activations=[torch.nn.Tanh, torch.nn.ReLU])
        self.encoder = nn.Sequential(
            nn.Linear(self.dim, self.out_dim),
            nn.Sigmoid(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.out_dim, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.dim),
            nn.ReLU(),
        )
        self.weights = self.init_weights()
    #
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                stdv = 1. / math.sqrt(m.weight.size(1))
                m.weight.data.uniform_(-stdv, stdv)
                m.bias.data.zero_()

    def forward(self,x):
        encoded = self.encoder(x)  ## 得到降维后的公共表示
        decoded = self.decoder(encoded)  ## 得到重构后的表示
        return encoded, decoded

