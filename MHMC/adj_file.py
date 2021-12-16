import numpy as np
import torch
import scipy.io as sio
import pickle
from easydict import EasyDict
import yaml

def make_adj_file():
    with open(r"/home/lyx/PycharmProjects/MHMCv1/args.yaml") as f:
        opt = yaml.load(f)
    opt = EasyDict(opt['PARAMETER'])
    dataset = sio.loadmat(opt.train_label_dir)['train']
    adj_matrix = np.zeros(shape=(opt.classes, opt.classes))
    nums_matrix = np.zeros(shape=(opt.classes))

    for index in range(len(dataset)):  #依次取出每行，即每个实例所属的类别
        data = dataset[index]
        for i in range(opt.classes):
            if data[i] == 1:          #先统计每个类别的个数
                nums_matrix[i] += 1
                for j in range(opt.classes):  #去除自己，统计两两类别共同出现的次数
                    if j != i:
                        if data[j] == 1:
                            adj_matrix[i][j] += 1

    adj = {'adj': adj_matrix,    #保存为字典
           'nums': nums_matrix}
    pickle.dump(adj, open('./adj.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    make_adj_file()