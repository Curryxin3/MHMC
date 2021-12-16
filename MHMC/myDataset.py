# coding=utf-8
from  torch.utils.data import Dataset
import scipy.io as sio

class myDataset(Dataset):

    def __init__(self, label_dir, scene_dir, tra_dir, labelgcn_name, split):
        """
        :param label_dir:
        :param scene_dir:
        :param mfcc_dir:
        :param tra_dir:
        :param labelgcn_name:
        :param split:
        """
        self.split = split
        self.scene = sio.loadmat(scene_dir)[self.split]
        #self.mfcc = sio.loadmat(mfcc_dir)[self.split]
        self.tra = sio.loadmat(tra_dir)[self.split]
        self.label = sio.loadmat(label_dir)[self.split]
        self.labelgcn = sio.loadmat(labelgcn_name)['labelVector']

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, idx):
        scene = self.scene[idx]
        #mfcc = self.mfcc[idx]
        tra = self.tra[idx]
        label = self.label[idx]
        return scene, tra, label, self.labelgcn