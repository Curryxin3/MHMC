# coding = utf-8
import os
import shutil
from easydict import EasyDict
from myDataset import myDataset
from torch.utils.data import DataLoader
from model import *
from utils import *
import yaml
import scipy.io as scio
from tensorboardX import SummaryWriter
import logging

def main():
    # 读取公共参数
    with open(r"/home/lyx/MHMC/MHMCv3/args.yaml") as f:
        opt = yaml.load(f)
    opt = EasyDict(opt['PARAMETER'])
    writer = SummaryWriter(log_dir=opt.tensorboard, flush_secs=5)
    # logging的基本配置
    logging.basicConfig(level=logging.INFO,  # 控制台打印的日志级别
                        filename=opt.log,
                        filemode='a',  ##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                                       # a是追加模式，默认如果不写的话，就是追加模式
                        format=
                        '%(asctime)s - %(levelname)s: %(message)s'
                        # 日志格式
                        )
    # 加载数据
    train_sce = scio.loadmat(opt.train_scene_dir)['train'][::]
    train_tra = scio.loadmat(opt.train_tra_dir)['train'][::]
    train_y = scio.loadmat(opt.train_label_dir)['train'][::]
    test_sce = scio.loadmat(opt.test_scene_dir)['test'][::]
    test_tra = scio.loadmat(opt.test_tra_dir)['test'][::]
    test_label = scio.loadmat(opt.test_label_dir)['test'][::]
    print('...loading and splitting data finish')
    # 定义模型
    model = MHMC(opt)
    model.train(opt, train_sce, train_tra, train_y, model, writer)
    #model.retune(opt, scene, tra, y_true, H_train, model, writer)
    model.test(opt, test_sce, test_tra, test_label, model, writer)  ## 输出验证集的精确度

def save_checkpoint(state, is_best, filename = 'checkpoint.pth.tar', prefix = ''):
    torch.save(state,prefix + filename)
    if is_best:    ## 保存最佳模型
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')

if __name__ == '__main__':
    main()
