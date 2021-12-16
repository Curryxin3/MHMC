# coding = utf-8
import torch
from CPM import *
from GCN import *
import numpy as np
from torch.autograd import Variable
from torch.optim import SGD
from utils import *
from collections import OrderedDict
import logging
import os
from sklearn.utils import shuffle
import shutil

class MHMC(object):
    def __init__(self,opt):
        self.opt = opt
        #特征空间学习和标签空间学习
        self.CPM = CPMNet(opt.cpm_lsd_dim, opt.cpm_hidden_dim1, opt.cpm_hidden_dim2, opt.cpm_out_dim, opt.classes)
        self.GCN = GCNNet(opt, opt.adj,opt.classes, opt.t, opt.adj_file, opt.labelgcn_name, opt.in_channel)
        self.H_train = self.H_init('train')
        self.H_test = self.H_init('test')
        self.params_theta = list(self.CPM.parameters())
        self.params_theta += list(self.GCN.parameters())
        self.optimizer_theta = SGD(self.params_theta, lr=self.opt.learning_rate,
                                   momentum=opt.optimizer.momentum,
                                   weight_decay=opt.optimizer.weight_decay,
                                   nesterov=opt.optimizer.nesterov
                                   )
        lr = opt.learning_rate
        self.learning_rate_train = np.linspace(opt.learning_rate, np.power(10, -4.), opt.train_epochs + 1)
        self.learning_rate_test = np.linspace(opt.learning_rate, np.power(10, -4.), opt.test_epochs + 1)

        if torch.cuda.is_available():
            self.CPM.cuda(opt.device)
            self.GCN.cuda(opt.device)

    def adjust_learning_rate(opt, optimizer, epoch):
        """Sets the learning rate to the initial LR
           decayed by 10 every 10 epochs"""
        lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    #H初始化
    def H_init(self, a):
        if a == 'train':
            h = self.xavier_init(self.opt.num_train, self.opt.cpm_lsd_dim)
        elif a == 'test':
            h = self.xavier_init(self.opt.num_test, self.opt.cpm_lsd_dim)
        return h
    def xavier_init(self, fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        a = np.random.uniform(low, high, (fan_in, fan_out))
        a = a.astype('float32')
        return a
    def train_start(self):
        self.CPM.train()
        self.GCN.train()
    def eval_start(self):
        self.CPM.eval()
        self.GCN.eval()

    def train(self, opt, scene, tra, y_true, model, writer):
        best_accuracy = 0
        if opt.resume:
            if os.path.isfile(opt.resume):
                print("=> loading checkpoint '{}'".format(opt.resume))
                checkpoint = torch.load(opt.resume)
                start_epoch = checkpoint['epoch']
                best_accuracy = checkpoint['best_accuracy']
                self.load_state_dict(checkpoint['model'])
                print("=> loaded checkpoint '{}' (epoch {}, best_accuracy {})"
                      .format(opt.resume, start_epoch, best_accuracy))
            else:
                print("=> no checkpoint found at '{}'".format(opt.resume))
        model.train_start()
        print("Start train . . . ")
        for epoch in range(opt.train_epochs):
            loss_epoch = 0
            loss_reco_epoch = 0
            loss_class_epoch = 0
            map = 0
            scene, tra, self.H_train, y_true = shuffle(scene, tra, self.H_train, y_true) #打乱数据
            scene = torch.from_numpy(scene).cuda(self.opt.device)  #转为cuda，tensor
            tra = torch.from_numpy(tra).cuda(self.opt.device)
            y_true = torch.from_numpy(y_true).cuda(self.opt.device)
            self.H_train = Variable(torch.from_numpy(self.H_train)).cuda(self.opt.device)
            num_batchs = math.ceil(opt.num_train / opt.train_batch_size)  # fix the last batch
            for num_batch_i in range(int(num_batchs)):
                start_idx, end_idx = num_batch_i * opt.train_batch_size, (num_batch_i + 1) * opt.train_batch_size
                end_idx = min(opt.num_train, end_idx)
                batch_scene = scene[start_idx: end_idx, ...]
                batch_tra = tra[start_idx: end_idx, ...]
                batch_y_true = y_true[start_idx: end_idx:, ...]
                batch_h = self.H_train[start_idx: end_idx, ...]
                batch_h.requires_grad = True
                self.optimizer_Htr = SGD([batch_h], lr=self.opt.learning_rate,
                                 momentum=opt.optimizer.momentum,
                                 weight_decay=opt.optimizer.weight_decay,
                                 nesterov=opt.optimizer.nesterov
                                 )
                #step1：update theta
                scene_hat, tra_hat, H = self.CPM(batch_h)
                y_hat = self.GCN(batch_h)
                loss_theta , loss_reco, loss_class = self.loss_criterion(batch_scene, scene_hat, batch_tra, tra_hat, batch_y_true, y_hat)
                self.optimizer_theta.zero_grad()  # 梯度清零
                loss_theta.backward()
                self.optimizer_theta.step()
                #step2：update H
                for i in range(15):
                    scene_hat, tra_hat, H = self.CPM(batch_h)
                    y_hat = self.GCN(batch_h)
                    loss_h, loss_reco, loss_class = self.loss_criterion(batch_scene, scene_hat, batch_tra, tra_hat, batch_y_true, y_hat)
                    self.optimizer_Htr.zero_grad()  # 梯度清零
                    loss_h.backward()
                    self.optimizer_Htr.step()

                # scene_hat, tra_hat, H = self.CPM(batch_h)
                # y_hat = self.GCN(batch_h)
                # loss_gcn = self.loss3_criterion(batch_y_true, y_hat)
                # self.optimizer_gcn.zero_grad()  # 梯度清零
                # loss_gcn.backward()
                # self.optimizer_gcn.step()
                #cal loss & map
                y_hat = self.GCN(batch_h)
                loss_batch, loss_reco, loss_class = self.loss_criterion(batch_scene, scene_hat, batch_tra, tra_hat, batch_y_true, y_hat)
                # batch_y_true = batch_y_true.cuda().data.cpu()
                map_batch = cal_ap(y_hat, batch_y_true).mean()

                loss_epoch = loss_epoch + loss_batch
                loss_reco_epoch = loss_reco_epoch + loss_reco
                loss_class_epoch = loss_class_epoch + loss_class
                map = map + map_batch
                print('Train: [{0}/{1}]\t'
                  'Loss_batch {loss_batch:.4f}\t'
                  'Loss_reco: {loss_reco:.4f}\t'
                  'Loss_class: {loss_class:.4f}\t'
                  'lr: {lr: .4f}\t'
                  'map_batch: {map_batch:.4f}\t'.format(
                num_batch_i, num_batchs, loss_batch=loss_batch, loss_reco=loss_reco, loss_class=loss_class, lr = self.optimizer_theta.state_dict()['param_groups'][0]['lr'],
                map_batch=map_batch))
            self.H_train = self.H_train.cuda().data.cpu().numpy()
            scene = scene.cuda().data.cpu().numpy()
            tra = tra.cuda().data.cpu().numpy()
            y_true = y_true.cuda().data.cpu().numpy()

            map = map / num_batchs
            loss_avg = loss_epoch / num_batchs
            loss_reco_avg = loss_reco_epoch / num_batchs
            loss_class_avg = loss_class_epoch / num_batchs
            print('Epoch: [{0}]\t'
                  'Train Loss_epoch {loss:.4f} \t'
                  'Loss_avg {loss_avg:.4f} \t'
                  'Loss_reco_avg {loss_reco_avg:.4f}\t'
                  'Loss_class_avg {loss_class_avg:.4f}\t'
                  'lr: {lr: .4f}\t'
                  'mAP {map:.4f}\t'.format(epoch, loss=loss_epoch, loss_avg=loss_avg, loss_reco_avg=loss_reco_avg,
                                         loss_class_avg=loss_class_avg, lr=self.optimizer_theta.state_dict()['param_groups'][0]['lr'],
                          map=map))
            # remember best accuracy and save checkpoint
            is_best = map > best_accuracy  ## 记录最大精确度值
            best_accuracy = max(map, best_accuracy)  ## 在最大精确度的时候save 模型
            if is_best:
                self.save_checkpoint({  ## 定义的save_checkpoint
                    'epoch': epoch + 1,
                    'model': self.state_dict(),
                    'best_accuracy': best_accuracy,
                    'learning_rate': self.optimizer_theta.state_dict()['param_groups'][0]['lr'],
                    'opt': opt,
                }, is_best, filename='checkpoint_' + str(epoch) + '.pth.tar', prefix=opt.logger_name + '/')
            # 输出日志信息
            logging.info(
                'Train: Epoch: [{0}]\t'
                'Loss_avg {loss_avg:.4f} \t'
                'Loss_reco_avg {loss_reco_avg:.4f} \t'
                'Loss_class_avg {loss_class_avg:.4f} \t'
                'mAP {map:.4f}\t'
                'lr {lr: .4f}'.format(
                epoch, loss_avg=loss_avg,loss_reco_avg=loss_reco_avg,loss_class_avg=loss_class_avg,
                map=map, lr=self.optimizer_theta.state_dict()['param_groups'][0]['lr']))
            writer.add_scalar('Train_Loss', loss_avg, epoch)
            writer.add_scalar('Train_mAP', map, epoch)
            writer.add_scalar('Train_Loss_reco', loss_reco_avg, epoch)

    def retune(self, opt, scene, tra, y_true, H_train, model, writer):
        model.train_start()
        print("Start fine retune . . . ")
        loss_epoch = 0
        for epoch in range(opt.rt_epochs):
            scene, tra, H_train, y_true = shuffle(scene, tra, H_train, y_true)
            scene = torch.from_numpy(scene).cuda(self.opt.device)
            tra = torch.from_numpy(tra).cuda(self.opt.device)
            y_true = torch.from_numpy(y_true).cuda(self.opt.device)
            H_train = Variable(torch.from_numpy(H_train)).cuda(self.opt.device)
            num_batchs = math.ceil(opt.num_train / opt.train_batch_size)  # fix the last batch
            for num_batch_i in range(int(num_batchs)):
                start_idx, end_idx = num_batch_i * opt.train_batch_size, (num_batch_i + 1) * opt.train_batch_size
                end_idx = min(opt.num_train, end_idx)
                batch_scene = scene[start_idx: end_idx, ...]
                batch_tra = tra[start_idx: end_idx, ...]
                batch_h = H_train[start_idx: end_idx, ...]
                batch_h.requires_grad = True
            #update cpm
                scene_hat, tra_hat, H = self.CPM(batch_h)
                loss_cpm = self.loss2_criterion(batch_scene, scene_hat, batch_tra, tra_hat)
                self.optimizer_rt.zero_grad()
                loss_cpm.backward()
                self.optimizer_rt.step()
            #cal loss：
                scene_hat, tra_hat, H = self.CPM(batch_h)
                loss_rt = self.loss2_criterion(batch_scene, scene_hat, batch_tra, tra_hat)
                loss_epoch = loss_epoch + loss_rt
                print('Retune: [{0}/{1}]\t'
                      'Loss_batch {loss_batch:.4f}'.format(
                      num_batch_i, num_batchs, loss_batch=loss_rt))
            H_train = H_train.cuda().data.cpu().numpy()
            scene = scene.cuda().data.cpu().numpy()
            tra = tra.cuda().data.cpu().numpy()
            y_true = y_true.cuda().data.cpu().numpy()

            loss_avg = loss_epoch / num_batchs
            print('Epoch: [{0}]\t'
                  'Retune Loss_epoch {loss:.4f} \t'
                  'Loss_avg {loss_avg:.4f} '.format(epoch, loss=loss_epoch, loss_avg=loss_avg))
            logging.info(
                  'Retune: Epoch: [{0}]\t'
                  'Loss_avg {loss_avg:.4f}'.format(
                   epoch, loss_avg=loss_avg))
            writer.add_scalar('Retune_Loss', loss_avg, epoch)
    def test(self, opt, scene, tra, y_true, model, writer):
        model.eval_start()
        print("Start test . . . ")
        for epoch in range(opt.test_epochs):
            map = 0
            coverage = 0
            rankingLoss = 0
            HammingLoss = 0
            one_error = 0
            loss_epoch = 0
            loss_reco_epoch = 0
            loss_class_epoch = 0
            scene, tra, self.H_test, y_true = shuffle(scene, tra, self.H_test, y_true)
            scene = torch.from_numpy(scene).cuda(self.opt.device)
            tra = torch.from_numpy(tra).cuda(self.opt.device)
            y_true = torch.from_numpy(y_true).cuda(self.opt.device)
            self.H_test = Variable(torch.from_numpy(self.H_test)).cuda(self.opt.device)
            num_batchs = math.ceil(opt.num_test / opt.test_batch_size)  # fix the last batch
            for num_batch_i in range(int(num_batchs) ):
                start_idx, end_idx = num_batch_i * opt.test_batch_size, (num_batch_i + 1) * opt.test_batch_size
                end_idx = min(opt.num_test, end_idx)
                batch_scene = scene[start_idx: end_idx, ...]
                batch_tra = tra[start_idx: end_idx, ...]
                batch_y_true = y_true[start_idx: end_idx:, ...]
                batch_h = self.H_test[start_idx: end_idx, ...]
                batch_h.requires_grad = True
                self.optimizer_Hte = SGD([batch_h], lr=self.opt.learning_rate,
                                     momentum=opt.optimizer.momentum,
                                     weight_decay=opt.optimizer.weight_decay,
                                     nesterov=opt.optimizer.nesterov
                                     )
            #updata h_test:
                for i in range(15):
                    scene_hat, tra_hat, H = self.CPM(batch_h)
                    loss_h = self.loss2_criterion(batch_scene, scene_hat, batch_tra, tra_hat)
                    self.optimizer_Hte.zero_grad()  # 梯度清零
                    loss_h.backward()
                    self.optimizer_Hte.step()
            #cal loss & map:
                y_hat = self.GCN(batch_h)
                loss_batch, loss_reco, loss_class= self.loss_criterion(batch_scene, scene_hat, batch_tra, tra_hat, batch_y_true, y_hat)
                map_batch = cal_ap(y_hat, batch_y_true).mean()
                coverage_batch = cal_coverage(opt, y_hat, batch_y_true)
                rankingLoss_batch = cal_RankingLoss(opt, y_hat, batch_y_true)
                HammingLoss_batch = cal_HammingLoss(opt, y_hat, batch_y_true)
                one_error_batch = cal_one_error(opt, y_hat, batch_y_true)
                loss_epoch = loss_epoch + loss_batch
                loss_reco_epoch = loss_reco_epoch +loss_reco
                loss_class_epoch = loss_class_epoch + loss_class
                map = map + map_batch
                coverage = coverage + coverage_batch
                rankingLoss = rankingLoss + rankingLoss_batch
                HammingLoss = HammingLoss + HammingLoss_batch
                one_error = one_error + one_error_batch
                print('Test: [{0}/{1}]\t'
                      'Loss_batch {loss_batch:.4f}\t'
                      'Loss_reco {loss_reco:.4f}\t'
                      'Loss_class {loss_class:.4f}\t'
                      'lr: {lr: .4f}\n'
                      'map_batch: {map_batch:.4f}\t'
                      'coverage_batch {coverage_batch: .4f}\t'
                      'rankingLoss_batch {rankingLoss_batch: .4f}\t'
                      'HammingLoss_batch {HammingLoss_batch: .4f}\t'
                      'one_error_batch {one_error_batch: .4f}'.format(
                      num_batch_i, num_batchs, loss_batch=loss_batch,loss_reco=loss_reco, loss_class=loss_class,
                      lr = self.optimizer_Hte.state_dict()['param_groups'][0]['lr'],map_batch=map_batch,
                      coverage_batch=coverage_batch, rankingLoss_batch=rankingLoss_batch, HammingLoss_batch=HammingLoss_batch, one_error_batch=one_error_batch))
            self.H_test = self.H_test.cuda().data.cpu().numpy()
            scene = scene.cuda().data.cpu().numpy()
            tra = tra.cuda().data.cpu().numpy()
            y_true = y_true.cuda().data.cpu().numpy()

            map = map / num_batchs
            coverage = coverage / num_batchs
            rankingLoss = rankingLoss / num_batchs
            HammingLoss = HammingLoss / num_batchs
            one_error = one_error / num_batchs
            loss_avg = loss_epoch / num_batchs
            loss_reco_avg = loss_reco_epoch / num_batchs
            loss_class_avg = loss_class_epoch / num_batchs
            print('Epoch: [{0}]\t'
                  'Test Loss_epoch {loss:.4f} \t'
                  'Loss_avg {loss_avg:.4f} \t'
                  'Loss_reco_avg {loss_reco_avg:.4f} \t'
                  'Loss_class_avg {loss_class_avg:.4f} \t'
                  'lr: {lr: .4f}\n'
                  'mAP {map:.4f}\t'
                  'coverage {coverage:.4f}\t'
                  'rankingLoss {rankingLoss:.4f}\t'
                  'HammingLoss {HammingLoss:.4f}\t'
                  'one_error {one_error:.4f}'
                  .format(epoch, loss=loss_epoch, loss_avg=loss_avg, loss_reco_avg=loss_reco_avg,
                                 loss_class_avg=loss_class_avg, lr=self.optimizer_Hte.state_dict()['param_groups'][0]['lr'],
                                 map=map, coverage = coverage, rankingLoss = rankingLoss, HammingLoss = HammingLoss,one_error = one_error))
            logging.info(
                'Test: Epoch: [{0}]\t'
                'Loss_avg {loss_avg:.4f} \t'
                'Loss_reco_avg {loss_reco_avg:.4f} \t'
                'Loss_class_avg {loss_class_avg:.4f} \t'
                'lr: {lr: .4f}\n'
                'mAP {map:.4f}\t'
                'coverage {coverage:.4f}\t'
                'rankingLoss {rankingLoss:.4f}\t'
                'HammingLoss {HammingLoss:.4f}\t'
                'one_error {one_error:.4f}'.format(
                    epoch, loss_avg=loss_avg, loss_reco_avg=loss_reco_avg, loss_class_avg=loss_class_avg,
                    lr=self.optimizer_Hte.state_dict()['param_groups'][0]['lr'], map=map, coverage=coverage, rankingLoss=rankingLoss, HammingLoss=HammingLoss, one_error=one_error))
            writer.add_scalar('Test_Loss', loss_avg, epoch)
            writer.add_scalar('Test_mAP', map, epoch)
    def loss_criterion(self, scene, x_scene_hat, tra, x_tra_hat,  y, y_hat):
        recon_loss = torch.norm(scene - x_scene_hat) \
                       + torch.norm(tra - x_tra_hat)
        # recon_loss = torch.div(recon_loss, scene.shape[0])
        recon_loss = self.opt.eta * recon_loss
        classification = torch.nn.BCEWithLogitsLoss()
        class_loss = classification(y_hat, y.float())
        all_loss = class_loss + recon_loss
        return all_loss, recon_loss, class_loss

    def loss2_criterion(self, scene, x_scene_hat, tra, x_tra_hat):
        recon_loss = torch.norm(scene - x_scene_hat) \
                       + torch.norm(tra - x_tra_hat)
        # recon_loss = torch.div(recon_loss, scene.shape[0])
        all_loss = self.opt.eta * recon_loss
        return all_loss

    def loss3_criterion(self, y, y_hat):
        classification = torch.nn.BCEWithLogitsLoss()
        class_loss = classification(y_hat, y.float())
        all_loss = class_loss
        return all_loss


    def state_dict(self):
        state_dict =[self.CPM.state_dict(),self.GCN.state_dict()]
        return state_dict

    def load_state_dict(self,state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict[0].items():
            new_state_dict[k] = v
        self.CPM.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[1].items():
            new_state_dict[k] = v
        self.GCN.load_state_dict(new_state_dict, strict=True)

    def state_dict(self):
        state_dict =[self.CPM.state_dict(),self.GCN.state_dict()]
        return state_dict

    def load_state_dict(self,state_dict):
        new_state_dict = OrderedDict()
        for k, v in state_dict[0].items():
            new_state_dict[k] = v
        self.CPM.load_state_dict(new_state_dict, strict=True)

        new_state_dict = OrderedDict()
        for k, v in state_dict[1].items():
            new_state_dict[k] = v
        self.GCN.load_state_dict(new_state_dict, strict=True)

    def save_checkpoint(self, state, is_best, filename ='checkpoint.pth.tar', prefix=''):
        torch.save(state, prefix + filename)
        if is_best:  ## 保存最佳模型
            shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')

