# -*-coding:utf-8-*-
import torch
import torch.nn as nn
import torch.utils.data as Data
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from tqdm import tqdm
import os
import os.path as ops
import numpy as np
import time
from data.dataset_IRSTD1K import Dataset
from cffnet import cffnet
from loss import SoftLoULoss1, FocalLoss
from metrics import SigmoidMetric, SamplewiseSigmoidMetric
from torchvision import utils as vutils
import torch.nn.functional as F

from metric import PD_FA, ROCMetric, mIoU


def parse_args():
    parser = ArgumentParser(description='Implement of DETL-Net model')

    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--batch-size', type=int, default=2, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs, depends on your lr schedule 500 or 1000+ is available!')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--learning_rate', type=float, default=0.04, help='learning rate')
    parser.add_argument('--fuse-mode', type=str, default='AsymBi',
                        help='fuse mode: BiLocal, AsymBi, BiGlobal')
    parser.add_argument('--device', type=str, default="cuda", help='if use gpu')
    parser.add_argument('--blocks-per-layer', type=int, default=4, help='blocks per layer')

    args = parser.parse_args()
    return args


class Trainer(object):
    def __init__(self, args):
        self.args = args
        trainset = Dataset(args, mode='train')
        valset = Dataset(args, mode='val')
        self.train_data_loader = Data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                                 pin_memory=True)
        self.val_data_loader = Data.DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)

        print("the step of validation : ", len(self.val_data_loader))

        self.grad = Get_gradient_nopadding()
        self.gradmask = Get_gradientmask_nopadding()

        layer_blocks = [args.blocks_per_layer] * 3
        channels = [8, 16, 32, 64]
        self.net = cffnet(layer_blocks, channels)
        # device = torch.device("cuda")

        self.device = args.device

        self.net.apply(self.weight_init)
        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            self.net = nn.DataParallel(self.net, device_ids=[0, 1]).cuda()

        self.net.to(self.device)
        self.criterion1 = SoftLoULoss1()
        self.criterion2 = nn.BCELoss()
        self.criterion3 = FocalLoss()
        self.bce = nn.BCELoss()

        self.optimizer = torch.optim.Adagrad(self.net.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=30)
        # iou和niou的评估
        self.iou_metric = SigmoidMetric()
        self.nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        # 初始化
        self.best_iou = 0
        self.best_nIoU = 0
        self.best_FA = 1000000000000000
        self.best_PD = 0
        # ROC 曲线的评估
        self.ROC = ROCMetric(1, 10)
        self.PD_FA = PD_FA(1, 10)
        self.mIoU = mIoU(1)

        # folders
        folder_name = '%s_%s' % (time.strftime('%Y-%m-%d-%H-%M-%S',
                                               time.localtime(time.time())), args.fuse_mode)
        self.save_folder = ops.join('result/', folder_name)

        self.save_pkl = ops.join(self.save_folder, 'checkpoint')
        if not ops.exists('result'):
            os.mkdir('result')
        if not ops.exists(self.save_folder):\
            os.mkdir(self.save_folder)
        if not ops.exists(self.save_pkl):
            os.mkdir(self.save_pkl)

        # SummaryWriter
        self.writer = SummaryWriter(log_dir=self.save_folder)
        self.writer.add_text(folder_name, 'Args:%s, ' % args)

        # Print info
        print('folder: %s' % self.save_folder)
        print('Args: %s' % args)
        print('fuse mode: %s' % args.fuse_mode)
        print('layer block number:', layer_blocks)
        print('channels', channels)

    def training(self, epoch):
        # training step
        losses = []
        losses_edge = []
        self.net.train()
        tbar = tqdm(self.train_data_loader)

        for i, (data, labels, edge) in enumerate(tbar):
            # data = data.cuda()
            if torch.cuda.is_available():
                data = data.cuda()
                labels = labels[:, 0:1, :, :].cuda()
                # 处理边缘特征和mask的边缘特征
                edge_in = self.grad(data.cuda())
                edge_gt = self.gradmask(edge.cuda())
            else:
                data = data.cpu()
                labels = labels[:, 0:1, :, :].cpu()
                edge_in = self.grad(data.cpu())
                edge_gt = self.gradmask(edge.cpu())

            # 这部分使用的强制转化，训练精度有待提升
            labels = torch.clamp(labels, 0.0, 1.0)
            edge_gt = torch.clamp(edge_gt, 0.0, 1.0)

            # 在net中将data和edge_in加到此cuda上
            output, edge_out = self.net(data, edge_in)

            loss_io = self.criterion1(output, labels)
            loss_edge = 10 * self.criterion2(edge_out, edge_gt) + self.criterion1(edge_out, edge_gt)

            loss = loss_io + loss_edge

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses_edge.append(loss_edge.item())
            losses.append(loss.item())

            tbar.set_description('Epoch:%3d, lr:%f, train loss:%f, edge_loss:%f'
                                 % (
                                 epoch, trainer.optimizer.param_groups[0]['lr'], np.mean(losses), np.mean(losses_edge)))

        self.scheduler.step(epoch)

        self.writer.add_scalar('Losses/train loss', np.mean(losses), epoch)
        self.writer.add_scalar('Learning rate/', trainer.optimizer.param_groups[0]['lr'], epoch)

    def validation(self, epoch):
        self.iou_metric.reset()
        self.mIoU.reset()
        self.nIoU_metric.reset()
        self.PD_FA.reset()
        eval_losses = []
        eval_losses_edge = []

        self.net.eval()
        tbar = tqdm(self.val_data_loader)

        for i, (data, labels, edge) in enumerate(tbar):
            with torch.no_grad():
                if torch.cuda.is_available():
                    edge_in = self.grad(data.cuda())
                    edge_gt = self.gradmask(edge.cuda())
                    output, edge_out = self.net(data.cuda(), edge_in.cuda())
                else:
                    edge_in = self.grad(data.cpu())
                    edge_gt = self.gradmask(edge.cpu())
                    output, edge_out = self.net(data.cpu(), edge_in.cpu())

                labels = labels[:, 0:1, :, :].cpu()
                output = output.cpu()
                edge_out = edge_out.cpu()
                edge_gt = edge_gt.cpu()

            # optim here ！！
            labels = torch.clamp(labels, 0.0, 1.0)
            edge_gt = torch.clamp(edge_gt, 0.0, 1.0)

            loss_io = self.criterion1(output, labels)
            loss_edge = 10 * self.bce(edge_out, edge_gt) + self.criterion1(edge_out, edge_gt)
            loss = loss_io + loss_edge

            eval_losses.append(loss.item())
            eval_losses_edge.append(loss_edge.item())

            self.iou_metric.update(output, labels)
            self.nIoU_metric.update(output, labels)
            self.ROC.update(output, labels)
            self.mIoU.update(output, labels)
            self.PD_FA.update(output, labels)
            FA, PD = self.PD_FA.get(len(self.val_data_loader))
            # print()

            _, mean_IOU = self.mIoU.get()
            _, IoU = self.iou_metric.get()
            _, nIoU = self.nIoU_metric.get()

            tbar.set_description('  Epoch:%3d, eval loss:%f, eval_edge:%f, IoU:%f, nIoU:%f, mIoU:%f, FA:%f, PD:%f'
                                 % (
                                 epoch, np.mean(eval_losses), np.mean(eval_losses_edge), IoU, nIoU, mean_IOU, FA[0], PD[0]))
        # 权重命名
        pkl_name = 'Epoch-%3d_IoU-%.4f_nIoU-%.4f.pkl' % (epoch, IoU, nIoU)

        if IoU > self.best_iou:
            torch.save(self.net, ops.join(self.save_pkl, pkl_name))
            self.best_iou = IoU
        if nIoU > self.best_nIoU:
            torch.save(self.net, ops.join(self.save_pkl, pkl_name))
            self.best_nIoU = nIoU
        # FA_PD
        if FA[0] * 1000000 > 0 and FA[0] * 1000000 < self.best_FA:
            self.best_FA = FA[0] * 1000000
        if PD[0] > self.best_PD:
            self.best_PD = PD[0]

        img_grid_i = vutils.make_grid(data, normalize=True, scale_each=True, nrow=8)
        self.writer.add_image('input img', img_grid_i, global_step=None)  # j 表示feature map数
        img_grid_o = vutils.make_grid(output, normalize=True, scale_each=True, nrow=8)
        self.writer.add_image('output img', img_grid_o, global_step=None)  # j 表示feature map数
        img_grid_eg = vutils.make_grid(edge_in, normalize=True, scale_each=True, nrow=8)
        self.writer.add_image('edge in', img_grid_eg, global_step=None)  # j 表示feature map数
        img_grid_eo = vutils.make_grid(edge_out, normalize=True, scale_each=True, nrow=8)
        self.writer.add_image('edge out', img_grid_eo, global_step=None)  # j 表示feature map数
        img_grad_gt = vutils.make_grid(edge_gt, normalize=True, scale_each=True, nrow=8)
        self.writer.add_image('edge gt', img_grad_gt, global_step=None)  # j 表示feature map数
        img_gt = vutils.make_grid(labels, normalize=True, scale_each=True, nrow=8)
        self.writer.add_image('img gt', img_gt, global_step=None)  # j 表示feature map数

        self.writer.add_scalar('Losses/eval_loss', np.mean(eval_losses), epoch)
        self.writer.add_scalar('Eval/IoU', IoU, epoch)
        self.writer.add_scalar('Eval/nIoU', nIoU, epoch)
        self.writer.add_scalar('Best/IoU', self.best_iou, epoch)
        self.writer.add_scalar('Best/nIoU', self.best_nIoU, epoch)
        # self.writer.add_scalar('Best/FA', self.best_FA, epoch)
        # self.writer.add_scalar('Best/PD', self.best_PD, epoch)
        # self.writer.add_scalar('FA_PD', PD, FA)
        # self.writer.add_scalar('FP_TP', ture_positive_rate, false_positive_rate)
        # self.writer.add_scalar('Pre_Recall', recall, precision)

    def weight_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.normal_(m.bias, 0)


class Get_gradient_nopadding(nn.Module):
    """
    stract edge feature
    """
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        # self.weight_h = nn.Parameter(data = kernel_h, requires_grad = False).to("cpu")
        if torch.cuda.is_available():
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
            # self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).to("cpu")
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        else:
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cpu()
            # self.weight_v = nn.Parameter(data = kernel_v, requires_grad = False).to("cpu")
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cpu()

    def forward(self, x):
        x0 = x[:, 0]

        x1 = x[:, 1]
        x2 = x[:, 2]

        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)

        x1_v = F.conv2d(x1.unsqueeze(1), self.weight_v, padding=1)
        x1_h = F.conv2d(x1.unsqueeze(1), self.weight_h, padding=1)

        x2_v = F.conv2d(x2.unsqueeze(1), self.weight_v, padding=1)
        x2_h = F.conv2d(x2.unsqueeze(1), self.weight_h, padding=1)

        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)
        x1 = torch.sqrt(torch.pow(x1_v, 2) + torch.pow(x1_h, 2) + 1e-6)
        x2 = torch.sqrt(torch.pow(x2_v, 2) + torch.pow(x2_h, 2) + 1e-6)

        x = torch.cat([x0, x1, x2], dim=1)
        return x


class Get_gradientmask_nopadding(nn.Module):
    """
    stract edge of mask feature
    """
    def __init__(self):
        super(Get_gradientmask_nopadding, self).__init__()
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)
        if torch.cuda.is_available():
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        else:
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cpu()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cpu()

    def forward(self, x):
        x0 = x[:, 0]
        x0_v = F.conv2d(x0.unsqueeze(1), self.weight_v, padding=1)
        x0_h = F.conv2d(x0.unsqueeze(1), self.weight_h, padding=1)
        x0 = torch.sqrt(torch.pow(x0_v, 2) + torch.pow(x0_h, 2) + 1e-6)

        return x0


if __name__ == '__main__':
    args = parse_args()
    trainer = Trainer(args)
    for epoch in range(1, args.epochs + 1):
        trainer.training(epoch)
        trainer.validation(epoch)

    print('Best IoU: %.5f, best nIoU: %.5f, best FA: %f, best PD: %.5f' % (
    trainer.best_iou, trainer.best_nIoU, trainer.best_FA, trainer.best_PD))
