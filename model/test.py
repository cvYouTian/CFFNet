import torch
from tqdm import tqdm
import numpy as np
import torch.utils.data as Data
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F
from cffnet import cffnet
import os.path as osp
import torchvision.transforms as transforms
from PIL import Image, ImageOps, ImageFilter
from metrics import SigmoidMetric, SamplewiseSigmoidMetric
from pathlib import Path
from metric import PD_FA, ROCMetric, mIoU


def parse_args():
    parser = ArgumentParser(description='Implement of DETL-Net model')
    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--batch-size', type=int, default=2, help='batch_size for training')
    parser.add_argument('--device', type=str, default="cuda", help='if use gpu')
    parser.add_argument('--output_dir', type=str,
                        default="/home/youtian/Documents/pro/pyCode/CFFNet/exp/output", help='output dir')
    parser.add_argument('--weight_path', type=str,
                        default="/home/youtian/Documents/pro/pyCode/CFFNet/model/result/"
                                "Epoch-344_IoU-0.6603_nIoU-0.6718.pkl", help='weight path')
    parser.add_argument('--base_datadir', type=str,
                        default="/home/youtian/Documents/pro/pyCode/MLF-IRDet/data/dataset/"
                                "IRSTD-1k", help='data root path')
    parser.add_argument('--blocks-per-layer', type=int, default=4, help='blocks per layer')
    args = parser.parse_args()

    return args


class Dataset(Data.Dataset):
    def __init__(self, args):
        base_dir = args.base_datadir
        # 测试集
        txtfile = "test.txt"

        self.list_dir = osp.join(base_dir, txtfile)
        self.imgs_dir = osp.join(base_dir, 'images')
        self.label_dir = osp.join(base_dir, 'masks')

        self.names = []
        with open(self.list_dir, 'r') as f:
            self.names += [line.strip() for line in f.readlines()]

        self.crop_size = args.crop_size
        self.base_size = args.base_size
        # 初始化
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

    def __getitem__(self, i):
        name = self.names[i]
        # 图片
        img_path = osp.join(self.imgs_dir, name + '.png')
        # 标注
        label_path = osp.join(self.label_dir, name + '.png')

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(label_path)
        # 通过计算iamge和label得到gt——edge
        img, mask = self._testval_sync_transform(img, mask)
        edgemap = mask

        img, mask, edgemap = self.transform(img), transforms.ToTensor()(mask), transforms.ToTensor()(edgemap)

        return img, mask, edgemap, name

    def __len__(self):
        return len(self.names)

    def _testval_sync_transform(self, img, mask):
        base_size = self.base_size
        img = img.resize((base_size, base_size), Image.BILINEAR)
        mask = mask.resize((base_size, base_size), Image.NEAREST)

        return img, mask


class Get_gradient_nopadding(nn.Module):
    def __init__(self):
        super(Get_gradient_nopadding, self).__init__()
        # 定义两个卷积核
        kernel_v = [[0, -1, 0],
                    [0, 0, 0],
                    [0, 1, 0]]
        kernel_h = [[0, 0, 0],
                    [-1, 0, 1],
                    [0, 0, 0]]
        # [1, 1, 3, 3]
        kernel_h = torch.FloatTensor(kernel_h).unsqueeze(0).unsqueeze(0)
        # [1, 1, 3, 3]
        kernel_v = torch.FloatTensor(kernel_v).unsqueeze(0).unsqueeze(0)

        if torch.cuda.is_available():
            # tensor形式的参数
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cuda()
            self.weight_v = nn.Parameter(data=kernel_v, requires_grad=False).cuda()
        else:
            self.weight_h = nn.Parameter(data=kernel_h, requires_grad=False).cpu()
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


# 直接使用pkl格式文件加载模型
# def load_model():
#     """加载训练好的模型"""
#     model = torch.load(
#         "/home/youtian/Documents/pro/pyCode/CFFNet/model/result/Epoch- 58_IoU-0.4287_nIoU-0.3879.pkl",
#         map_location='cpu'
#     )
#
#     return model.eval().cuda()
#

def load_model(args):
    # 1. 加载模型实例
    model = torch.load(
        args.weight_path,
        map_location='cpu')

    # 2. 提取state_dict
    state_dict = model.state_dict()

    # 3. 重建新模型并加载权重
    new_model = cffnet(layer_blocks=[4, 4, 4], channels=[8, 16, 32, 64])
    new_model.load_state_dict(state_dict)

    return new_model.eval().cuda()


def tensor_to_plt(tensor):
    """
    将PyTorch张量转换为Matplotlib可显示格式
    参数:
        tensor: 输入张量 (C,H,W)或(B,C,H,W)
    返回:
        numpy数组 (H,W,C)或(H,W) [0-1范围或0-255]
    """
    # 1. 分离批次维度（如有）
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)  # 去除batch维度 [B,C,H,W] -> [C,H,W]

    # 2. 转换通道顺序 PyTorch的[C,H,W] -> Matplotlib的[H,W,C]
    tensor = tensor.permute(1, 2, 0) if tensor.dim() == 3 else tensor

    # 3. 转换为CPU和numpy格式
    img = tensor.cpu().detach().numpy()

    # 4. 归一化处理
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
    elif img.dtype == torch.uint8:
        img = img.astype(np.uint8)  # 保持0-255范围

    return img.squeeze()  # 去除单通道维度（如果是灰度图）


def save_iamge(output, name, args):
    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)

    tensor = output
    if tensor.dim() == 4:
        tensor = output.squeeze(0)  # 去除batch维度 [B,C,H,W] -> [C,H,W]

    # 2. 转换通道顺序 PyTorch的[C,H,W] -> Matplotlib的[H,W,C]
    tensor = tensor.permute(1, 2, 0) if tensor.dim() == 3 else tensor

    # 3. 转换为CPU和numpy格式
    img = tensor.cpu().detach().numpy()

    # 4. 归一化处理
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img - img.min()) / (img.max() - img.min())  # 归一化到[0,1]
    elif img.dtype == torch.uint8:
        img = img.astype(np.uint8)  # 保持0-255范围
    path = f"{output_dir}/{name}.png"
    print(f"save to {path} \n")

    # 单独保存预测mask
    Image.fromarray((img.squeeze() * 255).astype(np.uint8)).save(path)


def plot_comp(data, output, label=None):
    # 可视化保存
    plt.figure(figsize=(15, 5))

    # 原始图像
    plt.subplot(1, 3, 1)
    plt.imshow(data)
    plt.title("Input Image")
    plt.axis('off')
    #
    # 预测结果
    plt.subplot(1, 3, 2)
    plt.imshow(output, cmap='gray')
    plt.title("Prediction")
    plt.axis('off')
    #
    # 叠加显示
    plt.subplot(1, 3, 3)
    plt.imshow(data)
    plt.imshow(output, alpha=0.3, cmap='jet')
    plt.title("Overlay")
    plt.axis('off')

    # 保存结果
    plt.savefig(f"../exp/result.png", bbox_inches='tight', dpi=600)
    plt.show()
    plt.close()

    # 单独保存预测mask
    Image.fromarray((output * 255).astype(np.uint8)).save(
        f"../exp/mask.png"
    )


if __name__ == '__main__':
    grad = Get_gradient_nopadding()
    gradmask = Get_gradientmask_nopadding()
    args = parse_args()
    valset = Dataset(args)
    # 对测试集进行测试
    folders_test = True

    net = load_model(args)
    net.eval()

    # 测试一系列图片
    if folders_test:
        print("初始化指标...")
        iou_metric = SigmoidMetric()
        nIoU_metric = SamplewiseSigmoidMetric(1, score_thresh=0.5)
        # 初始化
        best_iou = 0
        best_nIoU = 0
        best_FA = 1000000000000000
        best_PD = 0
        # ROC 曲线的评估
        ROC = ROCMetric(1, 10)
        PD_FA = PD_FA(1, 10)
        mIoU = mIoU(1)

        iou_metric.reset()
        mIoU.reset()
        nIoU_metric.reset()
        PD_FA.reset()
        print("开始测试...")
        for origin_data, labels, first_edge, name in valset:
            origin_data = origin_data.unsqueeze(0)
            first_edge = first_edge.unsqueeze(0)
            labels = labels.unsqueeze(0)

            edge_in = grad(origin_data.cuda())
            second_edge_gt = gradmask(first_edge.cuda())

            output, third_edge_out = net(origin_data.cuda(), edge_in.cuda())
            # 将测试结果保存在文件夹下
            save_iamge(second_edge_gt, name, args)

            # 打印指标
            output = output.cpu()
            labels = labels.cpu()

            iou_metric.update(output, labels)
            nIoU_metric.update(output, labels)
            ROC.update(output, labels)
            mIoU.update(output, labels)
            PD_FA.update(output, labels)

            FA, PD = PD_FA.get(len(valset))
            # print()

            _, mean_IOU = mIoU.get()
            _, IoU = iou_metric.get()
            _, nIoU = nIoU_metric.get()

        print("测试结束")
        print("指标如下:")
        if IoU > best_iou:
            best_iou = IoU
        if nIoU > best_nIoU:
            best_nIoU = nIoU
            # FA_PD
        if FA[0] * 1000000 > 0 and FA[0] * 1000000 < best_FA:
            best_FA = FA[0] * 1000000
        if PD[0] > best_PD:
            best_PD = PD[0]

        print("IOU:", best_iou, "\n", "nIOU:", best_nIoU, "\n", "FA:", best_FA, "\n", "PD", best_PD)


    # 测试单帧图片，输入的压验证集图片的索引
    else:
        origin_data, labels, first_edge = valset[8]

        origin_data.unsqueeze(0)
        first_edge.unsqueeze(0)
        labels.unsqueeze(0)

        edge_in = grad(origin_data.cuda())
        second_edge_gt = gradmask(first_edge.cuda())

        output, third_edge_out = net(origin_data.cuda(), edge_in.cuda())

        labels = labels[:, 0:1, :, :].cpu()
        output = output.cpu()
        edge_out = third_edge_out.cpu()
        edge_gt = second_edge_gt.cpu()

        labels = torch.clamp(labels, 0.0, 1.0)
        edge_gt = torch.clamp(edge_gt, 0.0, 1.0)

        data = tensor_to_plt(origin_data)
        edge = tensor_to_plt(third_edge_out)
        labels = tensor_to_plt(labels)
        output = tensor_to_plt(output)

        plot_comp(data, output)
