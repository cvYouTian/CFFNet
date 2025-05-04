import torch
from tqdm import tqdm
import numpy as np
from data.dataset_IRSTD1K import Dataset
import torch.utils.data as Data
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F
# 1. 加载模型结构
import os
from PIL import Image
from cffnet import cffnet  # 你的模型类



def parse_args():

    parser = ArgumentParser(description='Implement of DETL-Net model')

    parser.add_argument('--crop-size', type=int, default=480, help='crop image size')
    parser.add_argument('--base-size', type=int, default=512, help='base image size')
    parser.add_argument('--batch-size', type=int, default=2, help='batch_size for training')
    parser.add_argument('--epochs', type=int, default=500,
                        help='number of epochs, depends on your lr schedule 500 or 1000+ is available!')
    parser.add_argument('--warm-up-epochs', type=int, default=0, help='warm up epochs')
    parser.add_argument('--learning_rate', type=float, default=0.04, help='learning rate')
    parser.add_argument('--backbone-mode', type=str, default='detlnet_1k',
                        help='backbone mode: detlnet_sir, detlnet_1k')
    parser.add_argument('--fuse-mode', type=str, default='AsymBi',
                        help='fuse mode: BiLocal, AsymBi, BiGlobal')
    parser.add_argument('--device', type=str, default="cuda", help='if use gpu')
    parser.add_argument('--blocks-per-layer', type=int, default=4, help='blocks per layer')

    args = parser.parse_args()
    return args

# load
# def load_model():
#     """加载训练好的模型"""
#     model = torch.load(
#         "/home/youtian/Documents/pro/pyCode/CFFNet/model/result/Epoch- 58_IoU-0.4287_nIoU-0.3879.pkl",
#         map_location='cpu'
#     )
#
#     return model.eval().cuda()
#

def load_model():
    # 1. 加载模型实例
    model = torch.load(
            "/home/youtian/Documents/pro/pyCode/CFFNet/model/result/Epoch- 58_IoU-0.4287_nIoU-0.3879.pkl",
            map_location='cpu')

    # 2. 提取state_dict
    state_dict = model.state_dict()

    # 3. 重建新模型并加载权重
    new_model = cffnet(layer_blocks=[4, 4, 4], channels=[8, 16, 32, 64])
    new_model.load_state_dict(state_dict)

    return new_model.eval().cuda()


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


def plot_comp(data, output):


    # 可视化保存
    plt.figure(figsize=(15, 5))

    # 原始图像
    plt.subplot(1, 4, 1)
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
    args = parse_args()
    valset = Dataset(args, mode='val')
    # val_data_loader = Data.DataLoader(valset, batch_size=1, num_workers=1, pin_memory=True)
    net = load_model()

    grad = Get_gradient_nopadding()
    gradmask = Get_gradientmask_nopadding()

    net.eval()

    # tbar = tqdm(val_data_loader)
    #
    # for i, (data, labels, edge) in enumerate(tbar):
    #     with torch.no_grad:
    #         edge_in = grad(data.cuda())
    #         edge_gt = gradmask(edge.cuda())
    #         output, edge_out = net(data.cuda(), edge_in.cuda())
    #
    #         labels = labels[:, 0:1, :, :].cpu()
    #         output = output.cpu()
    #         edge_out = edge_out.cpu()
    #         edge_gt = edge_gt.cpu()
    #     labels = torch.clamp(labels, 0.0, 1.0)
    #     edge_gt = torch.clamp(edge_gt, 0.0, 1.0)

    origin_data, labels, first_edge = valset[0]
    origin_data = origin_data.unsqueeze(0)
    first_edge = first_edge.unsqueeze(0)
    labels = labels.unsqueeze(0)

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
