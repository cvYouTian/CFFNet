import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


##################################

class SpatialAttention(nn.Module):
    """
    空间注意力
    """
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.sa = nn.Conv2d(2, 1, 7, padding=3, padding_mode='reflect', bias=True)

    def forward(self, x):
        x_avg = torch.mean(x, dim=1, keepdim=True)
        x_max, _ = torch.max(x, dim=1, keepdim=True)
        x2 = torch.cat([x_avg, x_max], dim=1)
        sattn = self.sa(x2)
        return sattn


class ChannelAttention(nn.Module):
    """
    时间注意力
    """
    def __init__(self, dim, reduction=8):
        super(ChannelAttention, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, dim, 1, padding=0, bias=True),
        )

    def forward(self, x):
        x_gap = self.gap(x)
        cattn = self.ca(x_gap)
        return cattn


class FusionAttention(nn.Module):
    """
    融合注意力
    """
    def __init__(self, dim):
        super(FusionAttention, self).__init__()
        self.pa2 = nn.Conv2d(2 * dim, dim, 7, padding=3, padding_mode='reflect', groups=dim, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, pattn1):
        x = x.unsqueeze(dim=2)  # B, C, 1, H, W
        pattn1 = pattn1.unsqueeze(dim=2)  # B, C, 1, H, W
        x2 = torch.cat([x, pattn1], dim=2)  # B, C, 2, H, W
        x2 = Rearrange('b c t h w -> b (c t) h w')(x2)
        pattn2 = self.pa2(x2)
        pattn2 = self.sigmoid(pattn2)
        return pattn2


class CSAModule(nn.Module):
    def __init__(self, dim, reduction=8):
        super(CSAModule, self).__init__()
        self.sa = SpatialAttention()
        self.ca = ChannelAttention(dim, reduction)
        self.pa = FusionAttention(dim)
        self.conv = nn.Conv2d(dim, dim, 1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        #cn
        initial = x + y

        cattn = self.ca(initial)
        sattn = self.sa(initial)

        pattn1 = sattn + cattn
        pattn2 = self.sigmoid(self.pa(initial, pattn1))

        result = initial + pattn2 * x + (1 - pattn2) * y
        result = self.conv(result)
        return result


if __name__ == '__main__':
    feture_test1 = torch.rand(1, 64, 120, 120)*255
    feture_test1 = feture_test1.to(torch.float32)
    feture_test2 = torch.rand(1, 64, 120, 120)*255
    feture_test2 = feture_test2.to(torch.float32)

    c1 = CSAModule(64)
    f1 = c1(feture_test1, feture_test2)
    print(f1.shape)
