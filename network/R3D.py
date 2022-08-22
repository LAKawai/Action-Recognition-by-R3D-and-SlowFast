import torch
import torch.nn as nn
from torch.nn.modules.utils import _triple
from torchinfo import summary


class SpatialTemporalConv(nn.Module):
    def __init__(self, in_, out_, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), bias=False):
        super().__init__()
        kernel_size = _triple(kernel_size)
        self.conv = nn.Conv3d(in_, out_, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialTemporalResBlock(nn.Module):
    def __init__(self, in_, out_, kernel_size, down_sample=False):
        super().__init__()
        self.down_sample = down_sample
        padding = kernel_size // 2
        if self.down_sample:
            self.down_sample_conv = nn.Conv3d(in_, out_, (1, 1, 1), stride=(2, 2, 2))
            self.down_sample_bn = nn.BatchNorm3d(out_)
            self.conv1 = SpatialTemporalConv(in_, out_, kernel_size, stride=2, padding=padding)
        else:
            self.conv1 = SpatialTemporalConv(in_, out_, kernel_size, padding=padding)
        self.conv2 = nn.Conv3d(out_, out_, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(out_)
        self.out_relu = nn.ReLU()

    def forward(self, x):
        res = self.bn2(self.conv2(self.conv1(x)))
        if self.down_sample:
            x = self.down_sample_bn(self.down_sample_conv(x))
        return self.out_relu(res + x)


class SpatialTemporalResLayer(nn.Module):
    def __init__(self, in_, out_, kernel_size, layer_size, block_type=SpatialTemporalResBlock, down_sample=False):
        super().__init__()
        self.first_block = block_type(in_, out_, kernel_size, down_sample)
        self.blocks = nn.ModuleList([])
        for i in range(layer_size - 1):
            self.blocks += [block_type(out_, out_, kernel_size)]

    def forward(self, x):
        x = self.first_block(x)
        for block in self.blocks:
            x = block(x)
        return x


class R3DNet(nn.Module):
    def __init__(self, layer_sizes, block_type=SpatialTemporalResBlock):
        super().__init__()
        self.features = nn.Sequential(SpatialTemporalConv(3, 64, (3, 7, 7), stride=(1, 2, 2), padding=(1, 3, 3)),
                                      SpatialTemporalResLayer(64, 64, 3, layer_sizes[0], block_type),
                                      SpatialTemporalResLayer(64, 128, 3, layer_sizes[1], block_type, True),
                                      SpatialTemporalResLayer(128, 256, 3, layer_sizes[2], block_type, True),
                                      SpatialTemporalResLayer(256, 512, 3, layer_sizes[3], block_type, True))
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

    def forward(self, x):
        return self.avg_pool(self.features(x)).view(-1, 512)


class R3DModel(nn.Module):
    def __init__(self, num_classes, layer_sizes, block_type=SpatialTemporalResBlock, pretrained=False):
        super().__init__()
        self.res3d = R3DNet(layer_sizes, block_type)
        self.linear = nn.Linear(512, num_classes)
        self.init_weight()

    def forward(self, x):
        return self.linear(self.res3d(x))

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


if __name__ == '__main__':
    inputs = torch.rand(10, 3, 16, 112, 112)
    net = R3DModel(101, (2, 2, 2, 2), pretrained=False)
    outputs = net.forward(inputs)
    print(outputs.size())
    # summary(net, inputs.shape)
    print('total params: %.2fM' % (sum(p.numel() for p in net.parameters()) / 1000000.0))
