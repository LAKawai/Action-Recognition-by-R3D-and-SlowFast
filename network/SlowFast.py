import torch
from torch import nn
import config
from torchinfo import summary

__all__ = ['resnet50', 'resnet101', 'resnet152', 'resnet200']


class BasicConv(nn.Module):
    def __init__(self, in_, out_, kernel_size, stride=(1, 1, 1), padding=(0, 0, 0), bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_, out_, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm3d(out_)

    def forward(self, x):
        return self.bn(self.conv(x))


class ResBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, head_conv=1):
        super().__init__()
        if stride != 1 or in_planes != planes * config.EXPANSION:
            self.conv_x = BasicConv(in_planes, planes * config.EXPANSION, 1, (1, stride, stride))
        else:
            self.conv_x = None

        if head_conv == 1:
            self.conv_first = nn.Sequential(BasicConv(in_planes, planes, 1),
                                            nn.ReLU(inplace=True))
        elif head_conv == 3:
            self.conv_first = nn.Sequential(BasicConv(in_planes, planes, (3, 1, 1), padding=(1, 0, 0)),
                                            nn.ReLU(inplace=True))
        else:
            raise ValueError('head conv is error!')

        self.conv = nn.Sequential(BasicConv(planes, planes, (1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1)),
                                  nn.ReLU(inplace=True),
                                  BasicConv(planes, planes * config.EXPANSION, 1))
        self.out_relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv(self.conv_first(x))
        if self.conv_x is not None:
            residual = self.conv_x(x)
        return self.out_relu(residual + out)


class ResLayer(nn.Module):
    def __init__(self, block=ResBlock, in_planes=8, planes=8, layer_size=3, stride=1, head_conv=1):
        super().__init__()
        self.blocks = nn.ModuleList([])
        self.block1 = block(in_planes, planes, stride, head_conv)
        for i in range(1, layer_size):
            self.blocks += [block(planes * config.EXPANSION, planes, head_conv=head_conv)]

    def forward(self, x):
        x = self.block1(x)
        for block in self.blocks:
            x = block(x)
        return x


class SlowFast(nn.Module):
    def __init__(self, block=ResBlock, layers=(3, 4, 6, 3), class_num=101, dropout=0.5):
        super().__init__()
        self.fast_in_places = 8
        self.fast_conv1 = nn.Sequential(BasicConv(3, 8, (5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3)),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.fast_res2 = ResLayer(block, self.fast_in_places, 8, layers[0], head_conv=3)
        self.fast_res3 = ResLayer(block, 32,  16, layers[1], stride=2, head_conv=3)
        self.fast_res4 = ResLayer(block, 64,  32, layers[2], stride=2, head_conv=3)
        self.fast_res5 = ResLayer(block, 128, 64, layers[3], stride=2, head_conv=3)

        self.lateral_pool = nn.Conv3d(8,   16,  (5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.lateral_res2 = nn.Conv3d(32,  64,  (5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.lateral_res3 = nn.Conv3d(64,  128, (5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)
        self.lateral_res4 = nn.Conv3d(128, 256, (5, 1, 1), stride=(8, 1, 1), padding=(2, 0, 0), bias=False)

        self.slow_in_planes = 80
        self.slow_conv1 = nn.Sequential(BasicConv(3, 64, (1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3), bias=False),
                                        nn.ReLU(inplace=True),
                                        nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)))
        self.slow_res2 = ResLayer(block, self.slow_in_planes, 64, layers[0], head_conv=1)
        self.slow_res3 = ResLayer(block, 320, 128, layers[1], stride=2, head_conv=1)
        self.slow_res4 = ResLayer(block, 640, 256, layers[2], stride=2, head_conv=3)
        self.slow_res5 = ResLayer(block, 1280, 512, layers[3], stride=2, head_conv=3)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256 + 2048, class_num, bias=False)

    def forward(self, x):
        fast, lateral = self.fast_path(x[:, :, ::2, :, :])
        slow = self.slow_path(x[:, :, ::16, :, :], lateral)
        return self.fc(self.dropout(torch.cat([slow, fast], dim=1)))

    def fast_path(self, x):
        lateral = []

        pool1 = self.fast_conv1(x)
        lateral_p = self.lateral_pool(pool1)
        lateral.append(lateral_p)

        res2 = self.fast_res2(pool1)
        lateral_r2 = self.lateral_res2(res2)
        lateral.append(lateral_r2)

        res3 = self.fast_res3(res2)
        lateral_r3 = self.lateral_res3(res3)
        lateral.append(lateral_r3)

        res4 = self.fast_res4(res3)
        lateral_r4 = self.lateral_res4(res4)
        lateral.append(lateral_r4)

        res5 = self.fast_res5(res4)
        x = nn.AdaptiveAvgPool3d(1)(res5)

        return x.view(-1, x.shape[1]), lateral

    def slow_path(self, x, lateral):
        pool1 = self.slow_conv1(x)

        x = torch.cat([pool1, lateral[0]], dim=1)
        res2 = self.slow_res2(x)

        x = torch.cat([res2, lateral[1]], dim=1)
        res3 = self.slow_res3(x)

        x = torch.cat([res3, lateral[2]], dim=1)
        res4 = self.slow_res4(x)

        x = torch.cat([res4, lateral[3]], dim=1)
        res5 = self.slow_res5(x)

        x = nn.AdaptiveAvgPool3d(1)(res5)

        return x.view(-1, x.shape[1])


def resnet20(**kwargs):
    model = SlowFast(layers=(1, 2, 2, 1), **kwargs)
    return model


def resnet32(**kwargs):
    model = SlowFast(layers=(2, 3, 3, 2), **kwargs)
    return model


def resnet50(**kwargs):
    model = SlowFast(layers=(3, 4, 6, 3), **kwargs)
    return model


def resnet101(**kwargs):
    model = SlowFast(layers=(3, 4, 6, 3), **kwargs)
    return model


def resnet152(**kwargs):
    model = SlowFast(layers=(3, 8, 36, 3), **kwargs)
    return model


def resnet200(**kwargs):
    model = SlowFast(layers=(3, 24, 36, 3), **kwargs)
    return model


if __name__ == '__main__':
    num_classes = 101
    inputs = torch.rand(1, 3, 64, 224, 224)
    net = resnet50(class_num=num_classes)
    output = net.forward(inputs)
    print(output.shape)
    summary(net, inputs.shape)
