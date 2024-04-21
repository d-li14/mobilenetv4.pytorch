"""
Creates a MobileNetV4 Model as defined in:
Danfeng Qin, Chas Leichner, Manolis Delakis, Marco Fornoni, Shixin Luo, Fan Yang, Weijun Wang, Colby Banbury, Chengxi Ye, Berkin Akin, Vaibhav Aggarwal, Tenghui Zhu, Daniele Moro, Andrew Howard. (2024).
MobileNetV4 - Universal Models for the Mobile Ecosystem
arXiv preprint arXiv:2404.10518.
"""

import torch
import torch.nn as nn
import math


__all__ = ['mobilenetv4_conv_small', 'mobilenetv4_conv_medium', 'mobilenetv4_conv_large',
           'mobilenetv4_hybrid_medium', 'mobilenetv4_hybrid_large']


def make_divisible(value, divisor, min_value=None, round_down_protect=True):
    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if round_down_protect and new_value < 0.9 * value:
        new_value += divisor
    return new_value


class ConvBN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ConvBN, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, (kernel_size - 1)//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x):
        return self.block(x)


class UniversalInvertedBottleneck(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 expand_ratio,
                 start_dw_kernel_size,
                 middle_dw_kernel_size,
                 stride,
                 middle_dw_downsample: bool = True,
                 use_layer_scale: bool = False,
                 layer_scale_init_value: float = 1e-5):
        super(UniversalInvertedBottleneck, self).__init__()
        self.start_dw_kernel_size = start_dw_kernel_size
        self.middle_dw_kernel_size = middle_dw_kernel_size

        if start_dw_kernel_size:
           self.start_dw_conv = nn.Conv2d(in_channels, in_channels, start_dw_kernel_size, 
                                          stride if not middle_dw_downsample else 1,
                                          (start_dw_kernel_size - 1) // 2,
                                          groups=in_channels, bias=False)
           self.start_dw_norm = nn.BatchNorm2d(in_channels)
        
        expand_channels = make_divisible(in_channels * expand_ratio, 8)
        self.expand_conv = nn.Conv2d(in_channels, expand_channels, 1, 1, bias=False)
        self.expand_norm = nn.BatchNorm2d(expand_channels)
        self.expand_act = nn.ReLU(inplace=True)

        if middle_dw_kernel_size:
           self.middle_dw_conv = nn.Conv2d(expand_channels, expand_channels, middle_dw_kernel_size,
                                           stride if middle_dw_downsample else 1,
                                           (middle_dw_kernel_size - 1) // 2,
                                           groups=expand_channels, bias=False)
           self.middle_dw_norm = nn.BatchNorm2d(expand_channels)
           self.middle_dw_act = nn.ReLU(inplace=True)
        
        self.proj_conv = nn.Conv2d(expand_channels, out_channels, 1, 1, bias=False)
        self.proj_norm = nn.BatchNorm2d(out_channels)

        if use_layer_scale:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), requires_grad=True)

        self.use_layer_scale = use_layer_scale
        self.identity = stride == 1 and in_channels == out_channels

    def forward(self, x):
        shortcut = x

        if self.start_dw_kernel_size:
            x = self.start_dw_conv(x)
            x = self.start_dw_norm(x)

        x = self.expand_conv(x)
        x = self.expand_norm(x)
        x = self.expand_act(x)

        if self.middle_dw_kernel_size:
            x = self.middle_dw_conv(x)
            x = self.middle_dw_norm(x)
            x = self.middle_dw_act(x)

        x = self.proj_conv(x)
        x = self.proj_norm(x)

        if self.use_layer_scale:
            x = self.gamma * x

        return x + shortcut if self.identity else x


class MobileNetV4(nn.Module):
    def __init__(self, block_specs, num_classes=1000):
        super(MobileNetV4, self).__init__()

        c = 3
        layers = []
        for block_type, *block_cfg in block_specs:
            if block_type == 'conv_bn':
                block = ConvBN
                k, s, f = block_cfg
                layers.append(block(c, f, k, s))
            elif block_type == 'uib':
                block = UniversalInvertedBottleneck
                start_k, middle_k, s, f, e = block_cfg
                layers.append(block(c, f, e, start_k, middle_k, s))
            else:
                raise NotImplementedError
            c = f
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        hidden_channels = 1280
        self.conv = ConvBN(c, hidden_channels, 1)
        self.classifier = nn.Linear(hidden_channels, num_classes)

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def mobilenetv4_conv_small(**kwargs):
    """
    Constructs a MobileNetV4-Conv-Small model
    """
    block_specs = [
        # conv_bn, kernel_size, stride, out_channels
        # uib, start_dw_kernel_size, middle_dw_kernel_size, stride, out_channels, expand_ratio
        # 112px
        ('conv_bn', 3, 2, 32),
        # 56px
        ('conv_bn', 3, 2, 32),
        ('conv_bn', 1, 1, 32),
        # 28px
        ('conv_bn', 96, 3, 2),
        ('conv_bn', 64, 1, 1),
        # 14px
        ('uib', 5, 5, 2, 96, 3.0),  # ExtraDW
        ('uib', 0, 3, 1, 96, 2.0),  # IB
        ('uib', 0, 3, 1, 96, 2.0),  # IB
        ('uib', 0, 3, 1, 96, 2.0),  # IB
        ('uib', 0, 3, 1, 96, 2.0),  # IB
        ('uib', 3, 0, 1, 96, 4.0),  # ConvNext
        # 7px
        ('uib', 3, 3, 2, 128, 6.0),  # ExtraDW
        ('uib', 5, 5, 1, 128, 4.0),  # ExtraDW
        ('uib', 0, 5, 1, 128, 4.0),  # IB
        ('uib', 0, 5, 1, 128, 3.0),  # IB
        ('uib', 0, 3, 1, 128, 4.0),  # IB
        ('uib', 0, 3, 1, 128, 4.0),  # IB
        ('conv_bn', 1, 1, 960),  # Conv
    ]
    return MobileNetV4(block_specs, **kwargs)


def mobilenetv4_conv_medium(**kwargs):
    """
    Constructs a MobileNetV4-Conv-Medium model
    """
    block_specs = [
        ('conv_bn', 3, 2, 32),
        ('conv_bn', 3, 2, 128),
        ('conv_bn', 1, 1, 48),
        # 3rd stage
        ('uib', 3, 5, 2, 80, 4.0),
        ('uib', 3, 3, 1, 80, 2.0),
        # 4th stage
        ('uib', 3, 5, 2, 160, 6.0),
        ('uib', 3, 3, 1, 160, 4.0),
        ('uib', 3, 3, 1, 160, 4.0),
        ('uib', 3, 5, 1, 160, 4.0),
        ('uib', 3, 3, 1, 160, 4.0),
        ('uib', 3, 0, 1, 160, 4.0),
        ('uib', 0, 0, 1, 160, 2.0),
        ('uib', 3, 0, 1, 160, 4.0),
        # 5th stage
        ('uib', 5, 5, 2, 256, 6.0),
        ('uib', 5, 5, 1, 256, 4.0),
        ('uib', 3, 5, 1, 256, 4.0),
        ('uib', 3, 5, 1, 256, 4.0),
        ('uib', 0, 0, 1, 256, 4.0),
        ('uib', 3, 0, 1, 256, 4.0),
        ('uib', 3, 5, 1, 256, 2.0),
        ('uib', 5, 5, 1, 256, 4.0),
        ('uib', 0, 0, 1, 256, 4.0),
        ('uib', 0, 0, 1, 256, 4.0),
        ('uib', 5, 0, 1, 256, 2.0),
        # FC layers
        ('conv_bn', 1, 1, 960),
    ]

    return MobileNetV4(block_specs, **kwargs)


def mobilenetv4_conv_large(**kwargs):
    """
    Constructs a MobileNetV4-Conv-Large model
    """
    block_specs = [
        ('conv_bn', 3, 2, 24),
        ('conv_bn', 3, 2, 96),
        ('conv_bn', 1, 1, 48),
        ('uib', 3, 5, 2, 96, 4.0),
        ('uib', 3, 3, 1, 96, 4.0),
        ('uib', 3, 5, 2, 192, 4.0),
        ('uib', 3, 3, 1, 192, 4.0),
        ('uib', 3, 3, 1, 192, 4.0),
        ('uib', 3, 3, 1, 192, 4.0),
        ('uib', 3, 5, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 5, 3, 1, 192, 4.0),
        ('uib', 3, 0, 1, 192, 4.0),
        ('uib', 5, 5, 2, 512, 4.0),
        ('uib', 5, 5, 1, 512, 4.0),
        ('uib', 5, 5, 1, 512, 4.0),
        ('uib', 5, 5, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 3, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 3, 1, 512, 4.0),
        ('uib', 5, 5, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('uib', 5, 0, 1, 512, 4.0),
        ('conv_bn', 1, 1, 960),
    ]

    return MobileNetV4(block_specs, **kwargs)
