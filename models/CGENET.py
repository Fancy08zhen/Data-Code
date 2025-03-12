import math
from collections import OrderedDict
from typing import Optional, Callable
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F


def _make_divisible(ch, divisor=8, min_ch=None):
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConvBNActivation(nn.Sequential):
    def __init__(self,
                 in_planes: int,
                 out_planes: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):
        padding = (kernel_size - 1) // 2
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.SiLU  # alias Swish  (torch>=1.7)

        super(ConvBNActivation, self).__init__(nn.Conv2d(in_channels=in_planes,
                                                         out_channels=out_planes,
                                                         kernel_size=kernel_size,
                                                         stride=stride,
                                                         padding=padding,
                                                         groups=groups,
                                                         bias=False),
                                               norm_layer(out_planes),
                                               activation_layer())


class SqueezeExcitation(nn.Module):
    def __init__(self,
                 input_c: int,   # block input channel
                 expand_c: int,  # block expand channel
                 squeeze_factor: int = 4):
        super(SqueezeExcitation, self).__init__()
        squeeze_c = input_c // squeeze_factor
        self.fc1 = nn.Conv2d(expand_c, squeeze_c, 1)
        self.ac1 = nn.SiLU()  # alias Swish
        self.fc2 = nn.Conv2d(squeeze_c, expand_c, 1)
        self.ac2 = nn.Sigmoid()

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = self.ac1(scale)
        scale = self.fc2(scale)
        scale = self.ac2(scale)
        return scale * x


class InvertedResidualConfig:
    # kernel_size, in_channel, out_channel, exp_ratio, strides, use_SE, drop_connect_rate
    def __init__(self,
                 kernel: int,          # 3 or 5
                 input_c: int,
                 out_c: int,
                 expanded_ratio: int,  # 1 or 6
                 stride: int,          # 1 or 2
                 use_se: bool,         # True
                 drop_rate: float,
                 index: str,           # 1a, 2a, 2b, ...
                 width_coefficient: float):
        self.input_c = self.adjust_channels(input_c, width_coefficient)
        self.kernel = kernel
        self.expanded_c = self.input_c * expanded_ratio
        self.out_c = self.adjust_channels(out_c, width_coefficient)
        self.use_se = use_se
        self.stride = stride
        self.drop_rate = drop_rate
        self.index = index

    @staticmethod
    def adjust_channels(channels: int, width_coefficient: float):
        return _make_divisible(channels * width_coefficient, 8)


class InvertedResidual(nn.Module):
    def __init__(self,
                 cnf: InvertedResidualConfig,
                 norm_layer: Callable[..., nn.Module]):
        super(InvertedResidual, self).__init__()

        if cnf.stride not in [1, 2]:
            raise ValueError("illegal stride value.")

        self.use_res_connect = (cnf.stride == 1 and cnf.input_c == cnf.out_c)

        layers = OrderedDict()
        activation_layer = nn.SiLU  # alias Swish

        # expand
        if cnf.expanded_c != cnf.input_c:
            layers.update({"expand_conv": ConvBNActivation(cnf.input_c,
                                                           cnf.expanded_c,
                                                           kernel_size=1,
                                                           norm_layer=norm_layer,
                                                           activation_layer=activation_layer)})

        # depthwise
        layers.update({"dwconv": ConvBNActivation(cnf.expanded_c,
                                                  cnf.expanded_c,
                                                  kernel_size=cnf.kernel,
                                                  stride=cnf.stride,
                                                  groups=cnf.expanded_c,
                                                  norm_layer=norm_layer,
                                                  activation_layer=activation_layer)})

        if cnf.use_se:
            layers.update({"se": SqueezeExcitation(cnf.input_c,
                                                   cnf.expanded_c)})

        # project
        layers.update({"project_conv": ConvBNActivation(cnf.expanded_c,
                                                        cnf.out_c,
                                                        kernel_size=1,
                                                        norm_layer=norm_layer,
                                                        activation_layer=nn.Identity)})

        self.block = nn.Sequential(layers)
        self.out_channels = cnf.out_c
        self.is_strided = cnf.stride > 1

        # 只有在使用shortcut连接时才使用dropout层
        if self.use_res_connect and cnf.drop_rate > 0:
            self.dropout = DropPath(cnf.drop_rate)
        else:
            self.dropout = nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        result = self.block(x)
        result = self.dropout(result)
        if self.use_res_connect:
            result += x

        return result


class ECAAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        t = t if t % 2 else t + 1  # t should be odd
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=t, padding=t // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channels)
        y = self.conv(y.unsqueeze(1)).view(batch_size, channels)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y


class GhostBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expansion=6, use_eca=False, kernel_size=3):
        super(GhostBottleneck, self).__init__()
        assert stride in [1, 2] and kernel_size in [3, 5]

        mid_channels = int(out_channels / expansion)

        # Primary 1x1 convolution
        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # Ghost module with depthwise convolution
        self.ghost_module = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2,
                      groups=mid_channels, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True)
        )

        # ECA module (optional)
        self.eca = ECAAttention(mid_channels) if use_eca else nn.Identity()

        # Pointwise 1x1 convolution to project back to out_channels
        self.pointwise_conv = nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

        # Batch normalization for the output
        self.bn = nn.BatchNorm2d(out_channels)

        # Shortcut for residual connection
        self.shortcut = self._make_shortcut(in_channels, out_channels, stride)

    def _make_shortcut(self, in_channels, out_channels, stride):
        if stride == 1 and in_channels == out_channels:
            return nn.Identity()  # Use nn.Identity instead of nn.Sequential() for no-op
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels,
                          bias=False),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        # Primary convolution
        x = self.primary_conv(x)

        # Ghost module
        x = self.ghost_module(x)

        # ECA module (if used)
        x = self.eca(x)

        # Pointwise convolution and batch normalization
        x = self.bn(self.pointwise_conv(x))

        # Residual connection
        x += self.shortcut(residual)

        return x


class CPCA_ChannelAttention(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(CPCA_ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1,
                             bias=True)
        self.fc2 = nn.Conv2d(in_channels=internal_neurons, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=True)
        self.input_channels = input_channels

    def forward(self, inputs):
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)
        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1 + x2
        x = x.view(-1, self.input_channels, 1, 1)
        return inputs * x


class CPCA(nn.Module):
    def __init__(self, channels, channelAttention_reduce=4):
        super().__init__()

        self.ca = CPCA_ChannelAttention(input_channels=channels, internal_neurons=channels // channelAttention_reduce)
        self.dconv5_5 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=channels)
        self.dconv1_7 = nn.Conv2d(channels, channels, kernel_size=(1, 7), padding=(0, 3), groups=channels)
        self.dconv7_1 = nn.Conv2d(channels, channels, kernel_size=(7, 1), padding=(3, 0), groups=channels)
        self.dconv1_11 = nn.Conv2d(channels, channels, kernel_size=(1, 11), padding=(0, 5), groups=channels)
        self.dconv11_1 = nn.Conv2d(channels, channels, kernel_size=(11, 1), padding=(5, 0), groups=channels)
        self.dconv1_21 = nn.Conv2d(channels, channels, kernel_size=(1, 21), padding=(0, 10), groups=channels)
        self.dconv21_1 = nn.Conv2d(channels, channels, kernel_size=(21, 1), padding=(10, 0), groups=channels)
        self.conv = nn.Conv2d(channels, channels, kernel_size=(1, 1), padding=0)
        self.act = nn.GELU()

    def forward(self, inputs):
        #   Global Perceptron
        inputs = self.conv(inputs)
        inputs = self.act(inputs)

        inputs = self.ca(inputs)

        x_init = self.dconv5_5(inputs)
        x_1 = self.dconv1_7(x_init)
        x_1 = self.dconv7_1(x_1)
        x_2 = self.dconv1_11(x_init)
        x_2 = self.dconv11_1(x_2)
        x_3 = self.dconv1_21(x_init)
        x_3 = self.dconv21_1(x_3)
        x = x_1 + x_2 + x_3 + x_init
        spatial_att = self.conv(x)
        out = spatial_att * inputs
        out = self.conv(out)
        return out


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1000, width_coefficient=1.0, depth_coefficient=1.0, dropout_rate=0.2):
        super(EfficientNet, self).__init__()
        self.width_coefficient = width_coefficient
        self.depth_coefficient = depth_coefficient

        # Stem layers
        self.stem = nn.Sequential(
            nn.Conv2d(3, self.adjust_channels(32), kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),
            *self._make_CPCA_layer(num_blocks=1, channels=32, channelAttention_reduce=4),
            *self._make_mbconv_layer(32, 16, num_blocks=1, kernel_size=3, stride=1, expand_ratio=1, se_ratio=0.25),
            *self._make_mbconv_layer(16, 24, num_blocks=2, kernel_size=3, stride=2, expand_ratio=6, se_ratio=0.25),
            *self._make_ghost_layer(24, 40, num_blocks=2, kernel_size=5, stride=2,  expand_ratio=6, eca_ratio=0.25),
            *self._make_ghost_layer(40, 80, num_blocks=3, kernel_size=3, stride=2,  expand_ratio=6, eca_ratio=0.25),
            *self._make_ghost_layer(80, 112, num_blocks=3, kernel_size=5, stride=1,  expand_ratio=6, eca_ratio=0.25),
            *self._make_mbconv_layer(112, 192, num_blocks=4, kernel_size=5, stride=2, expand_ratio=6, se_ratio=0.25),
            *self._make_mbconv_layer(192, 320, num_blocks=1, kernel_size=3, stride=1, expand_ratio=6, se_ratio=0.25),
        )
        # Head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity(),
            nn.Linear(self.adjust_channels(320), num_classes)
        )
        # Initialize weights
        self._initialize_weights()

    def adjust_channels(self, channels):
        return int(math.ceil(channels * self.width_coefficient / 8) * 8)

    def _make_ghost_layer(self, in_channels, out_channels, num_blocks, kernel_size, stride, eca_ratio,expand_ratio):
        layers = []
        for i in range(num_blocks):
            layers.append(GhostBottleneck(
                in_channels=in_channels if i == 0 else out_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride if i == 0 else 1,
                expansion=expand_ratio,
                use_eca=(eca_ratio > 0)
            ))
        return layers

    def _make_mbconv_layer(self, in_channels, out_channels, num_blocks, kernel_size, stride, expand_ratio, se_ratio):
        layers = []
        for i in range(num_blocks):
            cnf = InvertedResidualConfig(
                kernel=kernel_size,
                input_c=in_channels if i == 0 else out_channels,
                out_c=out_channels,
                expanded_ratio=expand_ratio,
                stride=stride if i == 0 else 1,
                use_se=(se_ratio > 0),
                drop_rate=0.0,  # 根据需要设置
                index=f"{i + 1}{chr(97 + (i // num_blocks))}",  # 生成索引
                width_coefficient=self.width_coefficient
            )
            layers.append(InvertedResidual(cnf, norm_layer=nn.BatchNorm2d))
        return layers

    def _make_CPCA_layer(self, num_blocks, channels, channelAttention_reduce):
        layers = []
        for i in range(num_blocks):
            layers.append(CPCA(
                channels=channels,
                channelAttention_reduce=channelAttention_reduce
            ))
        return layers

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.head(x)
        return x


