from functools import partial
from typing import Any, Callable, List, Optional, Type, Union
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models.resnet import BasicBlock

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1,
            output_padding: int = 0) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        output_padding=output_padding,
        groups=groups,
        bias=False,
        dilation=dilation,
    )  # 尺寸不变

def conv1x1(in_planes: int, out_planes: int, stride: int = 1, output_padding: int = 0) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                              output_padding=output_padding)

class Bottleneck_de(nn.Module):
    expansion: int = 2

    def __init__(
            self,
            inplanes: int,
            planes: int,
            stride: int = 1,
            output_padding: int = 0,
            upsample: Optional[nn.Module] = None,
            groups: int = 1,
            base_width: int = 32,
            dilation: int = 1,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 32.0)) * groups
        # Both self.conv2 and self.upsample layers upsample the input when stride != 1
        self.conv3 = conv1x1(planes * self.expansion, planes)
        self.bn3 = norm_layer(planes)

        self.conv2 = conv3x3(planes, planes, stride, groups, dilation, output_padding)
        self.bn2 = norm_layer(planes)

        self.conv1 = conv1x1(planes, inplanes)
        self.bn1 = norm_layer(inplanes)

        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv3(x)
        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.bn1(out)
        if self.upsample is not None:
            identity = self.upsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet_Decoder(nn.Module):
    def __init__(
            self,
            block,
            layers: List[int],
            zero_init_residual: bool = False,
            groups: int = 1,
            width_per_group: int = 32,
            norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 1024
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        self.de_conv1 = nn.Conv2d(64, 1, kernel_size=7, stride=2, padding=3, bias=False)
        self.de_conv2 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False)

        self.unpool = nn.MaxUnpool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = norm_layer(1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.unsample = nn.Upsample(size=16, mode='nearest')

        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)  # output_padding=0, last_block_dim=64

    def _make_layer(
            self,
            block,
            planes: int,
            blocks: int,
            stride: int = 2,
            output_padding: int = 1,
            last_block_dim: int = 0,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        upsample = None
        previous_dilation = self.dilation

        layers = []
        self.inplanes = planes * block.expansion
        if last_block_dim == 0:
            last_block_dim = self.inplanes // 2
        if stride != 1 or self.inplanes != planes * block.expansion or output_padding == 0:
            upsample = nn.Sequential(
                conv1x1(planes * block.expansion, last_block_dim, stride, output_padding),
                norm_layer(last_block_dim),
            )
        last_block = block(
            last_block_dim, planes, stride, output_padding, upsample, self.groups, self.base_width, previous_dilation,
            norm_layer
        )

        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer
                )
            )
        layers.append(last_block)
        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor, indices) -> Tensor:
        # See note [TorchScript super()]
        x = self.unsample(x)  # 4 1024 16 16
        x = self.layer4(x)  # 4 512 32 32
        x = self.layer3(x)  # 4 256 64 64
        x = self.layer2(x)  # 4 128 128 128
        x = self.layer1(x)  # 4 64 256 256

        x = self.de_conv2(x)  # 4 64 512 512
        x = self.de_conv1(x)  # 4 1 256 256
        x = self.bn1(x)
        x = self.relu(x)
        return x

    def forward(self, x: Tensor, indices=None) -> Tensor:
        return self._forward_impl(x, indices)


    # def _forward_cnns_only(self, x: Tensor) -> Tensor:
    #     x = self.unsample(x)
    #     # print(f'Afunsample{x.shape}')
    #     x = self.layer4(x)
    #     # print(f'Aflayer4{x.shape}')
    #     x = self.layer3(x)
    #     # print(f'Aflayer3{x.shape}')
    #     x = self.layer2(x)
    #     x = self.layer1(x)
    #     return x