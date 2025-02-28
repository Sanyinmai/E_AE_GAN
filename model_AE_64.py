import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torchvision.models.densenet import _DenseBlock, _Transition
import math

#使用残差
class ResidualBlock(nn.Module):
    def __init__(self, num_features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features, num_features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_features)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

class REDNetEncoder(nn.Module):
    def __init__(self, num_layers=15, num_features=64):
        super(REDNetEncoder, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        conv_layers.append(nn.Sequential(nn.Conv2d(1, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(ResidualBlock(num_features))

        self.conv_layers = nn.Sequential(*conv_layers)

    def forward(self, x):
        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 4 == 0 and len(conv_feats) < ((self.num_layers + 1) // 4):
                conv_feats.append(x)
        return x, conv_feats

class REDNetDecoder(nn.Module):
    def __init__(self, num_layers=15, num_features=64):
        super(REDNetDecoder, self).__init__()
        self.num_layers = num_layers

        deconv_layers = []
        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, 1, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, conv_feats):
        conv_feats_idx = 0
        num_conv_feats = len(conv_feats)

        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1) % 4 == 0 and conv_feats_idx < num_conv_feats:
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()#输入张量尺寸：256*256*1
        self.main = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1, bias=False),#128,128,32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),  # 64,64,64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 32,32,128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 16,16,256
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 8,8,512
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),  # 4,4,1024
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Conv2d(1024, 2048, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(2048),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input)



# class DenseUNetEncoder(nn.Module):
#     def __init__(self, growth_rate=4, block_config=(2, 2, 2, 2), num_init_features=8, bn_size=2, drop_rate=0):
#         super(DenseUNetEncoder, self).__init__()
#
#         self.skip_connections = []
#
#         self.features = nn.Sequential(OrderedDict([
#             ('conv0', nn.Conv2d(1, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
#             ('norm0', nn.BatchNorm2d(num_init_features)),
#             ('relu0', nn.ReLU(inplace=True)),
#             ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=False))
#         ]))
#
#         num_features = num_init_features
#         for i, num_layers in enumerate(block_config):
#             block = _DenseBlock(
#                 num_layers=num_layers,
#                 num_input_features=num_features,
#                 bn_size=bn_size,
#                 growth_rate=growth_rate,
#                 drop_rate=drop_rate
#             )
#             self.features.add_module('denseblock%d' % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate
#             if i != len(block_config) - 1:
#                 trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
#                 self.features.add_module('transition%d' % (i + 1), trans)
#                 num_features = num_features // 2
#
#         self.features.add_module('norm5', nn.BatchNorm2d(num_features))
#         self.num_features = num_features
#
#     def forward(self, x):
#         features = self.features[:4](x)
#         self.skip_connections.append(features)
#         for i in range(4, len(self.features) - 1):
#             features = self.features[i](features)
#             if isinstance(self.features[i], _Transition):
#                 self.skip_connections.append(features)
#         features = self.features[-1](features)
#         self.skip_connections = [skip.detach() for skip in self.skip_connections]
#         return features, self.skip_connections
#
#
# class TransitionUp(nn.Sequential):
#     def __init__(self, num_input_features, num_output_features, skip_connections):
#         super(TransitionUp, self).__init__()
#         self.add_module('norm1', nn.BatchNorm2d(num_input_features))
#         self.add_module('relu1', nn.ReLU(inplace=True))
#         self.add_module('conv1',
#                         nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
#         self.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
#         self.add_module('norm2', nn.BatchNorm2d(num_output_features))
#         self.add_module('relu2', nn.ReLU(inplace=True))
#         self.add_module('conv2', nn.Conv2d(num_output_features, num_output_features, kernel_size=3, stride=1, padding=1,
#                                            bias=False))
#
#         self.skip_connections = skip_connections
#
#     def center_crop(self, larger_tensor, target_tensor):
#         _, _, target_height, target_width = target_tensor.size()
#         _, _, height, width = larger_tensor.size()
#         start_x = (width - target_width) // 2
#         start_y = (height - target_height) // 2
#         return larger_tensor[:, :, start_y:start_y + target_height, start_x:start_x + target_width]
#
#     def forward(self, x):
#         x = super(TransitionUp, self).forward(x)
#         if len(self.skip_connections) > 0:
#             with torch.no_grad():
#                 skip = self.skip_connections.pop()
#             skip = self.center_crop(skip, x)
#             x = torch.cat([x, skip], 1)
#         return x
#
# class DenseUNetDecoder(nn.Module):
#     def __init__(self, growth_rate=4, block_config=(2, 2, 2, 2), num_init_features=16, bn_size=2, drop_rate=0,
#                  skip_connections=None):
#         super(DenseUNetDecoder, self).__init__()
#
#         if skip_connections is None:
#             skip_connections = []
#
#         num_features = num_init_features
#         block_config = block_config[::-1]
#         num_features_list = [num_features + sum(block_config[:i]) * growth_rate for i in
#                              range(1, len(block_config) + 1)][::-1]
#
#         self.features = nn.Sequential()
#
#         for i in range(len(block_config)):
#             num_layers = block_config[i]
#             trans_up = TransitionUp(num_features, num_features_list[i], skip_connections)
#             self.features.add_module('transitionup%d' % (i + 1), trans_up)
#             num_features = num_features_list[i]
#             block = _DenseBlock(
#                 num_layers=num_layers,
#                 num_input_features=num_features,
#                 bn_size=bn_size,
#                 growth_rate=growth_rate,
#                 drop_rate=drop_rate
#             )
#             self.features.add_module('denseblock%d' % (i + 1), block)
#             num_features = num_features + num_layers * growth_rate
#
#         final_upsample = TransitionUp(num_features, num_features // 2, skip_connections)
#         self.features.add_module('final_upsample', final_upsample)
#         num_features = num_features // 2
#
#         self.features.add_module('final_norm', nn.BatchNorm2d(num_features))
#         self.features.add_module('final_relu', nn.ReLU(inplace=True))
#         self.features.add_module('final_conv', nn.Conv2d(num_features, 1, kernel_size=1, stride=1, bias=False))
#
#     def forward(self, x):
#         output = self.features(x)
#         self.skip_connections.clear()  # 清理 skip_connections 避免累积
#         return output

