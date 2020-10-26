# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock_back(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class BasicBlock(nn.Module):
    # BasicBlock_ACBlock
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        #self.conv1 = conv3x3(inplanes, planes, stride)
        #self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        #self.conv2 = conv3x3(planes, planes)
        #self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride
        self.acblock1 = ACBlock(inplanes, planes, stride=self.stride, padding=1, kernel_size=3, deploy=False)
        self.acblock2 = ACBlock(planes, planes, padding=1, kernel_size=3, deploy=False)


    def forward(self, x):
        residual = x

        out = self.acblock1(x)
        #out = self.bn1(out)
        out = self.relu(out)

        out = self.acblock2(out)
        #out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class PoseResNet(nn.Module):

    def __init__(self, block, layers, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # FPN
        self.fpn_c5p5 = nn.Conv2d(
            512, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c4p4 = nn.Conv2d(
            256, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c3p3 = nn.Conv2d(
            128, 256, kernel_size=1, stride=1, padding=0)
        self.fpn_c2p2 = nn.Conv2d(
            64, 256, kernel_size=1, stride=1, padding=0)

        self.fpn_p2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p4 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.fpn_p5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv_p2 = nn.Conv2d(256, 32, kernel_size=1)
        self.conv_p3 = nn.Conv2d(256, 48, kernel_size=1)
        self.conv_p4 = nn.Conv2d(256, 48, kernel_size=1)
        self.conv_p5 = nn.Conv2d(256, 64, kernel_size=1)


        # self.atten1 = ChannelAttention(256)
        # self.atten2 = ChannelAttention(256)
        # self.atten3 = ChannelAttention(256)
        # self.atten4 = ChannelAttention(256 * 4)
        self.atten = MixedAttettion(256, 3)
        self.final_conv = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)

        # used for deconv layers
        self.inplanes = 256
        self.deconv_layers1 = self._make_deconv_layer(
            1,
            [48],
            [4],
            2,
            48
        )
        self.deconv_layers2 = self._make_deconv_layer(
            1,
            [48],
            [4],
            4,
            48
        )
        self.deconv_layers3 = self._make_deconv_layer(
            1,
            [64],
            [8],
            8,
            64
        )
        # self.final_layer = []

        for head in sorted(self.heads):
          num_output = self.heads[head]
          if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(192, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output, 
                  kernel_size=1, stride=1, padding=0))
          else:
            fc = nn.Conv2d(
              in_channels=256,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          self.__setattr__(head, fc)

        # self.final_layer = nn.ModuleList(self.final_layer)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _get_deconv_cfg(self, deconv_kernel, stride, index):
        if deconv_kernel == 4:
            if stride == 2:
                padding = 1
                output_padding = 0
            else:
                padding = 0
                output_padding = 0
        elif deconv_kernel == 3:
            padding = 1
            output_padding = 1
        elif deconv_kernel == 2:
            padding = 0
            output_padding = 0
        elif deconv_kernel == 8:
            padding = 0
            output_padding = 0

        return deconv_kernel, padding, output_padding

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels, stride, initial_inplanes):
        assert num_layers == len(num_filters), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        assert num_layers == len(num_kernels), \
            'ERROR: num_deconv_layers is different len(num_deconv_filters)'
        self.inplanes = initial_inplanes
        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i], stride, i)

            planes = num_filters[i]
            layers.append(
                nn.ConvTranspose2d(
                    in_channels=self.inplanes,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=stride,
                    padding=padding,
                    output_padding=output_padding,
                    bias=self.deconv_with_bias))
            layers.append(nn.BatchNorm2d(planes, momentum=BN_MOMENTUM))
            layers.append(nn.ReLU(inplace=True))
            self.inplanes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        C1 = self.maxpool(x)

        C2 = self.layer1(C1)
        C3 = self.layer2(C2)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)
        # x = C5

        P5 = self.fpn_c5p5(C5)
        P4 = self.fpn_c4p4(C4) + F.upsample(P5,
                                            scale_factor=2, mode='bilinear')
        P3 = self.fpn_c3p3(C3) + F.upsample(P4,
                                            scale_factor=2, mode='bilinear')
        P2 = self.fpn_c2p2(C2) + F.upsample(P3,
                                            scale_factor=2, mode='bilinear')

        # Attach 3x3 conv to all P layers to get the final feature maps.
        # P2 is 256, P3 is 128, P4 is 64, P5 is 32
        P2 = self.fpn_p2(P2)
        P2 = self.conv_p2(P2)
        P3 = self.fpn_p3(P3)
        P3 = self.conv_p3(P3)
        P4 = self.fpn_p4(P4)
        P4 = self.conv_p4(P4)
        P5 = self.fpn_p5(P5)
        P5 = self.conv_p5(P5)

        # x = self.deconv_layers(x)
        # P3 = self.atten(P3)

        P3 = self.deconv_layers1(P3)
        # P4 = self.atten(P4)
        P4 = self.deconv_layers2(P4)
        # P5 = self.atten(P5)
        P5 = self.deconv_layers3(P5)
        P = [P2, P3, P4, P5]
        P = torch.cat(P, dim=1)
        final = self.final_conv(P)
        # final = self.atten(final)
        rets = []
        for i in range(1):
            ret = {}
            for head in self.heads:
                ret[head] = self.__getattr__(head)(final)
            rets.append(ret)
        return rets

    def init_weights(self, num_layers, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            for _, m in self.deconv_layers1.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.deconv_layers2.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for _, m in self.deconv_layers3.named_modules():
                if isinstance(m, nn.ConvTranspose2d):
                    nn.init.normal_(m.weight, std=0.001)
                    if self.deconv_with_bias:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
            for head in self.heads:
              final_layer = self.__getattr__(head)
              for i, m in enumerate(final_layer.modules()):
                  if isinstance(m, nn.Conv2d):
                      # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                      # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                      # print('=> init {}.bias as 0'.format(name))
                      if m.weight.shape[0] == self.heads[head]:
                          if 'hm' in head:
                              nn.init.constant_(m.bias, -2.19)
                          else:
                              nn.init.normal_(m.weight, std=0.001)
                              nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            url = model_urls['resnet{}'.format(num_layers)]
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')


class ChannelAttention(nn.Module):
    def __init__(self, C):
        super(ChannelAttention, self).__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.fc1 = nn.Linear(C, int(C / 4))
        self.fc2 = nn.Linear(int(C / 4), C)

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, kernel_size=x.size()[-1])
        avg_pool = avg_pool.permute(0, 2, 3, 1)
        fc = self.fc1(avg_pool)
        relu = self.relu(fc)
        fc = self.fc2(relu).permute(0, 3, 1, 2)
        atten = self.sigmoid(fc)
        # output = atten * x
        return atten


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class MixedAttettion(nn.Module):
    def __init__(self, C, kernel_size):
        super(MixedAttettion, self).__init__()
        self.spatial_att = SpatialAttention()
        self.channel_att = ChannelAttention(C)
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        f1 = self.channel_att(x)
        f2 = self.spatial_att(x)
        out = self.gamma1 * f1 * x + self.gamma2 * f2
        # out = self.gamma1 * f1 * x
        # out = torch.cat((f1,f2,x), dim = 1)
        return out

resnet_spec = {18: (BasicBlock, [2, 2, 2, 2]),
               34: (BasicBlock, [3, 4, 6, 3]),
               50: (Bottleneck, [3, 4, 6, 3]),
               101: (Bottleneck, [3, 4, 23, 3]),
               152: (Bottleneck, [3, 8, 36, 3])}


def get_pose_net(num_layers, heads, head_conv):
  block_class, layers = resnet_spec[num_layers]

  model = PoseResNet(block_class, layers, heads, head_conv=head_conv)
  model.init_weights(num_layers, pretrained=True)
  return model



#  convert train model to deploy model:   https://github.com/DingXiaoH/ACNet/blob/master/acnet/acnet_fusion.py

class ACBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, padding_mode='zeros', deploy=False):
        super(ACBlock, self).__init__()
        self.deploy = deploy
        if deploy:
            self.fused_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(kernel_size,kernel_size), stride=stride,
                                      padding=padding, dilation=dilation, groups=groups, bias=True)
        else:
            self.square_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                         kernel_size=(kernel_size, kernel_size), stride=stride,
                                         padding=padding, dilation=dilation, groups=groups, bias=False)
            self.square_bn = nn.BatchNorm2d(num_features=out_channels)

            center_offset_from_origin_border = padding - kernel_size // 2
            ver_pad_or_crop = (0, -1)
            hor_pad_or_crop = (-1, 0)

            ver_conv_padding = (1, 0)
            hor_conv_padding = (0, 1)
            self.ver_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 1),
                                      stride=stride, padding=ver_conv_padding, dilation=dilation, groups=groups, bias=False)

            self.hor_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(1, 3),
                                      stride=stride, padding=hor_conv_padding, dilation=dilation, groups=groups, bias=False)
            self.ver_bn = nn.BatchNorm2d(num_features=out_channels)
            self.hor_bn = nn.BatchNorm2d(num_features=out_channels)

            #self.corner_conv = four_corner_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=1)
            #self.corner_bn = nn.BatchNorm2d(num_features=out_channels)

            # self.left_diagonal_conv = left_diagonal_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3,3), stride=stride, padding=1)
            # self.left_diagonal_bn = nn.BatchNorm2d(num_features=out_channels)
            #
            # self.right_diagonal_conv = right_diagonal_conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=(3, 3), stride=stride, padding=1)
            # self.right_diagonal_bn = nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input):
        if self.deploy:
            return self.fused_conv(input)
        else:
            square_outputs = self.square_conv(input)
            square_outputs = self.square_bn(square_outputs)
            # print(square_outputs.size())
            # return square_outputs
            #vertical_outputs = self.ver_conv_crop_layer(input)
            vertical_outputs = self.ver_conv(input)
            vertical_outputs = self.ver_bn(vertical_outputs)
            # print(vertical_outputs.size())
            #horizontal_outputs = self.hor_conv_crop_layer(input)
            horizontal_outputs = self.hor_conv(input)
            horizontal_outputs = self.hor_bn(horizontal_outputs)
            # print(horizontal_outputs.size())
            #corner_outputs = self.corner_conv(input)
            #corner_outputs = self.corner_bn(corner_outputs)
            # left_diagonal_outputs = self.left_diagonal_conv(input)
            # left_diagonal_outputs = self.left_diagonal_bn(left_diagonal_outputs)
            # right_diagonal_outputs = self.right_diagonal_conv(input)
            # right_diagonal_outputs = self.right_diagonal_bn(right_diagonal_outputs)
            return square_outputs + vertical_outputs + horizontal_outputs

class CropLayer(nn.Module):

    #   E.g., (-1, 0) means this layer should crop the first and last rows of the feature map. And (0, -1) crops the first and last columns
    def __init__(self, crop_set):
        super(CropLayer, self).__init__()
        self.rows_to_crop = - crop_set[0]
        self.cols_to_crop = - crop_set[1]
        assert self.rows_to_crop >= 0
        assert self.cols_to_crop >= 0

    def forward(self, input):
        if self.rows_to_crop == 1:
            return input[:, :, 1:-1, :]
            #return input[:, :, :, 1:-1]
        elif self.cols_to_crop == 1:
            return input[:, :, :, 1:-1]
            #return input[:, :, 1:-1, :]
        #return input[:, :, self.rows_to_crop:-self.rows_to_crop, self.cols_to_crop:-self.cols_to_crop]


import torch.nn.functional as F
import torch

def four_corner(out_channels, in_channels, kernel_size):
    '''
    :return: a 3*3 kernel
    w 0 w
    0 0 0
    w 0 w
    '''
    kernel_size1, kernel_size2 = kernel_size, kernel_size
    if isinstance(kernel_size, tuple):
        kernel_size1 = kernel_size[0]
        kernel_size2 = kernel_size[1]
    new_kernel = torch.randn(out_channels, in_channels, kernel_size1, kernel_size2, requires_grad=True)
    with torch.no_grad():
        new_kernel[:, :, 0, 1] = 0.
        new_kernel[:, :, 1, 0] = 0.
        new_kernel[:, :, 1, 1] = 0.
        new_kernel[:, :, 1, 2] = 0.
        new_kernel[:, :, 2, 1] = 0.

    return new_kernel

class four_corner_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(four_corner_conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

    def forward(self, x):

        kernel = four_corner(self.out_channels, self.in_channels, self.kernel_size)
        kernel = kernel.float().to(torch.device('cuda'))   # [channel_out, channel_in, kernel, kernel]
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out

def left_diagonal(out_channels, in_channels, kernel_size):
    '''
    dui jiao xian
    :return: a 3*3 kernel
    w 0 0
    0 w 0
    0 0 w
    '''
    kernel_size1, kernel_size2 = kernel_size[0], kernel_size[1]
    # if isinstance(kernel_size, tuple):
    #     kernel_size1 = kernel_size[0]
    #     kernel_size2 = kernel_size[1]
    new_kernel = torch.randn(out_channels, in_channels, kernel_size1, kernel_size2, requires_grad=True).cuda()
    with torch.no_grad():
        new_kernel[:, :, 0, 1] = 0.
        new_kernel[:, :, 0, 2] = 0.
        new_kernel[:, :, 1, 0] = 0.
        new_kernel[:, :, 1, 2] = 0.
        new_kernel[:, :, 2, 0] = 0.
        new_kernel[:, :, 2, 1] = 0.

    return new_kernel

class left_diagonal_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(left_diagonal_conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

    def forward(self, x):

        kernel = left_diagonal(self.out_channels, self.in_channels, self.kernel_size)
        kernel = kernel.float().to(torch.device('cuda'))   # [channel_out, channel_in, kernel, kernel]
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out

def right_diagonal(out_channels, in_channels, kernel_size):
    '''
    dui jiao xian
    :return: a 3*3 kernel
    0 0 w
    0 w 0
    w 0 0
    '''
    kernel_size1, kernel_size2 = kernel_size[0], kernel_size[1]
    # if isinstance(kernel_size, tuple):
    #     kernel_size1 = kernel_size[0]
    #     kernel_size2 = kernel_size[1]
    new_kernel = torch.randn(out_channels, in_channels, kernel_size1, kernel_size2, requires_grad=True).cuda()
    with torch.no_grad():
        new_kernel[:, :, 0, 0] = 0.
        new_kernel[:, :, 0, 1] = 0.
        new_kernel[:, :, 1, 0] = 0.
        new_kernel[:, :, 1, 2] = 0.
        new_kernel[:, :, 2, 1] = 0.
        new_kernel[:, :, 2, 2] = 0.

    return new_kernel

class right_diagonal_conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(right_diagonal_conv2d, self).__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = padding

    def forward(self, x):

        kernel = right_diagonal(self.out_channels, self.in_channels, self.kernel_size)
        kernel = kernel.float().to(torch.device('cuda'))   # [channel_out, channel_in, kernel, kernel]
        out = F.conv2d(x, kernel, stride=self.stride, padding=self.padding)

        return out