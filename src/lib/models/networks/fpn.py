from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from .msra_resnet import get_pose_net

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class resnet_fpn(nn.Module):
  def __init__(self, block, layers, heads, head_conv, pretrained=False, class_agnostic=False, **kwargs):
    self.model_path = 'https://download.pytorch.org/models/resnet50-19c8e357.pth'
    self.dout_base_model = 256
    self.pretrained = pretrained
    self.class_agnostic = class_agnostic
    self.block = block
    self.layers = layers
    self.heads = heads
    self.head_conv = head_conv

    super(resnet_fpn, self).__init__()
    #_FPN.__init__(self, classes, class_agnostic)

  def _init_modules(self):
    resnet = get_pose_net(num_layers=self.layers, heads=self.heads, head_conv=self.head_conv)

    if self.pretrained == True:
      print("Loading pretrained weights from %s" %(self.model_path))
      state_dict = torch.load(self.model_path)
      resnet.load_state_dict({k:v for k,v in state_dict.items() if k in resnet.state_dict()})

    self.RCNN_layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    self.RCNN_layer1 = nn.Sequential(resnet.layer1)
    self.RCNN_layer2 = nn.Sequential(resnet.layer2)
    self.RCNN_layer3 = nn.Sequential(resnet.layer3)
    self.RCNN_layer4 = nn.Sequential(resnet.layer4)

    # Top layer
    self.RCNN_toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # reduce channel

    # Smooth layers
    self.RCNN_smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.RCNN_smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.RCNN_smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    # Lateral layers
    self.RCNN_latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
    self.RCNN_latlayer2 = nn.Conv2d( 512, 256, kernel_size=1, stride=1, padding=0)
    self.RCNN_latlayer3 = nn.Conv2d( 256, 256, kernel_size=1, stride=1, padding=0)

    # ROI Pool feature downsampling
    self.RCNN_roi_feat_ds = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)

    self.RCNN_top = nn.Sequential(
      nn.Conv2d(256, 1024, kernel_size=7, stride=7, padding=0),
      nn.ReLU(True),
      nn.Conv2d(1024, 1024, kernel_size=1, stride=1, padding=0),
      nn.ReLU(True)
      )

    self.RCNN_cls_score = nn.Linear(1024, self.n_classes)

    if self.class_agnostic:
      self.RCNN_bbox_pred = nn.Linear(1024, 4)
    else:
      self.RCNN_bbox_pred = nn.Linear(1024, 4 * self.n_classes)
    for head in sorted(self.heads):
        num_output = self.heads[head]
        if self.head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(256, self.head_conv,
                          kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.head_conv, num_output,
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

  def forward(self, im_data, im_info, gt_boxes, num_boxes):
    batch_size = im_data.size(0)

    im_info = im_info.data
    gt_boxes = gt_boxes.data
    num_boxes = num_boxes.data

    # feed image data to base model to obtain base feature map
    # Bottom-up
    c1 = self.RCNN_layer0(im_data)
    c2 = self.RCNN_layer1(c1)
    c3 = self.RCNN_layer2(c2)
    c4 = self.RCNN_layer3(c3)
    c5 = self.RCNN_layer4(c4)
    # Top-down
    p5 = self.RCNN_toplayer(c5)
    p4 = self._upsample_add(p5, self.RCNN_latlayer1(c4))
    p4 = self.RCNN_smooth1(p4)
    p3 = self._upsample_add(p4, self.RCNN_latlayer2(c3))
    p3 = self.RCNN_smooth2(p3)
    p2 = self._upsample_add(p3, self.RCNN_latlayer3(c2))
    p2 = self.RCNN_smooth3(p2)

    p6 = self.maxpool2d(p5)

    rpn_feature_maps = [p2, p3, p4, p5, p6]
    mrcnn_feature_maps = [p2, p3, p4, p5]