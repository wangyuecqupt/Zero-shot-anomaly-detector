import glob
import copy
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import kaiming_normal_, constant_
from util import *
from torch.nn import init


class Net(nn.Module):
    def __init__(self, Norm='instance'):
        super(Net, self).__init__()

        self.Norm = Norm
        self.conv1 = conv(self.Norm, 6, 64, kernel_size=15, stride=2)
        self.conv2 = conv(self.Norm, 64, 128, kernel_size=11, stride=2)
        self.conv3 = conv(self.Norm, 128, 256, kernel_size=7, stride=2)
        self.conv3_1 = conv(self.Norm, 256, 256)
        self.conv4 = conv(self.Norm, 256, 512, stride=2)
        self.conv4_1 = conv(self.Norm, 512, 512)
        self.conv5 = conv(self.Norm, 512, 512, stride=2)
        self.conv5_1 = conv(self.Norm, 512, 512)
        self.conv6 = conv(self.Norm, 512, 1024, stride=2)
        self.conv6_1 = conv(self.Norm, 1024, 1024)

        self.deconv5 = deconv(1024, 512)
        self.deconv4 = deconv(1025, 256)
        self.deconv3 = deconv(769, 128)
        self.deconv2 = deconv(385, 64)

        self.predict_mask6 = predict_mask(1024)
        self.predict_mask5 = predict_mask(1025)
        self.predict_mask4 = predict_mask(769)
        self.predict_mask3 = predict_mask(385)
        self.predict_mask2 = predict_mask(193)

        self.upsampled_mask6_to_5 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_mask5_to_4 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_mask4_to_3_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_mask3_to_2 = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.upsampled_mask2_to_1 = nn.ConvTranspose2d(1, 1, 8, 4, 1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight, 0.1)
                if m.bias is not None:
                    constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_(m.weight, 1)
                constant_(m.bias, 0)

    def forward(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        mask6 = self.predict_mask6(out_conv6)
        mask6_up = crop_like(self.upsampled_mask6_to_5(mask6), out_conv5)
        out_deconv5 = crop_like(self.deconv5(out_conv6), out_conv5)

        concat5 = torch.cat((out_conv5, out_deconv5, mask6_up), 1)
        mask5 = self.predict_mask5(concat5)
        mask5_up = crop_like(self.upsampled_mask5_to_4(mask5), out_conv4)
        out_deconv4 = crop_like(self.deconv4(concat5), out_conv4)

        concat4 = torch.cat((out_conv4, out_deconv4, mask5_up), 1)
        mask4 = self.predict_mask4(concat4)
        mask4_up = crop_like(self.upsampled_mask4_to_3_2(mask4), out_conv3)
        out_deconv3 = crop_like(self.deconv3(concat4), out_conv3)

        concat3 = torch.cat((out_conv3, out_deconv3, mask4_up), 1)
        mask3 = self.predict_mask3(concat3)
        mask3_up = crop_like(self.upsampled_mask3_to_2(mask3), out_conv2)
        out_deconv2 = crop_like(self.deconv2(concat3), out_conv2)

        concat2 = torch.cat((out_conv2, out_deconv2, mask3_up), 1)
        mask2 = self.predict_mask2(concat2)
        mask2_up = crop_like(self.upsampled_mask2_to_1(mask2), x)

        if self.training:
            return mask2_up
        else:
            return nn.Sigmoid()(mask2_up)

        # if self.training:
        #     return mask2_up
        # else:
        #     return mask2_up.argmax(dim=1)

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def other_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' not in name and 'weight' not in name]