import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import torch
from torch import nn
from torch.nn import init

class External_attention(nn.Module):

    def __init__(self, C,S=64):
        super().__init__()
        self.mk=nn.Conv1d(C, S, 1, bias=False)
        self.mv=nn.Conv1d(S,C,1,bias=False)
        self.softmax=nn.Softmax(dim=1)
        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.view(b, c, h * w)
        attn=self.mk(x) #bs,n,S
        attn=self.softmax(attn) #bs,n,S
        attn=attn/torch.sum(attn,dim=2,keepdim=True) #bs,n,S
        out=self.mv(attn) #bs,n,d_model
        out = out.view(b, c, h, w)
        return out




#
# class External_attention(nn.Module):
#     '''
#     Arguments:
#         c (int): The input and output channel number. set 512
#     '''
#     def __init__(self, c):
#         super(External_attention, self).__init__()
#         self.conv1 = nn.Conv2d(c, c, 1)
#         self.k = 64
#         self.linear_0 = nn.Conv1d(c, self.k, 1, bias=False)
#
#         self.linear_1 = nn.Conv1d(self.k, c, 1, bias=False)
#         self.linear_1.weight.data = self.linear_0.weight.data.permute(1, 0, 2)
#
#         self.conv2 = nn.Sequential(
#             nn.Conv2d(c, c, 1, bias=False),
#             nn.LayerNorm(c))
#
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.Conv1d):
#                 n = m.kernel_size[0] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#
#     def forward(self, x):
#         idn = x
#         x = self.conv1(x)
#
#         b, c, h, w = x.size()
#         x = x.view(b, c, h * w)  # b * c * n
#
#         attn = self.linear_0(x)  # b, k, n
#
#         # linear_0是第一个memory unit
#         attn = F.softmax(attn, dim=-1)  # b, k, n
#
#         attn = attn / (1e-9 + attn.sum(dim=1, keepdim=True))  # # b, k, n
#
#         x = self.linear_1(attn)  # b, c, n
#
#         # linear_1是第二个memory unit
#         x = x.view(b, c, h, w)
#         x = self.conv2(x)
#
#         x = x + idn
#         x = F.relu(x)
#         return x