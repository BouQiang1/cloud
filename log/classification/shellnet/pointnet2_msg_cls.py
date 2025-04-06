
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction
import numpy as np
import torch

"""
epoch:200
batch_size:24
learning_rate:0.001
optimizer:Adam
    args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(
                classifier.parameters(),
                lr=args.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=args.decay_rate
            )
decay_rate:1e-4
scheduler: StepLR
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
"""

class get_model(nn.Module):
    def __init__(self,args):
        super(get_model, self).__init__()
        in_channel = 3 if args.use_normals else 0
        self.k = args.cls_num_category
        self.normal_channel = args.use_normals
        self.sa1 = PointNetSetAbstractionMsg(512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(128, [0.2, 0.4, 0.8], [32, 64, 128], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(256, self.k)

    def forward(self, xyz):

        xyz = xyz.transpose(2, 1)
        B, C, N = xyz.shape

        if self.normal_channel:
            xyz = xyz[:, :3, :]
            norm = xyz[:, 3:, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)#

        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x,l3_points

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
    def forward(self, pred, target,_):
        total_loss = self.loss(pred, target.view(-1))
        # total_loss = F.nll_loss(pred, target)
        return total_loss
