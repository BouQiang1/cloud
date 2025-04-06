import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetFeaturePropagation
import numpy as np


class get_model(nn.Module):
    def __init__(self, num_classes):
        super(get_model, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(1024, [0.05, 0.1], [16, 32], 9, [[16, 16, 32], [32, 32, 64]])
        self.sa2 = PointNetSetAbstractionMsg(256, [0.1, 0.2], [16, 32], 32 + 64, [[64, 64, 128], [64, 96, 128]])
        self.sa3 = PointNetSetAbstractionMsg(64, [0.2, 0.4], [16, 32], 128 + 128, [[128, 196, 256], [128, 196, 256]])
        self.sa4 = PointNetSetAbstractionMsg(16, [0.4, 0.8], [16, 32], 256 + 256, [[256, 256, 512], [256, 384, 512]])

        self.fp4 = PointNetFeaturePropagation(512 + 512 + 256 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(128 + 128 + 256, [256, 256])
        self.fp2 = PointNetFeaturePropagation(32 + 64 + 256, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])

        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        xyz = xyz.transpose(2, 1)  # B , C, N
        l0_points = xyz
        l0_xyz = xyz[:, :3, :]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)  # B, C, N   ->  B, 1024, 16

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat, weight):
        # pred [131072,13]
        # target [131072]
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss


def read_bin_file(file_path):
    # 每个点包含 x,y,z,R,G,B,label（7个 float32 特征）
    pointcloud = np.fromfile(file_path, dtype=np.float32).reshape(-1, 7)
    xyz = pointcloud[:, :3]
    rgb = pointcloud[:, 3:6]
    label = pointcloud[:, 6]

    # 处理无效值
    xyz = np.nan_to_num(xyz)
    rgb = np.nan_to_num(rgb)
    label = np.nan_to_num(label)

    label = label.astype(np.int64)  # 标签需转为整数
    rgb = rgb / 255.0
    return xyz  # 或 return np.hstack([xyz, rgb])


def preprocess(pointcloud):
    """
    数据预处理
    :param pointcloud: 点云数据，形状为 (N, C)
    :return: 预处理后的点云数据，形状为 (1, C, N)
    """
    # 归一化等操作可以根据实际情况添加
    pointcloud = pointcloud.T  # (N, C) -> (C, N)
    pointcloud = np.expand_dims(pointcloud, axis=0)  # (C, N) -> (1, C, N)
    pointcloud = torch.tensor(pointcloud, dtype=torch.float32)
    return pointcloud


if __name__ == '__main__':
    # 加载模型
    model = get_model(13)
    # 读取.bin文件
    bin_file_path = '../10.bin'
    pointcloud = read_bin_file(bin_file_path)
    # 数据预处理
    input_data = preprocess(pointcloud)
    # 模型推理
    with torch.no_grad():
        output, _ = model(input_data)
    # 获取预测结果
    predicted_labels = torch.argmax(output, dim=2).squeeze().numpy()
    print("预测结果形状:", predicted_labels.shape)
    print("预测结果:", predicted_labels)