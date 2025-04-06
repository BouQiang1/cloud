"""
--本代码有B站up主 "三维点云-小小武"提供,
--目前致力于学习三维点云的分类和分割、二维图像的目标检测和语义分割 以及比较前沿的模块技术。相关视频已经上传到B站，
--希望小伙伴们多多点赞支持，如果能充电支持就更好了，谢谢大家。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from time import time
import numpy as np

def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

#p归一化点云，以centroid中心，球半径1，数据预处理加快训练速度
def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))#求这个batch点云的模的最大值
    pc = pc / m
    return pc

def square_distance(src, dst):
    # square_distance确定球查询过程中每一个点距离采样
    # 函数是两组点，N为第一组点src的个数，M为第二组点dst的个数，C为通道数
    # 函数返回两组点两两之间的欧几里得距离，即N*M的矩阵
    # 在训练过程中以mini-batch的形式输入，第一个Batch数量的维度为B
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]

    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def index_points(points, idx):

    """
    用于将指定索引处的点从输入点集中提取出来，并返回经过索引后的点集
    就是根据采样/分组得到的点的坐标，在完整点集中根据这些坐标把相应的点挑出来
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

#最远点采样
#返回的结果是npoint个采样点在原始点云的索引
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    #初始化一个cenroids的矩阵，用于存储npoint个采样点的索引位置，大小为B x npoint
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    #distance矩阵(BxN)记录某个batch中所有点到某一点的距离，初始值很大，后面迭代更新
    distance = torch.ones(B, N).to(device) * 1e10
    #farthest表示当前最远的点，也是随机初始化，范围0-N 初始化B个， 每个batch都随机有个初始最远点
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    #batch_indices初始化为0-(B-1)数组
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    #直到采样点达到npoint，否则迭代
    for i in range(npoint):
        #设当前的采样点centroids为当前的最远点farthest
        centroids[:, i] = farthest
        #取出该中心点centroid的坐标
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        #求出该点到centroid的欧式距离，存在dist矩阵中
        dist = torch.sum((xyz - centroid) ** 2, -1)
        #建立一个mask，如果dist矩阵的元素小于distance保存的距离值，则更新distance中对应的值
        #随着迭代的继续，distance矩阵中的值会慢慢变小
        #相当于记录某个batch中每个点的距离所有已采样的最小距离
        mask = dist < distance
        distance[mask] = dist[mask]
        #从distance取出最远的点为farthest，继续迭代
        farthest = torch.max(distance, -1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    #query_ball_point 寻找球形领域中的点
    #
    """
    Input:
        radius: local region radius 邻域的半径
        nsample: max sample number in local region 在邻域的采样的点 [16,32,128]
        xyz: all points, [B, N, 3] 所有点云的数据
        new_xyz: query points, [B, S, 3]  为centroids点的数据
    Return:
        group_idx: grouped points index, [B, S, nsample]
        #输出位每个样本的每个球形邻域的nsample个采样点集用索引
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    #记录S个中心点与所有点之间的欧氏距离
    sqrdists = square_distance(new_xyz, xyz)
    #找到所有距离大于radius^ 的点，将group_idx直接为N；其保留原来的值
    group_idx[sqrdists > radius ** 2] = N
    #做升序排列，大于radius^2的都是N，在剩下的点中取钱nsample个点
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    #考虑到有可能nsample个点也被赋值为N（即球形区域不足nsample个点）
    #这种点需要舍弃，直接用第一个来替代
    #group_first：实际是把group_idx中第一个点复制;[B, S, K]的维度，方便后面替换
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    #找到group_idx中值等于N的点
    mask = group_idx == N
    #将这些点替换为第一个点的值
    group_idx[mask] = group_first[mask]
    return group_idx #S个group


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1) #b,n,c
        if points is not None:
            points = points.permute(0, 2, 1)

        B, N, C = xyz.shape
        S = self.npoint
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S)) #取出512个点
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None: #额外的向量
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))

            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat


class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self, xyz1, xyz2, points1, points2):
        """
        Input:
            xyz1: input points position data, [B, C, N]             [B, 64, 3]
            xyz2: sampled input points position data, [B, C, S]     [B, 16, 3]
            points1: input points data, [B, D, N]                   [B, 512, 64]
            points2: input points data, [B, D, S]                   [B, 1024, 16]
        Return:
            new_points: upsampled points data, [B, D', N]
        """
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            # 计算上一个点集中，每个点距离当前点集最近的三个点
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :3], idx[:, :, :3]  # [B, N, 3]   # 0.2 0.5 1.0

            #将 距离转换为倒数，距离越小的点在倒数中占比越大
            dist_recip = 1.0 / (dists + 1e-8)                          #1/0.2=5.0  1/0.5=2.0, 1/1.0=1
            # 计算每个点的倒数的和，得到一个归一化因子
            norm = torch.sum(dist_recip, dim=2, keepdim=True)          # 5.0+2.0 + 1 = 8
            # 倒数除以总合得到归一化，
            weight = dist_recip / norm                                  #5.0/8=0.625  2/8=0.25  1/8=0.125
            name = index_points(points2, idx)
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, 3, 1), dim=2) #从 points2 中根据 idx 提取出的点集

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = F.relu(bn(conv(new_points)))
        return new_points

