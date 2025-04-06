import torch

def tensor_centroid(points):
    value = None
    num_batch = points.shape[0]
    for i in range(num_batch):
        point = points[i]
        point_xyz_mean = point.mean(dim=0,keepdim=True)
        print(point_xyz_mean)
        point = point.unsqueeze(0)
        value = point if value is None else torch.cat((value,point))
    return value

def compute_dim(B, S, K, grouped_xyz, pointset_centroid):
    '''
    多加一维度
    '''
    # 每个簇的中心点
    grouped_xyz_centroid = grouped_xyz.mean(dim=2, keepdim=True)  # [B,S,1,3]
    # 维度对齐
    pointset_centroid = pointset_centroid.unsqueeze(1)  # [B,1,1,3]
    pointset_centroid_expand = pointset_centroid.expand(B, S, 1, 3)
    # 计算中心的距离
    sum_dis = torch.sum((grouped_xyz_centroid - pointset_centroid_expand), dim=3, keepdim=True)
    sum_dis_copy = sum_dis.expand(B, S, K, 1)
    grouped_xyz = torch.cat((grouped_xyz, sum_dis_copy), dim=3)  # [B,S,K,4]
    return grouped_xyz