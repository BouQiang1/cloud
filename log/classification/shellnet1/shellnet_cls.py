"""
--本代码有B站up主 "三维点云-小小武"提供,
--目前致力于学习三维点云的分类和分割、二维图像的目标检测和语义分割 以及比较前沿的模块技术。相关视频已经上传到B站，
--希望小伙伴们多多点赞支持，如果能充电支持就更好了，谢谢大家。
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import pdb

def index_points(points, idx):
    """
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
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids



def knn(points, queries, K):
    """
    Args:
        points ( B x N x 3 tensor )
        query  ( B x M x 3 tensor )  M < N
        K      (constant) num of neighbors
    Outputs:
        knn    (B x M x K x 3 tensor) sorted K nearest neighbor
        indice (B x M x K tensor) knn indices   
    """
    value = None
    indices = None
    num_batch = points.shape[0]
    for i in range(num_batch):
        point = points[i]
        query = queries[i]
        dist  = torch.cdist(point, query)
        idxs  = dist.topk(K, dim=0, largest=False, sorted=True).indices
        idxs  = idxs.transpose(0,1)
        nn    = point[idxs].unsqueeze(0)
        value = nn if value is None else torch.cat((value, nn))

        idxs  = idxs.unsqueeze(0)
        indices = idxs if indices is None else torch.cat((indices, idxs))
        
    return value.long(), indices.long()

def gather_feature(features, indices):
    """
    Args:
        features ( B x N x F tensor) -- feature from previous layer
        indices  ( B x M x K tensor) --  represents queries' k nearest neighbor
    Output:
        features ( B x M x K x F tensor) -- knn features from previous layer 
    """
    res = None
    num_batch = features.shape[0]
    for B in range(num_batch):
        knn_features = features[B][indices[B]].unsqueeze(0)
        res = knn_features if res is None else torch.cat((res, knn_features))
    return res

def random_sample(points, num_sample):
    """
    Args:
        points ( B x N x 3 tensor )
        num_sample (constant)
    Outputs:
        sampled_points (B x num_sample x 3 tensor)
    """    
    perm = torch.randperm(points.shape[1])
    return points[:, perm[:num_sample]].clone()
class Dense(nn.Module):
    def __init__(self, in_size, out_size, in_dim=3,
            has_bn=True, drop_out=None):
        super(Dense, self).__init__()
        """
        Args:
            input ( B x M x K x 3  tensor ) -- subtraction vectors 
                from query to its k nearest neighbor
        Output: 
            local point feature ( B x M x K x 64 tensor ) 
        """
        self.has_bn = has_bn
        self.in_dim = in_dim

        if in_dim == 3:
            self.batchnorm = nn.BatchNorm1d(in_size)
        elif in_dim == 4:
            self.batchnorm = nn.BatchNorm2d(in_size)
        else:
            self.batchnorm = None

        if drop_out is None:
            self.linear = nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU()
            )
        else:
            self.linear = nn.Sequential(
                nn.Linear(in_size, out_size),
                nn.ReLU(),
                nn.Dropout(drop_out)
            )

    def forward(self, inputs):

        if self.has_bn == True:
            d =  self.in_dim - 1
            outputs = self.batchnorm(inputs.transpose(1, d)).transpose(1, d)
            outputs = self.linear(outputs)
            return outputs

        else:
            outputs = self.linear(inputs)
            return outputs


class ShellConv(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division,
                 has_bn=True):
        super(ShellConv, self).__init__()
        """
        out_features  (int) num of output feature (dim = -1)
        prev_features (int) num of prev feature (dim = -1)
        neighbor      (int) num of nearest neighbor in knn
        division      (int) num of division
        """

        self.K = neighbor
        self.S = int(self.K / division)  # num of feaure per shell
        self.F = 64  # num of local point features
        self.neighbor = neighbor
        in_channel = self.F + prev_features
        out_channel = out_features

        self.dense1 = Dense(3, self.F // 2, in_dim=4, has_bn=has_bn)
        self.dense2 = Dense(self.F // 2, self.F, in_dim=4, has_bn=has_bn)
        self.maxpool = nn.MaxPool2d((1, self.S), stride=(1, self.S))
        if has_bn == True:
            self.conv = nn.Sequential(
                nn.BatchNorm2d(in_channel),
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, (1, division)),
                nn.ReLU(),
            )

    def forward(self, points, queries, prev_features):
        """
        Args:
            points          (B x N x 3 tensor)
            query           (B x M x 3 tensor) -- note that M < N
            prev_features   (B x N x F1 tensor)
        Outputs:
            feat            (B x M x F2 tensor)
        """

        knn_pts, idxs = knn(points, queries, self.K)
        knn_center = queries.unsqueeze(2)
        knn_points_local = knn_center - knn_pts

        knn_feat_local = self.dense1(knn_points_local)
        knn_feat_local = self.dense2(knn_feat_local)

        # shape: B x M x K x F
        if prev_features is not None:
            knn_feat_prev = gather_feature(prev_features, idxs)
            knn_feat_cat = torch.cat((knn_feat_local, knn_feat_prev), dim=-1)
        else:
            knn_feat_cat = knn_feat_local

        knn_feat_cat = knn_feat_cat.permute(0, 3, 1, 2)  # BMKF -> BFMK
        knn_feat_max = self.maxpool(knn_feat_cat)
        output = self.conv(knn_feat_max).permute(0, 2, 3, 1)

        return output.squeeze(2)


class ShellUp(nn.Module):
    def __init__(self, out_features, prev_features, neighbor, division, 
            has_bn=True):
        super(ShellUp, self).__init__()
        self.has_bn = has_bn
        self.sconv = ShellConv(out_features, prev_features, neighbor,
            division, has_bn)
        self.dense = Dense(2 * out_features, out_features, has_bn=has_bn)

    def forward(self, points, queries, prev_features, feat_skip_connect):
        sconv = self.sconv(points, queries, prev_features)
        feat_cat = torch.cat((sconv, feat_skip_connect), dim=-1)

        outputs = self.dense(feat_cat)
        return outputs


class get_model(nn.Module):
    def __init__(self, args, has_bn=True):
        super(get_model, self).__init__()
        self.num_class = args.cls_num_category # 分类数量
        self.num_points = args.cls_point   # 采样点
        self.conv_scale = 1  # 每层卷积输出通道缩放比
        self.dense_scale = 1 # FC层输出通道缩放比

        filters = [64, 128, 256, 512]
        filters = [int(x / self.conv_scale) for x in filters] # [32, 64, 128, 256]

        features = [256, 128]
        features = [int(x / self.dense_scale) for x in features] #[128, 64]

        self.shellconv1 = ShellConv(filters[1], 0, 32, 4, has_bn)
        self.shellconv2 = ShellConv(filters[2], filters[1], 16, 2, has_bn)
        self.shellconv3 = ShellConv(filters[3], filters[2], 8, 1, has_bn)

        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, self.num_class)

    def forward(self, inputs):
        '''
        :param inputs: B,N,C
        :return:
        '''
        B,_,_ = inputs.shape
        query1 = random_sample(inputs, self.num_points // 2)
        sconv1 = self.shellconv1(inputs, query1, None)
        # print("sconv1.shape = ", sconv1.shape)

        query2 = random_sample(query1, self.num_points // 4)
        sconv2 = self.shellconv2(query1, query2, sconv1)
        # print("sconv2.shape = ", sconv2.shape)

        query3 = random_sample(query2, self.num_points // 8)
        sconv3 = self.shellconv3(query2, query3, sconv2)
        # print("sconv3.shape = ", sconv3.shape)
        sconv3 = torch.max(sconv3, dim=1)[0]

        x = self.drop1(F.relu(self.bn1(self.fc1(sconv3))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        output = self.fc3(x)
        return output,_


class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
        self.loss = nn.CrossEntropyLoss()

    def forward(self, pred, target,_):
        total_loss = self.loss(pred, target.view(-1))
        return total_loss


if __name__ == '__main__':
    B, M, K = 2, 1024, 32
    # Create random Tensors to hold inputs and outputs
    p = torch.randn(B, M, 3)
    q = torch.randn(B, M//2, 3)
    f = torch.randn(B, M, 128)
    y = torch.randn(B, M//2, 128)

    nn_pts, idxs = knn(p, q, 32)
    nn_center    = q.unsqueeze(2)
    nn_points_local = nn_center - nn_pts

    # model = get_model(2, 1024, conv_scale=1, dense_scale=1)
    # print(model(p).shape)