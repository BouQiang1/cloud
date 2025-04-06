"""
--本代码有B站up主 "三维点云-小小武"提供,
--目前致力于学习三维点云的分类和分割、二维图像的目标检测和语义分割 以及比较前沿的模块技术。相关视频已经上传到B站，
--希望小伙伴们多多点赞支持，如果能充电支持就更好了，谢谢大家。
"""
import os
import numpy as np
import warnings
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')

#归一化点云，提高网络的收敛速度，将点集进行缩放到均值为0 ，方差唯一的标准正太分布中  ,
def pc_normalize(pc):
    #每一列求均值，共有三列
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    #方差中最大的一个
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc


def farthest_point_sample(point, npoint):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, D = point.shape
    xyz = point[:,:3]
    centroids = np.zeros((npoint,))  #centroids
    distance = np.ones((N,)) * 1e10
    #随机生成一个数
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]
    return point


class ModelNetDataLoader(Dataset):
    def __init__(self, args, split='train', process_data=False):
        self.root = '/bjteam/datasets/wrs/modelnet40_normal_resampled'#数据集地址
        self.npoints = args.num_point #采样点
        self.process_data = process_data#是否使用离线处理，一般False
        self.uniform = args.use_uniform_sample#使用最远点采样
        self.use_normals = args.use_normals#是否使用法向量
        self.num_category = args.num_category#类别数量
        if self.num_category == 10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        #rstrip()去除后面的空白字符,获得分类文件的名字，输出为List
        self.cat = [line.rstrip() for line in open(self.catfile)]
        #zip()返回以元组为元素的字典，输出{'key_1':value_1,'key_2':value_2,......}
        #输出为字典类型，为每个分类打上标签
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {}
        if self.num_category == 10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]

        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        #datapath接收一个元组类型的列表，如：[('bathtub', '/data/modelnet40_normal_resampled/bathtub\\bathtub_0068.txt'),(),...]
        #返回所有类别名称与相应的文件位置
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]
        print('The size of %s data is %d' % (split, len(self.datapath)))
        '''预训练数据'''
        if self.uniform:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts_fps.dat' % (self.num_category, split, self.npoints))
        else:
            self.save_path = os.path.join(self.root, 'modelnet%d_%s_%dpts.dat' % (self.num_category, split, self.npoints))

        if self.process_data:
            if not os.path.exists(self.save_path):
                print('Processing data %s (only running in the first time)...' % self.save_path)
                #输出一个列表，列表的长度是所有文件夹下文件的总数量， [None,None]
                self.list_of_points = [None] * len(self.datapath)
                self.list_of_labels = [None] * len(self.datapath)

                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                    #fn，返回列表中的一个元组，如('bathtub', '/data/modelnet40_normal_resampled/bathtub\\bathtub_0068.txt')
                    fn = self.datapath[index]
                    #返回每个文件的类别
                    cls = self.classes[self.datapath[index][0]]
                    cls = np.array([cls]).astype(np.int32)
                    #二维张量
                    point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

                    if self.uniform:
                        point_set = farthest_point_sample(point_set, self.npoints)
                    else:
                        point_set = point_set[0:self.npoints, :]

                    self.list_of_points[index] = point_set
                    self.list_of_labels[index] = cls

                with open(self.save_path, 'wb') as f:
                    pickle.dump([self.list_of_points, self.list_of_labels], f)
            else:
                print('Load processed data from %s...' % self.save_path)
                with open(self.save_path, 'rb') as f:
                    self.list_of_points, self.list_of_labels = pickle.load(f)
    #返回数据集的样本个数
    def __len__(self):
        return len(self.datapath)

    def _get_item(self, index):
        if self.process_data:
            point_set, label = self.list_of_points[index], self.list_of_labels[index]
        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            label = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)

            if self.uniform:
                point_set = farthest_point_sample(point_set, self.npoints)
            else:
                point_set = point_set[0:self.npoints, :]
                
        point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
        if not self.use_normals:
            point_set = point_set[:, 0:3]

        return point_set, label[0]

    def __getitem__(self, index):
        return self._get_item(index)


if __name__ == '__main__':
    import torch

    data = ModelNetDataLoader('/data/modelnet40_normal_resampled/', split='train')
    DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
    for point, label in DataLoader:
        print(point.shape)
        print(label.shape)
