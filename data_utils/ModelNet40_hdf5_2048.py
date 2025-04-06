"""
--本代码有B站up主 "三维点云-小小武"提供,
--目前致力于学习三维点云的分类和分割、二维图像的目标检测和语义分割 以及比较前沿的模块技术。相关视频已经上传到B站，
--希望小伙伴们多多点赞支持，如果能充电支持就更好了，谢谢大家。
"""
import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = BASE_DIR + '/../data'
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition,cls_path):
    DATA_DIR = cls_path
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)

    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    dropout_ratio = np.random.random()*max_dropout_ratio
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:]
    return pc


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud

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
def pc_normalize(pc):
    #每一列求均值，共有三列
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    #方差中最大的一个
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

class ModelNet40(Dataset):
    def __init__(self, partition='train', num_points=1024, cls_path='data', fps=False, use_normals=False):
        self.data, self.label = load_data(partition,cls_path)
        self.num_points = num_points
        self.fps = fps
        self.use_normals = use_normals
    # def __getitem__(self, item):
    #
    #     if self.uniform:  # Using the sampling method FPS
    #         current_points = farthest_point_sample(current_points, self.npoints)
    #     else:
    #         # Using the sampling method Random
    #         choice = np.random.choice(len(current_points), self.npoints, replace=True)
    #         current_points = current_points[choice, :]
    #
    #
    #     pointcloud = self.data[item][:self.num_points]
    #     label = self.label[item]
    #     # if self.partition == 'train':
    #         # if self.type == 'pn':
    #         #     pointcloud = translate_pointcloud(pointcloud)
    #         # np.random.shuffle(pointcloud)
    #     return pointcloud, label
    def __getitem__(self, index):
        pt_idxs = np.arange(0, self.data.shape[1])  # 2048
        # if self.train:
        #     np.random.shuffle(pt_idxs)
        pointcloud = self.data[index, pt_idxs].copy()
        label = self.label[index]

        if  self.use_normals:
            pointcloud = pointcloud[:, :6];
        else:
            pointcloud = pointcloud[:, :3];

        if self.fps:  # Using the sampling method FPS
            pointcloud = farthest_point_sample(pointcloud, self.num_points)
        else:
            # Using the sampling method Random
            choice = np.random.choice(len(pointcloud), self.num_points, replace=True)
            pointcloud = pointcloud[choice, :]

        pointcloud = pc_normalize(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    # train = ModelNet40(1024)
    # test = ModelNet40(1024, 'test')

    from torch.utils.data import DataLoader
    train_loader = DataLoader(ModelNet40(partition='train', num_points=1024, cls_path='D:/Datas/modelnet40', fps=True), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

    train_set = ModelNet40(partition='train', num_points=1024)
    test_set = ModelNet40(partition='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
