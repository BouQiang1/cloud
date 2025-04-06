import torch
import numpy as np
import random

def Height_Feature_Mask(neighborhood, groups,flag):
    pass

#uniform mask implementation
def UniformDistribution_mask(centreSample, channels, mask_ratio):
    G = centreSample
    C = channels
    #特征图片总数
    #计算出掩码的个数
    total_len = G * C
    len_keep = int(total_len * (1-mask_ratio)) #保留12个像素位置
    #生成0-1之间的均匀分布矩阵
    uniform_noise = torch.rand(1,total_len).cuda()
    #按照固定维度进行排序，返回升序排列的索引值。
    uniform_sort = torch.argsort(uniform_noise,dim=1)
    mask = torch.zeros([1,total_len]).cuda()
    mask[:,:len_keep] = 1
    mask = torch.gather(mask,dim=1,index=uniform_sort)
    mask = mask.reshape(G,C)
    return mask

#shuffle mask method
def Feature_Mask(neighborhood, groups,flag):
    """
    [B,C,K,S]
    neighborhood:相当K
    groups：S
    """
    foo = [1] * (int(neighborhood-flag)) + [0] * flag
    bar = []
    for i in range(int(groups)):
        random.shuffle(foo)
        bar += foo
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(neighborhood,groups)
    bar = torch.from_numpy(bar)
    bar = bar.cuda()
    return bar
#top1的掩码
def Automask_Feature(grouped_points,out_score_maxindex,flag):
    points = grouped_points
    B, G, C = points.shape
    # mask = Feature_Mask(K, S, flag)
    mask = UniformDistribution_mask(G, C, flag)

    #获得每个batch中最大的通道索引
    maxindex = out_score_maxindex
    features_list = []
    for i in range(int(B)):
        features = points[i]
        if maxindex[i].eq(i):
            flag = features * mask
            features_list.append(flag.unsqueeze(0))
        else:
            features_list.append(features.unsqueeze(0))
    mask_feature = torch.cat(features_list, dim=0)
    return  mask_feature

#topK的掩码
# def Automask_Feature(grouped_points,out_score_maxindex,flag):
#     B, C, K, S = grouped_points.shape
#     maxB,_,_,_ =out_score_maxindex.shape
#     mask = Feature_Mask(K, S, flag)
#     maxindex = out_score_maxindex.reshape(maxB,-1) #[12,3]
#     split_list = []
#     for i in range(int(B)):
#         features_list = []
#         features = grouped_points[i]
#         for j in range(C):
#             if j in maxindex[i]:
#                 flag = features[j] * mask
#                 features_list.append(flag.unsqueeze(0))
#             else:
#                 features_list.append(features[j].unsqueeze(0))
#         list_cat = torch.cat(features_list, dim=0)
#         split_list.append(list_cat.unsqueeze(0))
#     mask_feature = torch.cat(split_list, dim=0)
#     return  mask_feature

#另一种实现方式
# def Automask_Feature(grouped_points,out_score_maxindex,flag):
#     B, C, K, S = grouped_points.shape
#     mask = Feature_Mask(K, S, flag)
#     #获得每个batch中最大的通道索引
#     maxindex = out_score_maxindex
#     split_list = None
#     for i in range(int(B)):
#         features_list = None
#         features = grouped_points[i]
#         for j in range(C):
#             if maxindex[i].eq(j):
#                 flag = features[j] * mask
#                 flag_unsq = flag.unsqueeze(0)
#                 features_list = flag_unsq if features_list is None else torch.cat((features_list,flag_unsq))
#             else:
#                 flag_unsq = features[j].unsqueeze(0)
#                 features_list = flag_unsq if features_list is None else torch.cat((features_list, flag_unsq))
#         list_cat_unsq = features_list.unsqueeze(0)
#         split_list = list_cat_unsq if split_list is None else torch.cat((split_list,list_cat_unsq))
#     return  split_list