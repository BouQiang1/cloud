import torch
import numpy as np
import random

def Feature_Mask(neighborhood, groups,mask_ratio):
    N = neighborhood
    G = groups
    #特征图片总数
    #计算出掩码的个数
    total_len = N * G
    len_keep = int(total_len * (1-mask_ratio))
    #生成0-1之间的均匀分布矩阵
    uniform_noise = torch.rand(1,total_len).cuda()
    #按照固定维度进行排序，返回升序排列的索引值。
    uniform_sort = torch.argsort(uniform_noise,dim=1)
    mask = torch.zeros([1,total_len]).cuda()
    mask[:,:len_keep] = 1
    mask = torch.gather(mask,dim=1,index=uniform_sort)
    mask = mask.reshape(N,G)
    return mask
