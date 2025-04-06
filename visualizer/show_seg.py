from __future__ import print_function
from show3d_balls import showpoints
import argparse
import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import matplotlib.pyplot as plt
from point2.data_utils.FindGrained_v import FindGrainedLoader

#showpoints(np.random.randn(2500,3), c1 = np.random.uniform(0,1,size = (2500)))
parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
parser.add_argument('--data_path', default='Car', help='Airplane/Car/Chair')
args = parser.parse_args()
data_path = 'D:/localCode/point2/data'
train_dataset = FindGrainedLoader(root=data_path, args=args, split='train', process_data=False)
idx = np.random.randint(0, len(train_dataset))
data = train_dataset[idx]
point_set, _ = data
np.random.seed(100)

seg = np.array([1])
# cmap = plt.cm.get_cmap("hsv", 10)
# cmap = np.array([cmap(i) for i in range(10)])[:, :3]
cmap = np.array([[1.00000000e+00, 0.00000000e+00, 0.00000000e+00],
                 [3.12493437e-02, 1.00000000e+00, 1.31250131e-06],
                 [0.00000000e+00, 6.25019688e-02, 1.00000000e+00],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02],
                 [1.00000000e+00, 0.00000000e+00, 9.37500000e-02]])

gt = cmap[seg - 1, :]

pred_choice = np.array([1])
#print(pred_choice.size())
pred_color = cmap[pred_choice[0], :]

print(pred_color.shape)
showpoints(point_set,c_pred=pred_color)
