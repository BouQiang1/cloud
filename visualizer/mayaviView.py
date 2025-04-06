import numpy as np
import mayavi.mlab
import argparse
import os
import sys
from Apoint.data_utils.FG3Dload import FindGrainedLoader
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))
parser = argparse.ArgumentParser()
parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
parser.add_argument('--use_normals', action='store_true', default=True, help='use normals')
parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use uniform sampiling')
parser.add_argument('--data_path', default='Chair', help='Airplane/Car/Chair')
args = parser.parse_args()


# data_path = 'D:/localCode/point2/data'
train_dataset = FindGrainedLoader(args=args, split='train')

data = train_dataset[1]
pointcloud, label= data
print(label)
# lidar_path更换为自己的.bin文件路径

x = pointcloud[:, 0]  # x position of point
y = pointcloud[:, 1]  # y position of point
z = pointcloud[:, 2]  # z position of point

# r = pointcloud[:, 3]  # reflectance value of point
d = np.sqrt(x ** 2 + y ** 2)  # Map Dstance from sensor

degr = np.degrees(np.arctan(z / d))
s = 2
vals = 'height'
if vals == "height":
    col = z
else:
    col = d
s = [1.5,1.5]
fig = mayavi.mlab.figure(bgcolor=(0, 0, 0), size=(1200, 1000))
mayavi.mlab.points3d(x, y, z,
                     # col,  # Values used for Color
                     colormap='copper',  # 'bone', 'copper', 'gnuplot'
                     color=(1, 0, 0),   # Used a fixed (r,g,b) instead
                     figure=fig,
                     resolution=10
                     )

mayavi.mlab.show()