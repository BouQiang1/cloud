import numpy as np
from mayavi import mlab
import os
import sys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

# 读取文件中的坐标(x,y,z)和标签.输入的文本是已经归一化好的
sample_points = 2048
points = np.genfromtxt('Cap_0.txt', delimiter=' ')
x = points[:, 0]  # x position of point
y = points[:, 1]  # y position of point
z = points[:, 2]  # z position of point
# x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
# y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
# z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
part_label =np.expand_dims( points[:,6], axis=1)


#预定义部件颜色的值
# rgba = np.array([[84, 61, 32,0], [121, 13, 22,0], [72, 251, 126,0], [112, 208, 166,0], [14, 137, 100,0], [118, 153, 23,0]])
# rgba[:, -1] = 255  # 设置颜色第四维度-不透明
rgba = np.array([[84, 61, 32, 255],  # 颜色值转换为0-255范围的整数
                 [121, 13, 22, 255],
                 [72, 251, 126, 255],
                 [112, 208, 166, 255],
                 [14, 137, 100, 255],
                 [118, 153, 23, 255],
                 [0, 0, 0, 0],    # 默认黑色 (完全透明)
                 [0, 255, 0, 255]]) # 标签为7的点为绿色

#根据不同的label值选择不同的部件颜色值
part_label =  np.where(part_label == 6, rgba[5], part_label.astype(int))
part_label =  np.where(part_label == 7, rgba[7], part_label.astype(int))

mlab.figure(bgcolor=(1, 1, 1)) # 绘制背景面板为白色
pts = mlab.pipeline.scalar_scatter(x, y, z) # 绘制三维散点图
pts.add_attribute(part_label, 'colors')
pts.data.point_data.set_active_scalars('colors')

# 修改glyph对象的属性来设置点的缩放因子和缩放模式
g = mlab.pipeline.glyph(pts)
g.glyph.glyph.scale_factor = 0.03 # 缩放大小
g.glyph.scale_mode = 'data_scaling_off' # 缩放模式

mlab.show()

