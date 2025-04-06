
""" --本代码有B站up主 "小小武-酷爱计算机"提供,
--目前致力于学习三维点云的分类和分割、二维图像的目标检测和语义分割 以及比较前沿的模块技术。相关视频已经上传到B站，
--希望小伙伴们多多关注支持，如果能充电支持就更好了，谢谢大家。
"""

"""
V1.0：仅能使用三种颜色表示点集
V2.0：通过将RGB空间转换为HSV空间，可以适应多种颜色表示

"""

import numpy as np
from mayavi import mlab
import colorsys

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'data_utils'))

# 读取文件中的坐标(x,y,z)和标签.输入的文本是已经归一化好的
sample_points = 22003
points = np.genfromtxt("Datas.txt", delimiter=' ')   # 必须是归一化的！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！
x = points[:, 0]  # x position of point
y = points[:, 1]  # y position of point
z = points[:, 2]  # z position of point
# x_normalized = (x - np.min(x)) / (np.max(x) - np.min(x))
# y_normalized = (y - np.min(y)) / (np.max(y) - np.min(y))
# z_normalized = (z - np.min(z)) / (np.max(z) - np.min(z))
part_label =np.expand_dims( points[:,6], axis=1)


def rgba_to_hsv(rgba):
    rgb = rgba[:, :3] / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*rgb[i]) for i in range(rgb.shape[0])])
    hsv = np.hstack((hsv, rgba[:, 3:]))  # Preserve the alpha channel
    return hsv


#预定义部件颜色的值
rgba = np.array([[76, 76, 0, 255], [255, 127, 127, 255], [255, 255, 0, 255], [255, 0, 255, 255],
                 [255, 0, 0, 255], [82, 103, 165, 255], [152, 152, 152, 255], [254, 127, 0, 255],
                 [0, 0, 255, 255], [127, 95, 63, 255], [0, 255, 0, 255], [0, 255, 255, 255],
                 [255, 0, 255, 255], [128, 0, 128, 255]])


colors = np.zeros((part_label.shape[0], 4))

#  RGBA to HSV
hsv = rgba_to_hsv(rgba)
for i, label in enumerate(np.unique(part_label)):
    colors[part_label.flatten() == label] = hsv[i]

mlab.figure(bgcolor=(1, 1, 1)) # 绘制背景面板为白色
pts = mlab.pipeline.scalar_scatter(x, y, z) # 绘制三维散点图
pts.add_attribute(part_label, 'colors')
pts.data.point_data.set_active_scalars('colors')

# 修改glyph对象的属性来设置点的缩放因子和缩放模式
g = mlab.pipeline.glyph(pts)
g.glyph.glyph.scale_factor = 0.03 # 缩放大小
g.glyph.scale_mode = 'data_scaling_off' # 缩放模式

mlab.show()

