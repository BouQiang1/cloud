import torch
import numpy as np

# 假设 10.bin 是一个 N x C 的点云数据文件
def load_bin_file(file_path):
    # 这里需要根据实际的二进制文件格式进行读取
    # 假设文件中存储的是浮点数类型的点云数据
    data = np.fromfile(file_path, dtype=np.float32)
    # 尝试不同的特征数量
    possible_features = [3, 4, 6, 7]  # 常见的点云特征数量
    for num_features in possible_features:
        if data.size % num_features == 0:
            num_points = data.size // num_features
            data = data.reshape(num_points, num_features)
            # 只取前3个特征（x, y, z）
            if num_features > 3:
                data = data[:, :3]
            return data
    print("无法确定正确的特征数量，请检查二进制文件格式。")
    return None

# 修改测试数据加载部分
def get_test_data():
    bin_file_path = '10.bin'
    points = load_bin_file(bin_file_path)
    if points is not None:
        # 将数据转换为 PyTorch 张量
        points = torch.from_numpy(points).float().unsqueeze(0)  # 添加 batch 维度
        return points
    return None