import os

import numpy as np
import torch
import importlib
from tqdm import tqdm

from data_utils.S3DISDataLoader import ScannetDatasetWholeScene


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace = True

def add_vote(vote_label_pool, point_idx, pred_label, weight):
    B = pred_label.shape[0]
    N = pred_label.shape[1]
    for b in range(B):
        for n in range(N):
            if weight[b, n] != 0 and not np.isinf(weight[b, n]):
                vote_label_pool[int(point_idx[b, n]), int(pred_label[b, n])] += 1
    return vote_label_pool

def main(args):
    def log_string(str):
        print(str)

    # 设置 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = os.path.join(BASE_DIR, 'log/sem_seg', args.log_dir)
    visual_dir = os.path.join(experiment_dir, 'visual')
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    exp_dir, checkpoints_dir, view_dir, logger = dirset(args, cls_expdir=False)

    log_string('Load parameters ...')
    log_string(args)

    # 这里需要根据 10.bin 文件的实际情况读取数据
    # 假设读取后的数据格式和 ScannetDatasetWholeScene 中的一致
    # 暂时使用示例代码中的数据加载逻辑
    testDataLoader = ScannetDatasetWholeScene(root=args.seg_path, split='test', test_area=args.test_area, block_points=args.seg_point)
    # 假设 10.bin 对应的场景索引为 0，实际需要根据情况修改
    scene_data, scene_label, scene_smpw, scene_point_index = testDataLoader[0]
    num_blocks = scene_data.shape[0]
    s_batch_num = (num_blocks + args.seg_batch_size - 1) // args.seg_batch_size
    batch_data = np.zeros((args.seg_batch_size, args.seg_point, 9))
    batch_label = np.zeros((args.seg_batch_size, args.seg_point))
    batch_point_index = np.zeros((args.seg_batch_size, args.seg_point))
    batch_smpw = np.zeros((args.seg_batch_size, args.seg_point))

    # 读取训练权重文件
    exp_dir = Path(exp_dir)
    logs_dir = exp_dir / 'logs'
    model_name = os.listdir(logs_dir)[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(args.seg_num_category).cuda()
    checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    whole_scene_data = testDataLoader.scene_points_list[0]
    whole_scene_label = testDataLoader.semantic_labels_list[0]
    vote_label_pool = np.zeros((whole_scene_label.shape[0], args.seg_num_category))

    with torch.no_grad():
        for _ in tqdm(range(args.seg_votes), total=args.seg_votes):
            for sbatch in range(s_batch_num):
                start_idx = sbatch * args.seg_batch_size
                end_idx = min((sbatch + 1) * args.seg_batch_size, num_blocks)
                real_batch_size = end_idx - start_idx
                batch_data[0:real_batch_size, ...] = scene_data[start_idx:end_idx, ...]
                batch_label[0:real_batch_size, ...] = scene_label[start_idx:end_idx, ...]
                batch_point_index[0:real_batch_size, ...] = scene_point_index[start_idx:end_idx, ...]
                batch_smpw[0:real_batch_size, ...] = scene_smpw[start_idx:end_idx, ...]
                batch_data[:, :, 3:6] /= 1.0

                torch_data = torch.Tensor(batch_data)
                torch_data = torch_data.float().cuda()

                seg_pred, _ = classifier(torch_data)
                batch_pred_label = seg_pred.contiguous().cpu().data.max(2)[1].numpy()

                vote_label_pool = add_vote(vote_label_pool, batch_point_index[0:real_batch_size, ...],
                                           batch_pred_label[0:real_batch_size, ...],
                                           batch_smpw[0:real_batch_size, ...])

        pred_label = np.argmax(vote_label_pool, 1)

    # 保存分割结果
    output_file_path = 'segmentation_result.txt'
    with open(output_file_path, 'w') as f:
        for label in pred_label:
            f.write(str(label) + '\n')
    log_string(f'Segmentation results saved to {output_file_path}')


if __name__ == '__main__':
    from config.seg_setting import parse_args
    args = parse_args()
    main(args)