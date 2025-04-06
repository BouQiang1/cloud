import os
import sys
import torch
import numpy as np
import random
import importlib
import logging
from tqdm import tqdm
from pathlib import Path
from data_utils.S3DISDataLoader import S3DISDataset, ScannetDatasetWholeScene
from config.seg_setting import optimizer_set, scheduler_set, parse_args
from data_utils.indoor3d_util import g_label2color
from config.seg_dir_set import dirset
import argparse  # 导入 argparse 模块

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

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
        logger.info(str)
        print(str)

    '''设置GPU加载显卡，创建结果保存文件夹，日志生成等'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    experiment_dir = os.path.join(BASE_DIR, 'log/sem_seg', args.log_dir)
    print(experiment_dir)

    visual_dir = os.path.join(experiment_dir, 'visual')
    if not os.path.exists(visual_dir):
        os.makedirs(visual_dir)

    exp_dir, checkpoints_dir, view_dir, logger = dirset(args, cls_expdir=False)

    '''控制台输出超参数'''
    log_string('Load parameters ...')
    log_string(args)

    '''加载数据集'''
    log_string('Load dataset ...')
    testDataLoader = ScannetDatasetWholeScene(root=args.seg_path, split='test', test_area=args.test_area, block_points=args.seg_point)
    log_string("The number of test data is: %d" % len(testDataLoader))

    '''读取训练权重文件'''
    exp_dir = Path(exp_dir)
    logs_dir = exp_dir / 'logs'
    model_name = os.listdir(logs_dir)[0].split('.')[0]
    MODEL = importlib.import_module(model_name)
    classifier = MODEL.get_model(args.seg_num_category).cuda()
    checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
    classifier.load_state_dict(checkpoint['model_state_dict'])
    classifier = classifier.eval()

    '''开始验证结果'''
    with torch.no_grad():
        scene_id = testDataLoader.file_list
        scene_id = [x[:-4] for x in scene_id]
        num_batches = len(testDataLoader)

        total_seen_class = [0 for _ in range(args.seg_num_category)]
        total_correct_class = [0 for _ in range(args.seg_num_category)]
        total_iou_deno_class = [0 for _ in range(args.seg_num_category)]

        log_string('---- Evaluation Whole Scene----')
        for batch_idx in range(num_batches):
            print("Inference [%d/%d] %s ..." % (batch_idx + 1, num_batches, scene_id[batch_idx]))
            total_seen_class_tmp = [0 for _ in range(args.seg_num_category)]
            total_correct_class_tmp = [0 for _ in range(args.seg_num_category)]
            total_iou_deno_class_tmp = [0 for _ in range(args.seg_num_category)]

            if args.seg_visual:
                fout = open(os.path.join(visual_dir, scene_id[batch_idx] + '_pred.obj'), 'w')
                fout_gt = open(os.path.join(visual_dir, scene_id[batch_idx] + '_gt.obj'), 'w')

            whole_scene_data = testDataLoader.scene_points_list[batch_idx]
            whole_scene_label = testDataLoader.semantic_labels_list[batch_idx]
            vote_label_pool = np.zeros((whole_scene_label.shape[0], args.seg_num_category))

            for _ in tqdm(range(args.seg_votes), total=args.seg_votes):
                scene_data, scene_label, scene_smpw, scene_point_index = testDataLoader[batch_idx]
                num_blocks = scene_data.shape[0]
                s_batch_num = (num_blocks + args.seg_batch_size - 1) // args.seg_batch_size
                batch_data = np.zeros((args.seg_batch_size, args.seg_point, 9))

                batch_label = np.zeros((args.seg_batch_size, args.seg_point))
                batch_point_index = np.zeros((args.seg_batch_size, args.seg_point))
                batch_smpw = np.zeros((args.seg_batch_size, args.seg_point))

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

            for l in range(args.seg_num_category):
                total_seen_class_tmp[l] += np.sum((whole_scene_label == l))
                total_correct_class_tmp[l] += np.sum((pred_label == l) & (whole_scene_label == l))
                total_iou_deno_class_tmp[l] += np.sum(((pred_label == l) | (whole_scene_label == l)))
                total_seen_class[l] += total_seen_class_tmp[l]
                total_correct_class[l] += total_correct_class_tmp[l]
                total_iou_deno_class[l] += total_iou_deno_class_tmp[l]

            iou_map = np.array(total_correct_class_tmp) / (np.array(total_iou_deno_class_tmp, dtype=float) + 1e-6)
            print(iou_map)

            arr = np.array(total_seen_class_tmp)
            tmp_iou = np.mean(iou_map[arr != 0])
            log_string('Mean IoU of %s: %.4f' % (scene_id[batch_idx], tmp_iou))
            print('----------------------------')

            filename = os.path.join(visual_dir, scene_id[batch_idx] + '.txt')
            with open(filename, 'w') as pl_save:
                for i in pred_label:
                    pl_save.write(str(int(i)) + '\n')
                pl_save.close()
            for i in range(whole_scene_label.shape[0]):
                color = g_label2color[pred_label[i]]
                color_gt = g_label2color[whole_scene_label[i]]
                if args.seg_visual:
                    fout.write('v %f %f %f %d %d %d\n' % (
                        whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color[0], color[1],
                        color[2]))
                    fout_gt.write(
                        'v %f %f %f %d %d %d\n' % (
                            whole_scene_data[i, 0], whole_scene_data[i, 1], whole_scene_data[i, 2], color_gt[0],
                            color_gt[1], color_gt[2]))
            if args.seg_visual:
                fout.close()
                fout_gt.close()

        IoU = np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6)
        iou_per_class_str = '------- IoU --------\n'
        for l in range(args.seg_num_category):
            iou_per_class_str += 'class %s, IoU: %.3f \n' % (
                seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])),
                total_correct_class[l] / float(total_iou_deno_class[l]))
        log_string(iou_per_class_str)
        log_string('eval point avg class IoU: %f' % np.mean(IoU))
        log_string('eval whole scene point avg class acc: %f' % (
            np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))
        log_string('eval whole scene point accuracy: %f' % (
                np.sum(total_correct_class) / float(np.sum(total_seen_class) + 1e-6)))
    log_string('End of training...')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加 gpu 参数
    parser.add_argument('--gpu', type=str, default='0', help='Specify which GPU to use')
    # 添加 log_dir 参数
    parser.add_argument('--log_dir', type=str, default='default_log_dir', help='Directory for logs')
    # 添加 seed 参数
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    # 添加 seg_path 参数
    parser.add_argument('--seg_path', type=str, default='D:/Datas/stanford_indoor3d/', help='Path to the segmentation dataset')
    # 添加 test_area 参数
    parser.add_argument('--test_area', type=int, default=5, help='Test area')
    # 添加 seg_point 参数
    parser.add_argument('--seg_point', type=int, default=4096, help='Number of points per segment')
    # 添加 seg_batch_size 参数
    parser.add_argument('--seg_batch_size', type=int, default=12, help='Segmentation batch size')
    # 添加 seg_visual 参数
    parser.add_argument('--seg_visual', type=bool, default=False, help='Whether to visualize segmentation results')
    # 添加 seg_votes 参数
    parser.add_argument('--seg_votes', type=int, default=3, help='Number of votes for segmentation')
    # 添加 seg_num_category 参数
    parser.add_argument('--seg_num_category', type=int, default=13, help='Number of segmentation categories')
    args = parser.parse_args()

    '''Set seed'''
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''Set Algorithm Consistency'''
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = True
    main(args)