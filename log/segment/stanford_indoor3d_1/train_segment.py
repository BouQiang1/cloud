"""
--本代码有B站up主 "三维点云-小小武"提供,
--目前致力于学习三维点云的分类和分割、二维图像的目标检测和语义分割 以及比较前沿的模块技术。相关视频已经上传到B站，
--希望小伙伴们多多点赞支持，如果能充电支持就更好了，谢谢大家。
"""
import os
import sys
import torch
import numpy as np
import random
import provider
import importlib
import shutil
from tqdm import tqdm
from data_utils.S3DISDataLoader import S3DISDataset
from tensorboardX import SummaryWriter
from config.seg_setting import optimizer_set,scheduler_set, parse_args
from config.seg_dir_set import dirset
from thop import profile
from thop import clever_format
import time
import multiprocessing as mp
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #获取当前脚本文件的所在目录的绝对路径
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models')) #将当前工作目录下的 models 目录加入到 sys.path


classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat


def worker_init_fn(worker_id):
    np.random.seed(worker_id + int(time.time()))

def inplace_relu(m): # 该激活函数将进行原地操作，在输入上直接进行修改，而不是创建一个新的输出张量，为了节省内存
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''设置GPU加载显卡，创建结果保存文件夹，日志生成等'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    exp_dir, checkpoints_dir, view_dir,logger = dirset(args, cls_expdir=False) # if cls_expdir=True create classificaion dir, else create segment dir
    log_string('Load parameters ...')
    log_string(args)

    '''加载数据集'''
    train_dataset =  S3DISDataset(split='train', data_root=args.seg_path, num_point=args.seg_point, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    test_dataset =   S3DISDataset(split='test',  data_root=args.seg_path, num_point=args.seg_point,test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.seg_batch_size, shuffle=True, num_workers=0, drop_last=True, pin_memory=True, worker_init_fn=worker_init_fn)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.seg_batch_size, shuffle=False, num_workers=0, drop_last=True)
    weights = torch.Tensor(train_dataset.labelweights).cuda() #类别不平衡（某些类别的数据比其他类别的数据少得多）可能会导致模型性能下降，某些类别在数据集中出现的频率较低，则会赋予这些类别更高的权重，以避免模型偏向于频率较高的类别。
    log_string("The number of training data is: %d" % len(train_dataset))
    log_string("The number of test data is: %d" % len(test_dataset))

    '''每次训练时将以下py文件保存到log_dir参数设定的文件夹中'''
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('config/seg_setting.py', str(exp_dir))
    shutil.copy('models/utils.py', str(exp_dir))
    shutil.copy('train_segment.py', str(exp_dir))
    shutil.copy('test_segment.py', str(exp_dir))

    '''获取模型和损失'''
    classifier = model.get_model(args.seg_num_category)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)
    if args.use_gpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    '''模型初始化阶段，判断是否有保存的训练文件；如果有则加载，反之重置训练参数'''
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    '''获取优化器'''
    optimizer = optimizer_set(args, classifier)
    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    log_string('Start training...')
    start_time = time.time()     #记录模型训练开始时间
    best_iou = 0
    global_epoch = 0
    learning_rate_clip = 1e-5    #学习率下限
    momentum_original = 0.1      #动量的初始值
    momentum_deccay = 0.5        #动量的衰减系数
    momentum_daccay_step = args.step_size

    for epoch in range(start_epoch, args.epoch):
        '''每周期设置学习率lr'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), learning_rate_clip)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        log_string('Learning rate:%f' % lr)

        '''每周期设置动量momentum'''
        momentum = momentum_original * (momentum_deccay ** (epoch // momentum_daccay_step))
        if momentum < 0.01:
            momentum = 0.01
        log_string('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))

        '''训练模型阶段'''
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()
        trainDataLoader_num = len(trainDataLoader)
        for i, (points, label) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader),smoothing=0.9):
            optimizer.zero_grad()

            '''数据增强'''
            points = points.data.numpy()    # B, N, C
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)    # B, N, C
            if args.use_gpu:
                points, label = points.float().cuda(), label.long().cuda()

            '''输出模型预测特征seg_pred'''
            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, args.seg_num_category) #（B，seg_num_category）
            batch_label = label.view(-1, 1)[:, 0].cpu().data.numpy()

            '''计算模型损失'''
            label = label.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, label, trans_feat, weights)
            loss.backward()
            optimizer.step()

            '''计算预测标签pred_choice与真实标签label的准确率'''
            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (args.seg_batch_size * args.seg_point)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / trainDataLoader_num))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''测试模型阶段'''
        with torch.no_grad():
            total_correct = 0 #计算预测正确的标签数量
            total_seen = 0    #计算分割点的总数量
            loss_sum = 0      #损失总和
            labelweights = np.zeros(args.seg_num_category)
            testDataLoader_num = len(testDataLoader)
            total_seen_class = [0 for _ in range(args.seg_num_category)]
            total_correct_class = [0 for _ in range(args.seg_num_category)]
            total_iou_deno_class = [0 for _ in range(args.seg_num_category)]
            classifier = classifier.eval()

            log_string('---- Epoch %03d Evaluation ----' % (global_epoch + 1))
            for i, (points, label) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                if args.use_gpu:
                    points, label = points.float().cuda(), label.long().cuda()

                '''输出模型预测特征seg_pred'''
                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()

                seg_pred = seg_pred.contiguous().view(-1, args.seg_num_category)
                batch_label = label.view(-1, 1)[:, 0].cpu().data.numpy()
                # batch_label = label.cpu().data.numpy()

                '''计算模型损失'''
                label = label.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, label, trans_feat, weights)
                loss_sum += loss

                '''计算预测标签与真实标签的准确率'''
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (args.seg_batch_size * args.seg_point)

                '''计算每个类别的标签出现次数'''
                tmp, _ = np.histogram(batch_label, range(args.seg_num_category + 1)) #tmp获得每个类别出现的次数。假设：batch_label = [0, 1, 2, 0, 1, 1, 2, 0]，args.seg_num_category = 2。tmp 得到 [3, 3, 2]
                labelweights += tmp

                for l in range(args.seg_num_category):
                    total_seen_class[l] += np.sum((batch_label == l))                           # 每个类别的样本数量
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))      # 每个类别预测正确的数量
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))   # 每个类别IoU的分母（预测结果或真实标签中至少有一个匹配的样本的数量）

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(testDataLoader_num)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(args.seg_num_category):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / testDataLoader_num))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1
    log_string('End of training...')

    # 汇总用时
    end_time = time.time()
    run_time = end_time - start_time
    m, s = divmod(run_time, 60)
    h, m = divmod(m, 60)
    log_string("%02d:%02d:%02d" % (h, m, s))

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    '''Set Hyper Parameter'''
    args = parse_args()

    '''Set seed'''
    seed = args.seed  # np.random.randint(1, 10000)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    '''Set Algorithm Consistency'''
    torch.backends.cudnn.enabled = True #使用非确定性算法
    torch.backends.cudnn.deterministic = True #保证算法得到一样的结果
    # torch.backends.cudnn.benchmark = True    #是否自动加速，增加显存
    main(args)
