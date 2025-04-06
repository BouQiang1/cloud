import os
import sys
import torch
import numpy as np
import random
import provider
import importlib
import shutil
from tqdm import tqdm
# from data_utils.FG3Dload import FindGrainedLoader
from data_utils.ModelNet40_hdf5_2048 import ModelNet40
#from data_utils.ModelNetDataLoader import ModelNetDataLoader
from tensorboardX import SummaryWriter
from config.cls_setting import optimizer_set,scheduler_set, parse_args
from config.cls_dir_set import dirset
import sklearn.metrics as metrics
from thop import profile
from thop import clever_format
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #获取当前脚本文件的所在目录的绝对路径
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models')) #将当前工作目录下的 models 目录加入到 sys.path

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader, num_class, cls_votes=1):
    test_pred = []
    test_true = []
    classifier = model.eval()

    for j, (points, label) in tqdm(enumerate(loader), total=len(loader)):
        if  args.use_gpu:
            points, label = points.cuda(), label.cuda()
        points = points.transpose(2, 1)

        '''按照投票方式计算准确率'''
        vote_pool = torch.zeros(label.size()[0], num_class).cuda()
        for _ in range(cls_votes):
            pred,_ = classifier(points.float())
            vote_pool += pred
        pred = vote_pool / cls_votes
        pred_choice = pred.data.max(1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(pred_choice.detach().cpu().numpy())

    train_true = np.concatenate(test_true)
    train_pred = np.concatenate(test_pred)
    instance_acc = metrics.accuracy_score(train_true, train_pred)
    class_acc = metrics.balanced_accuracy_score(train_true, train_pred)
    return instance_acc, class_acc

def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''设置GPU加载显卡，创建结果保存文件夹，日志生成等'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    exp_dir, checkpoints_dir, view_dir,logger = dirset(args,cls_expdir=True)

    log_string('Load parameters ...')
    log_string(args)

    '''加载数据集'''
    log_string('Load dataset ...')
    train_dataset = ModelNet40(partition='train', num_points=args.cls_point, cls_path = args.cls_path,fps=args.use_uniform_sample)
    test_dataset =  ModelNet40(partition='test', num_points=args.cls_point, cls_path = args.cls_path,fps=args.use_uniform_sample)
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.clsTrain_batch_size, shuffle=True, num_workers=4, drop_last=True,pin_memory=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.clsTest_batch_size, shuffle=False, num_workers=4, drop_last=True)
    print('Training Sample:{},Test Sample:{}'.format(len(train_dataset), len(test_dataset)))

    '''每次训练时将以下py文件保存到log_dir参数设定的文件夹中'''
    model = importlib.import_module(args.model)
    shutil.copy('./models/%s.py' % args.model, str(exp_dir))
    shutil.copy('config/cls_setting.py', str(exp_dir))
    shutil.copy('models/utils.py', str(exp_dir))
    shutil.copy('train_classification.py', str(exp_dir))
    shutil.copy('test_classification.py', str(exp_dir))

    '''获取模型和损失'''
    classifier = model.get_model(40,args)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)
    if args.use_gpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    '''模型初始化阶段，判断是否有保存的训练文件；如果有则加载，反之重置训练参数'''
    try:
        model_path = str(exp_dir) + '/checkpoints/best_model.pth'
        print(f"Attempting to load model from: {model_path}")
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path)
            start_epoch = checkpoint['epoch']
            classifier.load_state_dict(checkpoint['model_state_dict'])
            log_string('Use pretrain model')
        else:
            log_string(f"Model file {model_path} not found. Starting training from scratch...")
            start_epoch = 0
    except Exception as e:
        log_string(f"Error loading model: {e}. Starting training from scratch...")
        start_epoch = 0

    '''获取优化器、调度器'''
    optimizer = optimizer_set(args, classifier)
    scheduler = scheduler_set(args.cls_scheduler, optimizer, args.epoch)
    writer = SummaryWriter(log_dir = view_dir)

    # '''计算模型参数量和Flops'''
    # input = torch.randn(1, 3, 1024).type(torch.cuda.FloatTensor)
    # flops, params = profile(classifier, inputs=(input,))
    # flops, params = clever_format([flops, params], "%.3f")
    # print('flops:{}'.format(flops))
    # print('params:{}'.format(params))

    log_string('Start training...')
    start_time = time.time()       #开始时间
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0        #总体准确率(OA)
    best_class_acc = 0.0           #类别准确率

    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_loss = []
        train_pred = []
        train_true = []
        classifier = classifier.train()

        '''训练模型阶段'''
        for batch_id, (points, label) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader),smoothing=0.9):
            optimizer.zero_grad()
            '''数据增强'''
            points = points.data.numpy() # B, N, C
            points = provider.random_point_dropout(points)
            points = provider.random_scale_point_cloud(points)
            points = provider.shift_point_cloud(points)
            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            # points = points.type(torch.cuda.FloatTensor) # B,C,N FG3D数据集专用

            '''输出模型预测特征pred和损失loss'''
            if args.use_gpu:
                points, label = points.cuda(), label.cuda()
            pred,trans_feat= classifier(points.float())
            loss = criterion(pred, label.long())

            '''计算预测标签pred_choice与真实标签label的准确率'''
            pred_choice = pred.data.max(1)[1]
            train_true.append(label.cpu().numpy())
            train_pred.append(pred_choice.detach().cpu().numpy())
            mean_loss.append(loss.item())
            loss.backward()
            optimizer.step()
            global_step += 1
        scheduler.step()
        train_loss = np.mean(mean_loss)
        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        writer.add_scalar('train loss /epoch', train_loss, epoch + 1)
        log_string('Train Instance Accuracy: %f,  Loss: %f'% (metrics.accuracy_score(train_true, train_pred), train_loss))

        '''测试模型阶段'''
        with torch.no_grad():
            instance_acc, class_acc = test(classifier.eval(), testDataLoader, num_class=args.cls_num_category, cls_votes=args.cls_votes)
            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1
            if (class_acc >= best_class_acc):
                best_class_acc = class_acc

            writer.add_scalar('Test Instance Accuracy/epoch', instance_acc, epoch + 1)
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f, Loss: %f' % (instance_acc, class_acc, train_loss))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                log_string('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1
    log_string('End of training...')

    # 汇总用时
    end_time = time.time()
    run_time = end_time - start_time
    m, s = divmod(run_time, 60)
    h, m = divmod(m, 60)
    log_string("%02d:%02d:%02d" % (h, m, s))

if __name__ == '__main__':

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
