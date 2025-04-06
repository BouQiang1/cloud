import numpy as np
import os
import torch
import random
import logging
from tqdm import tqdm
import sys
import copy
import importlib
import sklearn.metrics as metrics
from config.cls_dir_set import dirset
from data_utils.FG3Dload import FindGrainedLoader
from data_utils.ModelNet40_hdf5_2048 import ModelNet40
from config.cls_setting import parse_args

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def test(model, loader, num_class, cls_votes=3):
    test_pred = []
    test_true = []
    classifier = model.eval()

    for j, (points, label) in tqdm(enumerate(loader), total=len(loader)):
        if  args.use_gpu:
            points, target = points.cuda(), label.cuda()
        # points = points.type(torch.cuda.FloatTensor)
        points = points.transpose(2, 1)
        '''按照投票方式计算准确率'''
        vote_pool = torch.zeros(label.size()[0], num_class).cuda()
        for _ in range(cls_votes):
            pred,_ = classifier(points)
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
    experiment_dir = 'log/classification/' + args.log_dir
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/eval.txt' % experiment_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    '''加载数据集'''
    log_string('Evaluate the accuracy of the model on the test dataset ...')
    test_dataset =  ModelNet40(partition='test', num_points=args.cls_point, cls_path = args.cls_path)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.clsTest_batch_size, shuffle=False, num_workers=4, drop_last=True)

    '''加载训练模型参数'''
    model_name = os.listdir(experiment_dir + '/logs')[0].split('.')[0]
    model = importlib.import_module(model_name)
    classifier = model.get_model(40,False)
    if args.use_gpu:
        classifier = classifier.cuda()
    checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
    #如果开启了para和flops功能，在验证阶段需要屏蔽掉，否则保存的结果中会出现该信息
    filtered_state_dict = {k: v for k, v in checkpoint['model_state_dict'].items() if not (
            'total_ops' in k or 'total_params' in k)}
    classifier.load_state_dict(filtered_state_dict)

    # classifier.load_state_dict(checkpoint['model_state_dict'])
    # classifier.load_state_dict(checkpoint['model_state_dict'],False)

    '''验证模型准确率'''
    with torch.no_grad():
        instance_acc, class_acc = test(classifier.eval(), testDataLoader, cls_votes=args.cls_votes, num_class=args.cls_num_category)
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))


if __name__ == '__main__':

    '''Set Hyper Parameter'''
    args = parse_args()

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

    main(args)
