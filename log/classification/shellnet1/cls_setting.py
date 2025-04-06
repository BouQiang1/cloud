import torch
import argparse
def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('Training and Testing')
    # parser.add_argument('--conv_scale', type=int,default='2', help='')
    # parser.add_argument('--dense_scale',type=int, default='2', help='')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--seed', type=int, default=1206, help='set seed')
    parser.add_argument('--epoch', default=200, type=int, help='number of epoch in training')
    parser.add_argument('--model', default='shellnet_cls', help='model name')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training,SGD,adm,admw')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--learning_rate', default=1e-4, type=float, help='learning rate in training')
    parser.add_argument('--log_dir', type=str, default='shellnet1', help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use FPS sampiling')
    # cls
    parser.add_argument('--clsTrain_batch_size', type=int, default=24, help='batch size in training')
    parser.add_argument('--clsTest_batch_size', type=int, default=24, help='batch size in Testing')
    parser.add_argument('--cls_path', default='D:/Datas/modelnet40/', help='classification data path')
    parser.add_argument('--cls_scheduler', type=str, default='StepLR', help='scheduler for training')
    parser.add_argument('--cls_point', type=int, default=1024, help='Classification Number')
    parser.add_argument('--cls_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--cls_num_category', default=40, type=int, choices=[10,40], help='modelnet10,modelnet40')
    return parser.parse_args()


def optimizer_set(args,classifier):
    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    if args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
    if args.optimizer == 'Adamw':
        optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.decay_rate, amsgrad=True)
    return optimizer

def scheduler_set(name,optimizer,epoch):
    if name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    if name == 'CosineRestart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch, eta_min=1e-9)
    if name == 'Consine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-3)

    return scheduler
