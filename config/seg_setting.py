import torch
import argparse
def parse_args():

    parser = argparse.ArgumentParser('Training and Testing')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='use gpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--seed', type=int, default=3407, help='set seed')
    parser.add_argument('--epoch', default=32, type=int, help='number of epoch in training')
    parser.add_argument('--model', default='pointnet2_msg_seg', help='model name')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training,SGD,adm,admw')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--log_dir', type=str, default='stanford_indoor3d_1', help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')  #根据需要是否使用
    parser.add_argument('--use_uniform_sample', action='store_true', default=True, help='use FPS sampiling')
    # seg
    parser.add_argument('--seg_batch_size', type=int, default=12, help='Batch size in training')
    parser.add_argument('--seg_path', type=str, default='D:/Datas/stanford_indoor3d/', help='Segment Data Path')
    parser.add_argument('--seg_point', type=int, default=4096, help='Segment Number')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--seg_visual', action='store_true', default=False, help='visualize result [default: False]')
    parser.add_argument('--seg_votes', type=int, default=3, help='Aggregate classification scores with voting')
    parser.add_argument('--seg_num_category', default=13, type=int, choices=[13], help='S3DISDataset num')
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    if name == 'CosineRestart':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=epoch, eta_min=1e-9)
    if name == 'Consine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=1e-3)

    return scheduler
