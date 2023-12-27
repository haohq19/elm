import os
import argparse
import numpy as np
import random
import glob
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from spikingjelly.datasets import asl_dvs, cifar10_dvs, dvs128_gesture, n_caltech101, n_mnist
from spikingjelly.datasets import split_to_train_test_set
from utils import *
from models.event_transformer import EventResNet

_seed_ = 2023
random.seed(2023)
torch.manual_seed(_seed_)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

def parser_args():
    parser = argparse.ArgumentParser(description='train')
    # data
    parser.add_argument('--dataset', default='n_mnist', type=str, help='dataset')
    parser.add_argument('--root', default='/home/haohq/datasets/NMNIST', type=str, help='path to dataset')
    parser.add_argument('--nframes', default=16, type=int, help='number of frames')

    parser.add_argument('--nclasses', default=10, type=int, help='number of classes')
    parser.add_argument('--batch_size', default=1024, type=int, help='batch size')
    # model
    parser.add_argument('--d_model', default=512, type=int, help='d_model')

    # run
    parser.add_argument('--device_id', default=6, type=int, help='GPU id to use, only available in non-distributed training')
    parser.add_argument('--nepochs', default=200, type=int, help='number of epochs')
    parser.add_argument('--nworkers', default=16, type=int, help='number of workers')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--output_dir', default='outputs/', help='path to save')
    parser.add_argument('--save_freq', default=10, type=int, help='save frequency')
    # dist
    parser.add_argument('--world-size', default=8, type=int, help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    return parser.parse_args()


def load_data(args):

    if args.dataset == 'cifar10_dvs':  # downloaded
        dataset = cifar10_dvs.CIFAR10DVS(root=args.root, data_type='frame', split_by='time', frames_number=args.nframes)
        train_dataset, test_dataset = split_to_train_test_set(train_ratio=0.9, origin_dataset=dataset, num_classes=args.nclasses)
    elif args.dataset == 'dvs128_gesture':  # downloaded
        train_dataset = dvs128_gesture.DVS128Gesture(root=args.root, train=True, data_type='frame', split_by='time', frames_number=args.nframes)
        test_dataset = dvs128_gesture.DVS128Gesture(root=args.root, train=False, data_type='frame', split_by='time', frames_number=args.nframes)
    elif args.dataset == 'n_caltech101':  # downloaded
        dataset = n_caltech101.NCaltech101(root=args.root, data_type='frame', split_by='time', frames_number=args.nframes)
        train_dataset, test_dataset = split_to_train_test_set(train_ratio=0.9, origin_dataset=dataset, num_classes=args.nclasses)
    elif args.dataset == 'n_mnist':  # downloaded
        train_dataset = n_mnist.NMNIST(root=args.root, train=True, data_type='frame', split_by='time', frames_number=args.nframes)
        test_dataset = n_mnist.NMNIST(root=args.root, train=False, data_type='frame', split_by='time', frames_number=args.nframes)
    else:
        raise NotImplementedError(args.dataset)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)

    return DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=args.nworkers, pin_memory=True, drop_last=True), \
        DataLoader(test_dataset, batch_size=args.batch_size, sampler=test_sampler, num_workers=args.nworkers, pin_memory=True)
        

def load_model(args):
    
    model = EventResNet(
        in_channels=2,
        d_model=args.d_model,
        num_classes=args.nclasses,
        )

    return model


def _get_output_dir(args):

    output_dir = os.path.join(args.output_dir, f'resnet18_{args.dataset}_b{args.batch_size}_dm{args.d_model}_lr{args.lr}_T{args.nframes}')
    

    # criterion
    if args.criterion == 'CrossEntropyLoss':
        output_dir += '_ce'
    elif args.criterion == 'MSELoss':
        output_dir += '_mse'
    else:
        raise NotImplementedError(args.criterion)

    # optimizer
    if args.optimizer == 'Adam':
        output_dir += '_adam'
    elif args.optimizer == 'SGD':
        output_dir += '_sgd'
    else:
        raise NotImplementedError(args.optimizer)

    return output_dir

def train(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    test_loader: DataLoader,
    nepochs: int,
    epoch: int,
    output_dir: str,
    args: argparse.Namespace,
):  
    if is_master():
        tb_writer = SummaryWriter(output_dir + '/log')
        print('log saved to {}'.format(output_dir + '/log'))

    torch.cuda.empty_cache()
    # train 
    epoch = epoch
    while(epoch < nepochs):
        print('Epoch {}/{}'.format(epoch+1, nepochs))
        model.train()
        top1_correct = 0
        top5_correct = 0
        total = len(train_loader.dataset)
        total_loss = 0
        nsteps_per_epoch = len(train_loader)
        step = 0
        if is_master():
            import tqdm
            process_bar = tqdm.tqdm(total=nsteps_per_epoch)
        for input, label in train_loader:
            optimizer.zero_grad()
            input = input.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            input = input.transpose(0, 1)
            target = to_onehot(label, args.nclasses).cuda(non_blocking=True)
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # calculate the top5 and top1 accurate numbers
            _, predicted = output.topk(5, 1, True, True)
            top1_correct += predicted[:, 0].eq(label).sum().item()
            top5_correct += predicted.T.eq(label[None]).sum().item()
            total_loss += loss.item()
            step += 1
            if is_master():
                tb_writer.add_scalar('step_loss', loss.item(), epoch * nsteps_per_epoch + step)
                process_bar.update(1)
        if args.distributed:
            top1_correct, top5_correct, total_loss =global_meters_all_sum(args, top1_correct, top5_correct, total_loss)
        top1_accuracy = top1_correct / total * 100
        top5_accuracy = top5_correct / total * 100
        if is_master():    
            tb_writer.add_scalar('train_acc@1', top1_accuracy, epoch + 1)
            tb_writer.add_scalar('train_acc@5', top5_accuracy, epoch + 1)
            tb_writer.add_scalar('train_loss', total_loss, epoch + 1)
            process_bar.close()
        print('train_cor@1: {}, train_cor@5: {}, train_total: {}'.format(top1_correct, top5_correct, total))
        print('train_acc@1: {:.3f}%, train_acc@5: {:.3f}%, train_loss: {:.3f}'.format(top1_accuracy, top5_accuracy, total_loss))
        
        # evaluate
        model.eval()
        top1_correct = 0
        top5_correct = 0
        total = len(test_loader.dataset)
        total_loss = 0
        with torch.no_grad():
            for input, label in test_loader:
                input = input.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)
                input = input.transpose(0, 1)
                target = to_onehot(label, args.nclasses).cuda(non_blocking=True)
                output = model(input)  # batch_size, num_classes
                loss = criterion(output, target)

                # calculate the top5 and top1 accurate numbers
                _, predicted = output.topk(5, 1, True, True)  # batch_size, topk(5) 
                top1_correct += predicted[:, 0].eq(label).sum().item()
                top5_correct += predicted.T.eq(label[None]).sum().item()
                total_loss += loss.item()
        if args.distributed:
            top1_correct, top5_correct, total_loss = global_meters_all_sum(args, top1_correct, top5_correct, total_loss)
        top1_accuracy = top1_correct / total * 100
        top5_accuracy = top5_correct / total * 100
        if is_master():   
            tb_writer.add_scalar('val_acc@1', top1_accuracy, epoch + 1)
            tb_writer.add_scalar('val_acc@5', top5_accuracy, epoch + 1)
            tb_writer.add_scalar('val_loss', total_loss, epoch + 1)
        print('val_cor@1: {}, val_cor@5: {}, val_total: {}'.format(top1_correct, top5_correct, total))
        print('val_acc@1: {:.3f}%, val_acc@5: {:.3f}%, val_loss: {:.3f}'.format(top1_accuracy, top5_accuracy, total_loss))

        
        # save
        epoch += 1
        if epoch % args.save_freq == 0:
            checkpoint = {
                'model': model.embedding.embedding.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': args,
            }
            save_name = 'checkpoints/ep{}_vl{:.3f}.pth'.format(epoch, total_loss)
            save_on_master(checkpoint, os.path.join(output_dir, save_name))
            print('saved checkpoint to {}'.format(output_dir))


def main(args):
    # init distributed training
    init_dist(args)
    print(args)
    
    # device
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
    else:
        torch.cuda.set_device(args.device_id)

     # data
    train_loader, test_loader = load_data(args)

    # criterion
    criterion = nn.CrossEntropyLoss()
    args.criterion = criterion.__class__.__name__  

    # model
    model = load_model(args)
    model.cuda()
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], find_unused_parameters=True)
    params = filter(lambda p: p.requires_grad, model.parameters())

    # optimizer
    optimizer = torch.optim.Adam(params, lr=args.lr)
    args.optimizer = optimizer.__class__.__name__

    # run
    epoch = 0
    
    # output_dir
    output_dir = _get_output_dir(args)

    if is_master():
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(os.path.join(output_dir, 'checkpoints')):
            os.makedirs(os.path.join(output_dir, 'checkpoints'))
   

    train(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=test_loader,
        nepochs=args.nepochs,
        epoch=epoch,
        output_dir=output_dir,
        args=args
    )
    

if __name__ == '__main__':
    args = parser_args()
    main(args)


