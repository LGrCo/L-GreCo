from __future__ import print_function

import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data.distributed
from torchvision import datasets, transforms
import os
from models import resnet18_psgd
from tqdm import tqdm

import json

from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed

from hooks import lgreco_hook as lgreco
from hooks import powerSGD_hook as powerSGD
import time

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CIFAR10_TRAIN_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_TRAIN_STD = (0.2023, 0.1994, 0.2010)

NCOLS_SCREEN=85

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset-dir', default=os.path.expanduser('./cifar10'),
                    help='path to training data')

parser.add_argument('--powersgd-rank', type=int, default=None,
                    help='Rank of powersgd compression to run DDP with')

parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')

parser.add_argument('--batch-size', type=int, default=256,
                    help='input batch size for training')
parser.add_argument('--val-batch-size', type=int, default=256,
                    help='input batch size for validation')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.1,
                    help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=1e-4,
                    help='weight decay')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')
parser.add_argument('--dist-backend', choices=['qmpi', 'nccl', 'gloo'], default='nccl',
                    help='Backend for torch distributed')
parser.add_argument('--adjust-freq', type=int, default=1,
                    help='Frequency of running adjusting algorithm')
parser.add_argument('--method', type=str, default='lgreco',
                    help='Compression Method; powerSGD, lgreco')
parser.add_argument('--local_rank', type=int, default=-1,
                    help='Local rank in distributed launch')

args = parser.parse_args()
args.local_rank = int(os.environ["LOCAL_RANK"])
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.distributed.init_process_group(backend=args.dist_backend)
args.world_size = torch.distributed.get_world_size()
args.distributed = args.world_size > 1
args.local_rank = torch.distributed.get_rank()
rank = args.local_rank
local_rank = args.local_rank % torch.cuda.device_count()


print(args)

if args.cuda:
    torch.cuda.set_device(local_rank)
    torch.cuda.manual_seed(args.seed)

cudnn.benchmark = True

verbose = 1 if rank == 0 else 0

# Horovod: limit # of CPU threads to be used per worker.
torch.set_num_threads(4)

is_cifar100 = "cifar100" in args.dataset_dir
if is_cifar100:
    transform_mean, transform_std = CIFAR100_TRAIN_MEAN, CIFAR100_TRAIN_STD
else:
    transform_mean, transform_std = CIFAR10_TRAIN_MEAN, CIFAR10_TRAIN_STD

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(transform_mean, transform_std),
])

if is_cifar100:
    train_dataset = datasets.CIFAR100(root=args.dataset_dir, train=True, download=True, transform=transform_train)
else:
    train_dataset = datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=transform_train)

train_sampler = torch.utils.data.distributed.DistributedSampler(
    train_dataset, num_replicas=args.world_size, rank=rank)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=args.batch_size,
    sampler=train_sampler, **kwargs)

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(transform_mean, transform_std),
])

if is_cifar100:
    val_dataset = datasets.CIFAR100(root=args.dataset_dir, train=False, download=True, transform=transform_test)
else:
    val_dataset = datasets.CIFAR10(root=args.dataset_dir, train=False, download=True, transform=transform_test)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size,
                                         **kwargs)

# Set up standard ResNet-20 model.
# model = resnet32(num_classes=100)
if is_cifar100:
    num_classes = 100
else:
    num_classes = 10
model = resnet18_psgd(num_classes=num_classes)


if args.cuda:
    # Move model to GPU.
    model.cuda()

optimizer = optim.SGD(model.parameters(),
                      lr=args.base_lr,
                      momentum=args.momentum, weight_decay=args.wd)
if args.distributed:
    rank = torch.distributed.get_rank()
    model = DDP(model, device_ids=[rank], output_device=rank)
    if args.powersgd_rank:
        rank = torch.distributed.get_rank()
        model = DDP(model, device_ids=[rank], output_device=rank)
        if args.method == 'lgreco':
            print('L-GreCo')
            state = lgreco.PowerSGDState(torch.distributed.group.WORLD,
                                           matrix_approximation_rank=args.powersgd_rank,
                                           adjust_freq=len(train_loader))
            model.register_comm_hook(state, lgreco.powerSGD_hook)
        if args.method == 'powerSGD':
            print('powerSGD')
            state = powerSGD.PowerSGDState(torch.distributed.group.WORLD,
                                           matrix_approximation_rank=args.powersgd_rank)
            model.register_comm_hook(state, powerSGD.powerSGD_hook)

def adjust_learning_rate(epoch, batch_idx):
    if epoch < 60:
        lr_adj = 1.
    elif epoch < 120:
        lr_adj = 2e-1
    elif epoch < 160:
        lr_adj = 4e-2
    else:
        lr_adj = 8e-3
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.base_lr * lr_adj


def train(epoch):
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    train_loss = Metric('train_loss')
    train_accuracy = Metric('train_accuracy')

    with tqdm(total=len(train_loader),
              desc='TEpoch #{}'.format(epoch + 1), disable=not verbose, ncols=NCOLS_SCREEN) as t:
        for batch_idx, (data, target) in enumerate(train_loader):
            adjust_learning_rate(epoch, batch_idx)
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = model(data)
            train_accuracy.update(accuracy(output, target))
            loss = criterion(output, target)
            train_loss.update(loss)
            t.set_postfix({'loss': train_loss.avg.item(),
                           'accuracy': 100. * train_accuracy.avg.item()})
            loss.backward()
            optimizer.step()
            t.update(1)

def validate(epoch):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    val_loss = Metric('val_loss')
    val_accuracy = Metric('val_accuracy')
    with tqdm(total=len(val_loader),
              desc='Validate Epoch  #{}'.format(epoch + 1),
              disable=not verbose, ncols=NCOLS_SCREEN) as t:
        with torch.no_grad():
            for data, target in val_loader:
                if args.cuda:
                    data, target = data.cuda(), target.cuda()
                output = model(data)
                val_loss.update(criterion(output, target))

                val_accuracy.update(accuracy(output, target))
                t.set_postfix({'loss': val_loss.avg.item(),
                               'accuracy': 100. * val_accuracy.avg.item()})
                t.update(1)


def accuracy(output, target):
    # get the index of the max log-probability
    pred = output.max(1, keepdim=True)[1]
    return pred.eq(target.view_as(pred)).cpu().float().mean()


# Horovod: average metrics from distributed training.
class Metric(object):
    def __init__(self, name):
        self.name = name
        self.sum = torch.tensor(0.)
        self.n = torch.tensor(0.)

    def update(self, val):
        self.sum += val.detach().cpu()
        self.n += 1

    @property
    def avg(self):
        return self.sum / self.n


num_images = len(train_loader)
root = args.local_rank == 0
start_time = time.time()
for epoch in range(0, args.epochs):
    if args.world_size > 1:
        train_sampler.set_epoch(epoch)
    train(epoch)
    validate(epoch)
        
end_time = time.time()
total_time = end_time - start_time
print("total training time = " + str(total_time))