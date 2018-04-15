from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from tqdm import *
from models import Net

# Training settings
parser = argparse.ArgumentParser(description='Train Mnist models')
parser.add_argument('--model', type=str, default="FFN", metavar='MD',
                    help='model to train (default: FFN)')
parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--weight_decay', type=float, default=1e-04, metavar='W',
                    help='weigth_decay rate')
parser.add_argument('--attack', action='store_true', default=False,
                    help='attack after training?')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers' : 1 , 'pin_memory': True} if args.cuda else {}

#write the trainng and generation




if __name__=="__main__":
    model = Net(args,kwargs)

    if args.cuda:
        model.cuda()

    for epoch in tqdm(range(1,args.epochs+1)):
        model.trainn(epoch)
        model.test()
    model.save()