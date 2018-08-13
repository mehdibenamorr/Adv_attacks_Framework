import configargparse
import torch
from models.models import models
import os
import torch.backends.cudnn as cudnn
import numpy as np
import signal
import pandas as pd
import time

# Training settings
parser = configargparse.ArgParser()
parser.add('-c','--config-file', required=False, is_config_file= True,help='config file path')

parser.add('--model', type=str, default="FFN",
                    help='model to attack (default: FFN)')
parser.add('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add('--test-batch-size', type=int, default=100,
                    help='input batch size for testing (default: 100)')
parser.add('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add('--weight_decay', type=float, default=1e-04,
                    help='weigth_decay rate')
parser.add('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add('--resume', '-r', action='store_true', help='resume training from checkpoint')
parser.add('--cuda', action='store_true', help='build the model on GPU')
parser.add('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add('--nodes', type=int, default=350,
                    help='number of nodes for SNN training (default: 10)')
parser.add('--layers', type=int, default=-1,
                    help='a parameter for the experience between FFN and SNN (default: -1)')
parser.add('--m', type=int, default=10,
                    help='number of edges to attach a new node to existing ones (default: 3)')
parser.add('--p', type=float, default=0.8,
                    help='probability for edge creation (default: 0.5)')
parser.add('--k', type=int, default=6,
                    help='Each node is joined with its k nearest neighbors in a ring topology (default: 1)')

parser.add('--method', type=str, default="FGSM",
                    help='method to use for the adversarial attack (default: FGSM)')
parser.add('--norm', type=str, default="l2",
                    help='regularization norm (default: l2)')
parser.add('--max_iter', type=int, default=100,
                    help='maximum iter for DE algorithm (default: 100)')
parser.add('--pixels', type=int, default=1,
                    help='The number of pixels that can be perturbed.(default: 1)')
parser.add('--popsize', default=400, type=int, help='The number of adverisal examples in each iteration.')
parser.add('--samples', default=100, type=int, help='The number of image samples to attack.')
parser.add('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add('--V', action='store_true', default=False,
                    help='visualize generated adversarial examples')

def keyboardInterruptHandler(signal, frame):
    print("Training has been stopped. Cleaning up...")
    exit(0)

args = parser.parse_args()
args.cuda = torch.cuda.is_available() and args.cuda

path_to_results = "tests/results/Implementation_exp_100.csv"
df = pd.read_csv(path_to_results, encoding='utf-8', index_col=0)



torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers' : 4} if args.cuda else {}
accs = []
times = []
for i in range(1000):
    start_epoch = 0
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        if os.path.isfile(args.config_file+'.ckpt'):
            checkpoint = torch.load(args.config_file+'.ckpt')
            model = checkpoint['net']
            best_acc = checkpoint['acc']
            start_epoch = checkpoint['epoch']
        else:
            print('No checkpoint found for this model')
            print('==> Building model..')
            model = models[args.model](args, kwargs)
    else:
        print('==> Building model..')
        model = models[args.model](args,kwargs)

    if args.cuda:
        model.cuda()
        # import ipdb
        # ipdb.set_trace()
        # model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
        Net = model
    else:
        Net = model
    Net.Dataloader()
    signal.signal(signal.SIGINT, keyboardInterruptHandler)

    # Computing elapsed time
    start_time = time.clock()
    for epoch in range(start_epoch, args.epochs):
        Net.trainn(epoch)
        Net.test(epoch)
    elapsed_time = time.clock() - start_time
    best_acc = Net.best_acc
    print('Run: {} Acc : {:.3f}% Time_elapsed: {} seconds'.format(i,best_acc,elapsed_time))
    accs.append(best_acc)
    times.append(elapsed_time)


df[args.model+'_acc'] = np.array(accs)
df[args.model+'_Time'] = np.array(elapsed_time)

df.to_csv(path_to_results)

