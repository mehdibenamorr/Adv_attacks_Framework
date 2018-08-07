import numpy as np
from igraph import *

import configargparse
from torch.backends import cudnn

from models.models import Net
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils.common import flat_trans
from random import randint
import matplotlib.pyplot as plt
from paddll.sparse import SparsePyTorchNet
from paddll.data.mnist import MnistDataProvider
from paddll.graphs import *
import itertools

parser = configargparse.ArgParser()


# Training settings
# parser = argparse.ArgumentParser(description='Generate SNNs and train')
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
parser.add('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add('--nodes', type=int, default=350,
                    help='number of nodes for SNN training (default: 10)')
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


args = parser.parse_args()

args.cuda = torch.cuda.is_available()


torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers' : 4 , 'pin_memory': True} if args.cuda else {}




def generate_random_dag(N, k, p):

    g = Graph.Watts_Strogatz(1,N,k,p)
    Adj_matrix = np.tril(np.array(g.get_adjacency().data)).tolist()
    g = Graph.Adjacency(Adj_matrix)
    g.to_directed()

    # g = Graph()
    # for x in range(200):
    #     g.add_vertex(x)
    # for i in range(100):
    #     for j in range(100):
    #         g.add_edge(i,100+j)
    # Adj_matrix = np.tril(np.array(g.get_adjacency().data)).tolist()
    # g = Graph.Adjacency(Adj_matrix)
    # g.to_directed()
    return g

def epoch_printer(epoch, accuracy_test, epoch_error, accuracy_delta=None, accuracy_window_mean=None):
    print('Epoch: {} Accuracy_test : {}  Loss : {}'.format(
        epoch, accuracy_test, epoch_error))


class JSNN(SparsePyTorchNet):
    def __init__(self, args, kwargs=None):
        super(JSNN, self).__init__(784,10,generate_random_dag(args.nodes, args.k, args.p))
        self.build()
        self.train_epochs = args.epochs
        self.epoch_controller = epoch_printer
        # self.converge_in(15, 0.03)
    def Dataloader(self):
        self.dataprovider = MnistDataProvider()

    def cuda(self,device=None):
        self._torch_model.cuda()

    def forward(self, x):
        x = self.forward(x)

        return x


model = JSNN(args,kwargs)
# import ipdb
# ipdb.set_trace()
# if args.cuda:
#     model.cuda()
#     cudnn.benchmark = True
model.Dataloader()

model.train()
print(model.accuracy_test)
#
# import ipdb
# ipdb.set_trace()
# for epoch in range(1, args.epochs + 1):
#     model.trainn(epoch)
#     model.test(epoch)
# import ipdb
# ipdb.set_trace()






