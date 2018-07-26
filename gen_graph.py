import numpy as np
from igraph import *

import configargparse
from models.models import Net
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim  as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from utils.common import flat_trans
from random import randint
import matplotlib.pyplot as plt


parser = configargparse.ArgParser()


# Training settings
# parser = argparse.ArgumentParser(description='Generate SNNs and train')
parser.add('-c','--config-file', required=True, is_config_file= True,help='config file path')
parser.add('--model', type=str, default="SNN",
                    help='model to train (default: SNN)')
parser.add('--batch-size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add('--test-batch-size', type=int, default=256,
                    help='input batch size for testing (default: 256)')
parser.add('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add('--weight_decay', type=float, default=1e-04,
                    help='weigth_decay rate')
parser.add('--nodes', type=int, default=350,
                    help='number of nodes (default: 10)')
parser.add('--m', type=int, default=10,
                    help='number of edges to attach a new node to existing ones (default: 3)')
parser.add('--p', type=float, default=0.8,
                    help='probability for edge creation (default: 0.5)')
parser.add('--k', type=int, default=6,
                    help='Each node is joined with its k nearest neighbors in a ring topology (default: 1)')


args = parser.parse_args()
import ipdb
ipdb.set_trace()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers' : 1 , 'pin_memory': True} if args.cuda else {}

# nodes = [100,200,300,400,500,600,700,800,1000]


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


def layer_indexing(g):
    # vertices_index = [-1 for i in range(len(g.vs))]
    # for v in g.vs:
    #     if v.indegree() == 0:
    #         vertices_index[v.index] = 0
    # num_unindexed_vertices = len(g.vs)
    unindexed_vertices = [v for v in g.vs if v.indegree()>0]
    vertices_index = [0 if v.indegree()<1 else -1 for v in g.vs]
    while len(unindexed_vertices) > 0 :
        for v in unindexed_vertices:
            in_edges = v.predecessors()
            ind = { vertices_index[vv.index] for vv in in_edges }
            if -1 not in ind:
                vertices_index[v.index] = max(ind) + 1
                unindexed_vertices.remove(v)
            else:
                continue
    vertex_by_layers = [ [] for k in range(max(vertices_index)+1)]
    for i in range(len(vertices_index)):
        vertex_by_layers[vertices_index[i]].append(g.vs[i])
    # import ipdb
    # ipdb.set_trace()
    return vertex_by_layers


class Layer(nn.Module):
    def __init__(self, in_dims, out_dim, vertices, predecessors, cuda, bias=True):
        super(Layer,self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.predecessors = predecessors
        self.vertices = vertices
        self.cuda = cuda
        weights = []
        self.act_masks = []
        self.w_masks = []
        for i,pred in enumerate(self.predecessors):
            mask = torch.zeros((out_dim, in_dims[i]))
            if cuda:
                mask = mask.cuda()
            act_mask = torch.zeros(in_dims[i])
            for j,v in enumerate(self.vertices):
                for p in v.predecessors():
                    if p in pred:
                        ind = pred.index(p)
                        mask[j, ind] = 1
                        act_mask[ind] = 1
            self.act_masks.append(act_mask)
            self.w_masks.append(mask)
            # import ipdb
            # ipdb.set_trace()
            weights.append(nn.Parameter(torch.normal(mean=torch.zeros(out_dim,in_dims[i]), std=0.1)))
        self.weights = nn.ParameterList(weights)
        if bias:
            self.bias = nn.Parameter(torch.normal(mean=torch.zeros(out_dim), std=0.1))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        output = torch.zeros(self.out_dim)
        if self.cuda:
            output = output.cuda()
        for i, inp in enumerate(inputs):
            output = output.add(inp.matmul(self.weights[i].mul(self.w_masks[i]).t()) + self.bias)
        return output


class SNN(Net):
    def __init__(self, args, kwargs=None):
        super(SNN, self).__init__(args,kwargs)
        self.graph = generate_random_dag(args.nodes,args.k,args.p)
        vertex_by_layers = layer_indexing(self.graph)
        l = self.graph.layout('fr')
        plot(self.graph, layout=l)
        # Using matrix multiplactions
        self.input_layer = nn.Linear(784, len(vertex_by_layers[0]))
        self.output_layer = nn.Linear(len(vertex_by_layers[-1]),10)
        layers = []
        for i in range(1,len(vertex_by_layers)):
            layers.append(Layer([len(layer) for layer in vertex_by_layers[:i]],
                                         len(vertex_by_layers[i]),vertex_by_layers[i],vertex_by_layers[:i],self.args.cuda))
        self.layers = nn.ModuleList(layers)
        # import ipdb
        # ipdb.set_trace()

    def Dataloader(self):
        # self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum,
        #                            weight_decay=self.args.weight_decay)
        self.optimizer = optim.Adam(self.parameters())
        mnist_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(flat_trans)]
        )
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/FFN', train=True, download=True,
                           transform=mnist_transform),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/FFN', train=False, transform=mnist_transform),
            batch_size=self.args.test_batch_size, shuffle=True, **self.kwargs)

    def forward(self, x):
        activations = []
        x = F.relu(self.input_layer(x))
        activations.append(x)
        for layer in self.layers:
            activations.append(F.relu(layer(activations)))
        x = F.relu(self.output_layer(activations[-1]))

        return x


model = SNN(args,kwargs)
if args.cuda:
    model.cuda()
model.Dataloader()

# import ipdb
# ipdb.set_trace()
for epoch in range(1, args.epochs + 1):
    model.trainn(epoch)
    model.test(epoch)
# import ipdb
# ipdb.set_trace()






