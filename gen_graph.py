import numpy as np
from igraph import *
import argparse
from models.models import Net
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim  as optim
import torch.nn.functional as F
from random import randint
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='Generate SNNs and train')
parser.add_argument('--model', type=str, default="FFN",
                    help='model to train (default: FFN)')
parser.add_argument('--batch-size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=256,
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add_argument('--weight_decay', type=float, default=1e-04,
                    help='weigth_decay rate')
parser.add_argument('--nodes', type=int, default=100,
                    help='number of nodes (default: 10)')
parser.add_argument('--m', type=int, default=3,
                    help='number of edges to attach a new node to existing ones (default: 3)')
parser.add_argument('--p', type=float, default=0.5,
                    help='probability for edge creation (default: 0.5)')
parser.add_argument('--k', type=int, default=3,
                    help='Each node is joined with its k nearest neighbors in a ring topology (default: 1)')


args = parser.parse_args()


nodes = [100,200,300,400,500,600,700,800,1000]


def generate_random_dag(N, k, p):

    g = Graph.Watts_Strogatz(1,N,k,p)
    Adj_matrix = np.tril(np.array(g.get_adjacency().data)).tolist()
    g = Graph.Adjacency(Adj_matrix)
    g.to_directed()
    return g


def layer_indexing(g):
    vertices_index = [-1 for i in range(len(g.vs))]
    for v in g.vs:
        if v.indegree() == 0:
            vertices_index[v.index] = 0
    while -1 in vertices_index:
        for v in g.vs:
            if v.indegree() > 0 and vertices_index[v.index] == -1 :
                in_edges = v.predecessors()
                ind = { vertices_index[vv.index] for vv in in_edges }
                if -1 not in ind:
                    vertices_index[v.index] = max(ind.union({-1})) + 1
                else:
                    continue
    vertex_by_layers = [ [] for k in range(max(vertices_index)+1)]
    for i in range(len(vertices_index)):
        vertex_by_layers[vertices_index[i]].append(g.vs[i])
    return vertex_by_layers


class Layer(nn.Module):
    def __init__(self, in_dims, out_dim, vertices, predecessors, bias=True):
        super(Layer,self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.predecessors = predecessors
        self.vertices = vertices
        self.weights = []
        self.act_masks = []
        self.w_masks = []
        for i,pred in enumerate(self.predecessors):
            # mask = torch.ByteTensor(np.zeros((out_dim, in_dims[i])))
            # act_mask = torch.ByteTensor(np.zeros(in_dims[i]))
            mask = torch.zeros((out_dim, in_dims[i]))
            act_mask = torch.zeros(in_dims[i])
            for j,v in enumerate(self.vertices):
                for p in v.predecessors():
                    if p in pred:
                        ind = pred.index(p)
                        mask[j, ind] = 1
                        act_mask[ind] = 1
            self.act_masks.append(act_mask)
            self.w_masks.append(mask)
            import ipdb
            ipdb.set_trace()
            # self.weights.append(nn.Parameter(torch.normal(mean=torch.zeros(out_dim,in_dims[i]).masked_select(mask), std=0.1)))
            self.weights.append(nn.Parameter(torch.normal(mean=torch.zeros(out_dim,in_dims[i]), std=0.1)))

        if bias:
            self.bias = nn.Parameter(torch.normal(mean=torch.zeros(out_dim,1), std=0.1))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        output = torch.zeros(self.out_dim,1)
        for i, inp in enumerate(inputs):
            output += self.weights[i].mul(self.w_masks[i]).mm(inp)
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
        self.layers = []
        for i in range(1,len(vertex_by_layers)):
            self.layers.append(Layer([len(layer) for layer in vertex_by_layers[:i]],
                                         len(vertex_by_layers[i]),vertex_by_layers[i],vertex_by_layers[:i]))
        import ipdb
        ipdb.set_trace()

    def forward(self, x):
        activations = []
        x = F.relu(self.input_layer(x))
        activations.append(x)
        for layer in self.layers:
            activations.append(F.relu(layer(activations)))
        x = self.output_layer(activations[-1])

        return x


model = SNN(args)







