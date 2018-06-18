import cairocffi as cairo
import numpy as np
from igraph import *
import argparse
import torch
from random import randint
import matplotlib.pyplot as plt

# Training settings
parser = argparse.ArgumentParser(description='Generate SNNs')
parser.add_argument('--nodes', type=int, default=100,
                    help='number of nodes (default: 10)')
parser.add_argument('--m', type=int, default=3,
                    help='number of edges to attach a new node to existing ones (default: 3)')
parser.add_argument('--p', type=float, default=0.5,
                    help='probability for edge creation (default: 0.5)')
parser.add_argument('--k', type=int, default=2,
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
            if v.indegree() > 0 and vertices_index[v.index]== -1 :
                in_edges = v.predecessors()
                ind = { vertices_index[vv.index] for vv in in_edges }
                if -1 not in ind:
                    vertices_index[v.index] = max(ind.union({-1})) + 1
                else:
                    continue

    return vertices_index


graph = generate_random_dag(args.nodes,args.k,args.p)
vertices_index = layer_indexing(graph)
print(vertices_index)
l = graph.layout('fr')
plot(graph, layout=l)
import ipdb
ipdb.set_trace()






