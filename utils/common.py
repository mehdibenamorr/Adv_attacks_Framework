import torch
from torchvision import datasets, transforms
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from igraph import *
import torch.nn as nn
# from models.models import SNN  #TODO fix this import issue
import models.models

def flat_trans(x):
    x.resize_(28*28)

    return x


def generate_samples(model):
    #,transforms.Normalize((0.1307,), (0.3081,))
    mnist_transform = transforms.Compose([ transforms.ToTensor(),transforms.Lambda(flat_trans)]) if model != "CNN" else \
        transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('data/'+model, train=False, transform=mnist_transform),
        batch_size=1, shuffle=True, num_workers= 2)
    # images, labels = [], []
    # for idx, data in enumerate(test_loader):
    #     x_lots, y_lots = data
    #     for x, y in zip(x_lots, y_lots):
    #         images.append(x.numpy())
    #         labels.append(y)
    #
    # data_dict = {"images": images, "labels": labels}
    # with open("data/"+model+"/10k_samples.pkl", "wb") as f:
    #     pickle.dump(data_dict, f)
    return test_loader


def vis_adv_org(x_org, x_adv,y_pred,y_adv,target=None):
    style = {"figure.figsize": (5, 3.75),
             "figure.titlesize": 11,
             "legend.frameon": False,
             "legend.loc": 'upper right',
             "legend.fontsize": 11,
             "axes.labelsize": 11,
             "axes.titlesize": 11,
             "savefig.bbox": 'tight',
             "savefig.pad_inches": 0.05,
             "savefig.dpi": 300,
             "xtick.direction": 'in',
             "xtick.labelsize": 11,
             "xtick.major.size": 4,
             "xtick.major.width": 2,
             "xtick.minor.size": 2,
             "xtick.minor.width": 1,
             "xtick.minor.visible": True,
             "xtick.top": False,
             "xtick.bottom": True,
             "ytick.direction": 'in',
             "ytick.labelsize": 11,
             "ytick.major.size": 4,
             "ytick.major.width": 2,
             "ytick.minor.size": 2,
             "ytick.minor.width": 1,
             "ytick.minor.visible": True,
             "ytick.right": False,
             "ytick.left": True
             }
    plots_path = 'plots/'
    x_adv = x_adv.cpu().reshape(1,28,28)
    noise = (x_adv - x_org).data.numpy().reshape(28,28)
    x_org = x_org.data.numpy().reshape(28,28)
    x_adv = x_adv.data.numpy().reshape(28,28)
    disp_im = np.concatenate((x_org,x_adv),axis=1)
    if target:
        plt.title("Original : {}    Target/Adv : {}/{}    Perturbation : {:.5f}".format(y_pred,target,y_adv,noise.mean()))
    else:
        plt.title("Original: {}      Adversarial : {}".format(y_pred,y_adv))
    plt.imshow(disp_im,cmap='gray')
    plt.savefig(plots_path + 'onepixel_example.pdf')


def generate_random_dag(N, k, p, dense=-1):
    if dense == 1:
        g = Graph()
        for x in range(N):
            g.add_vertex(x)
        adj_matrix = np.tril(np.array(g.get_adjacency().data))
        g = Graph.Adjacency((adj_matrix>0).tolist())
        g.to_directed()
    elif dense == 2:
        g = Graph()
        for x in range(N):
            g.add_vertex(x)
        for i in range(int(N/2)):
            for j in range(int(N/2)):
                g.add_edge(i,int(N/2)+j)
        adj_matrix = np.tril(np.array(g.get_adjacency().data))
        g = Graph.Adjacency((adj_matrix>0).tolist())
        g.to_directed()
    elif dense == 3:
        g = Graph()
        for x in range(170):
            g.add_vertex(x)
        for i in range(100):
            for j in range(50):
                g.add_edge( 100 + j,i)
        for j in range(50):
            for k in range(20):
                g.add_edge( 150 + k,100 + j)
        adj_matrix = np.triu(np.array(g.get_adjacency().data))
        g = Graph.Adjacency((adj_matrix > 0).tolist())
        g.to_directed()
    elif dense == 4:
        g = Graph()
        for x in range(300):
            g.add_vertex(x)
        for i in range(50):
            for j in range(100):
                g.add_edge( 50 + j,i)
        for j in range(100):
            for k in range(100):
                g.add_edge( 150 + k,50 + j)
        for k in range(100):
            for l in range(50):
                g.add_edge(250 + l, 150 + k)
        adj_matrix = np.triu(np.array(g.get_adjacency().data))
        g = Graph.Adjacency((adj_matrix > 0).tolist())
        g.to_directed()
    else:
        g = Graph.Watts_Strogatz(1,N,k,p)
        adj_matrix = np.tril(np.array(g.get_adjacency().data))
        g = Graph.Adjacency((adj_matrix>0).tolist())
        g.to_directed()
    return g


def layer_indexing(g):
    # vertices_index = [-1 for i in range(len(g.vs))]
    # for v in g.vs:
    #     if v.indegree() == 0:
    #         vertices_index[v.index] = 0
    # num_unindexed_vertices = len(g.vs)
    unindexed_vertices = [v for v in g.vs if v.indegree() > 0]
    vertices_index = [0 if v.indegree() < 1 else -1 for v in g.vs]
    while len(unindexed_vertices) > 0:
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
        if g.vs[i].outdegree() < 1:
            vertex_by_layers[-1].append(g.vs[i])
        else:
            vertex_by_layers[vertices_index[i]].append(g.vs[i])

    return vertex_by_layers


def generate_SNNs(params_range, args, kwargs, nb=10, nodes=None, ks=None, ps=None):
    from models.models import SNN
    print("==> Generating SNNs with #paramerters in " + str(params_range))
    nb_SNNs = 0
    graph_structures = []
    while nb_SNNs < nb:
        if (nodes is not None) and (ks is not None) and (ps is not None):
            for node in nodes:
                for k in ks:
                    for p in ps:
                        snn = SNN(args,kwargs,nodes=node,k=k,p=p)
                        if args.cuda: snn.cuda()
                        print("nodes:",str(node),"k: ",str(k),"p: ",str(p), "#params: ", str(snn.count_parameters()))
                        if snn.count_parameters() in params_range:
                            params = {'nodes': node,'k': k,'p':p, '#params' : snn.count_parameters()}
                            graph_structures.append((params,snn.structure_graph()))
                            print("saved")
                            nb_SNNs+=1
                            if nb_SNNs == nb:
                                return graph_structures
        else:
            snn = SNN(args,kwargs)
            print("nodes:", str(args.nodes), "k: ", str(args.k), "p: ", str(args.p), "#params: ", str(snn.count_parameters()))
            if snn.count_parameters() in params_range:
                params = {'nodes': args.nodes, 'k': args.k, 'p': args.p ,'#params' : snn.count_parameters()}
                graph_structures.append((params,snn.structure_graph()))
                print("saved")
                nb_SNNs += 1
    return graph_structures

