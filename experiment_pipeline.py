from attacks.attack_methods import *
import configargparse
import torch
import torch.backends.cudnn as cudnn
from models.models import models
from utils.common import generate_SNNs
import os
import pandas as pd
import numpy as np


parser = configargparse.ArgumentParser()
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
parser.add('--experiment', type=str, default="SNN_experiment",
                    help='the name of the experiment (default: sNN_experiment)')
parser.add('--path', type=str, default="tests/results/",
                    help='path to save the results (default: tests/results/)')
parser.add('--resume', '-r', action='store_true', help='resume training from checkpoint')
parser.add('--save', action='store_true', help='save checkpoints when training')
parser.add('--cuda', action='store_true', help='build the model on GPU')
parser.add('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add('--nodes', type=int, default=200,
                    help='number of nodes for SNN training (default: 200)')
parser.add('--layers', type=int, default=-1,
                    help='number of layers of the fully connected network (default: -1)')
parser.add('--m', type=int, default=3,
                    help='number of edges to attach a new node to existing ones (default: 3)')
parser.add('--p', type=float, default=0.5,
                    help='probability for edge creation (default: 0.5)')
parser.add('--k', type=int, default=3,
                    help='Each node is joined with its k nearest neighbors in a ring topology (default: 3)')

parser.add('--method', type=str, default="FGSM",
                    help='method to use for the adversarial attack (default: FGSM)')
parser.add('--epsilon', type=float, default=0.1,
                    help='parameter of FGSM (default: 0.1)')
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
args.cuda = torch.cuda.is_available() and args.cuda

torch.manual_seed(args.seed)
kwargs = {'num_workers' : 4} if args.cuda else {}

if args.cuda:
    torch.cuda.manual_seed(args.seed)


class Experiment(object):
    def __init__(self,args,kwargs, experiment, attack, path):
        # Experiment setting
        self.args = args
        self.kwargs = kwargs
        self.experiment = experiment
        self.attack = attack
        self.path_to_results = path + experiment + ".csv"
        self.nodes = [50, 100, 150, 200, 250, 300, 350, 400]  # for FFN experiment
        self.ks = [3,4,5,6,8,10,]
        self.ps = [0.5,0.6,0.7,0.8]
        self.N = 10  # number of repetitions
        self.params = range(69500,89500)  #range of parameters for SNNs generation
        self.nb_SNNs = 10
        self.Trained_models={}
        self.Results = {}

    def train(self):
        if self.args.model == 'FFN':
            # Train All the models and store them temporarily
            # import ipdb
            # ipdb.set_trace()
            for n in self.nodes:
                # self.Trained_models[self.experiment + "_" + str(n)] = []
                self.args.nodes = n
                # for i in range(self.N):
                print('==> Building model..' + self.args.experiment + "_" + str(n) )
                model = models[args.model](self.args, self.kwargs)
                if self.args.cuda:
                    model.cuda()
                    cudnn.benchmark = True

                model.Dataloader()
                for epoch in range(self.args.epochs):
                    model.trainn(epoch)
                    model.test(epoch)
                self.Trained_models[self.args.experiment + "_" + str(n)] = model.best_state
            torch.save(self.Trained_models, self.args.path + "Trained_FFNs.pkl")
        elif self.args.model == 'SNN':
            #generate SNNs with parameters in range of (79500,89500) parameters
            #TODO save generated graph structures
            SNNs , graphs = generate_SNNs(self.params, self.args, self.kwargs, self.nb_SNNs, self.nodes[2:], self.ks,self.ps)
            for snn in SNNs:
                # self.Trained_models[self.experiment + "_" + str(snn.count_parameters()) + "_" + str(snn.args.nodes) + "_" + str(snn.args.k) + "_" + str(snn.args.p)] = []
                # for i in range(self.N):
                print("==> Training model.." + self.args.experiment + "_" + str(snn.count_parameters()) + "_" + str(snn.args.nodes)
                      + "_" + str(snn.args.k) + "_" + str(snn.args.p))
                if self.args.cuda:
                    snn.cuda()
                    cudnn.benchmark = True
                snn.Dataloader()
                for epoch in range(self.args.epochs):
                    snn.trainn(epoch)
                    snn.test(epoch)
                self.Trained_models[self.experiment + "_" + str(snn.count_parameters()) + "_" + str(snn.args.nodes) + "_" + str(snn.args.k) + "_" + str(snn.args.p)]=snn.best_state
            # save generated and trained models and TODO graphs
            torch.save(SNNs, self.args.path + "Generated_SNNS_graphs.pkl")
            torch.save(self.Trained_models , self.args.path + "Trained_SNNs_normal_init.pkl")


    def attack(self):
        # Attack all trained models and store the results
        self.Results = {}
        for model in self.Trained_models.keys():
            self.Results[model] = {'Robustness': dta['Success_Rate'], 'Accuracy': net.best_acc,
                                   '#params': self.Trained_models[model]['#params']}
            # for rep in self.Trained_models[model]:
            net = self.Trained_models[model]['model']
            # self.Results[model]['Accuracy'].append(net.best_acc)
            attacker = attacks[self.attack](self.args, Net=net)
            if self.args.cuda:
                attacker.cuda()
            dta = attacker.attack()
            # self.Results[model]['Robustness'].append(dta['Success_Rate'])
            self.Results[model] = {'Robustness': dta['Success_Rate'], 'Accuracy': net.best_acc,
                                   '#params': self.Trained_models[model]['#params']}

        df = pd.DataFrame.from_dict(self.Results, orient='index')
        df.to_csv(self.args.path_to_results)
        return df


exp = Experiment(args,kwargs, args.experiment, args.method, args.path)
exp.train()
results = exp.attack()
