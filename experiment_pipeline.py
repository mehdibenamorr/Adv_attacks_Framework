from attacks.attack_methods import *
import configargparse
import torch
import torch.backends.cudnn as cudnn
from models.models import models
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


# Experiment setting
experiment = 'FFN_1Layer'
attack = 'FGSM'
path_to_results = "tests/results/experiment_FFN_1Layer.csv"
nodes = [50,100,150,200,250,300,350,400]
N = 10  # number of repetitions


# Train All the models and store them temporarily
Trained_models = {}
for n in nodes:
    Trained_models[experiment + "_" + str(n)] = []
    args.nodes = n
    for i in range(N):
        print('==> Building model..' + experiment + "_" + str(n) + "_" + str(i))
        model = models[args.model](args, kwargs)
        if args.cuda:
            model.cuda()
            cudnn.benchmark = True

        model.Dataloader()
        for epoch in range(args.epochs):
            model.trainn(epoch)
            model.test(epoch)
        Trained_models[experiment + "_" + str(n)].append(model.best_state)
# import ipdb
# ipdb.set_trace()

# Attack all trained models and store the results
Results = {}
for model in Trained_models.keys():
    Results[model] = {'Robustness': [], 'Accuracy': []}
    for rep in Trained_models[model]:
        net = rep['model']
        Results[model]['Accuracy'].append(net.best_acc)
        attacker = attacks[attack](args, Net=net)
        if args.cuda:
            attacker.cuda()
        dta = attacker.attack()
        Results[model]['Robustness'].append(dta['Sucess_Rate'])

df = pd.DataFrame.from_dict(Results, orient='index')
df.to_csv(path_to_results)
