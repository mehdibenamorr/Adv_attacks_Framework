from attacks.attack_methods import *
import configargparse
import torch
from torch.nn import init
import torch.backends.cudnn as cudnn
from models.models import models
from utils.common import generate_SNNs
import os
import pandas as pd
import numpy as np


parser = configargparse.ArgumentParser()
parser.add('-c','--config-file', required=False, is_config_file= True,help='config file path')
parser.add('--model', type=str, default="SNN",
                    help='model to attack (default: FFN)')
parser.add('--batch-size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add('--test-batch-size', type=int, default=100,
                    help='input batch size for testing (default: 100)')
parser.add('--epochs', type=int, default=30,
                    help='number of epochs to train (default: 100)')
parser.add('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add('--weight_decay', type=float, default=0,
                    help='weigth_decay rate')
parser.add('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add('--experiment', type=str, default="SNN_pruning_experiment",
                    help='the name of the experiment (default: SNN_pruning_experiment)')
parser.add('--path', type=str, default="tests/results/",
                    help='path to save the results (default: tests/results/)')
parser.add('--saved_models', type=str, default="tests/results/Trained_models_pruning.pkl",
                    help='path to saved trained models (default: tests/results/Trained_models_pruning.pkl)')
parser.add('--resume', '-r', action='store_true', help='resume training from checkpoint')
parser.add('--save', action='store_true', help='save checkpoints when training')
parser.add('--cuda', action='store_true', help='build the model on GPU')
parser.add('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add('--nodes', type=int, default=200,
                    help='number of nodes for SNN training (default: 200)')
parser.add('--layers', type=int, default=3,
                    help='number of layers of the fully connected network (default: 3)')
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

#Experiment Hyper parameters
N = 10 #number of repetitions
attacks = ['FGSM', 'One_Pixel']
path_to_results = args.path + 'pruning_experiment.csv'
Trained_models = args.saved_models

#TODO write the training part of the models to prune and run it then write down the pruning algorithm
# Init functions to use when initializing weights
init_functions = [{'xavier_normal': init.xavier_normal_, 'kwargs': {'gain': init.calculate_gain('relu')}}
    ,{'xavier_uniform_': init.xavier_uniform_, 'kwargs' : {'gain': init.calculate_gain('relu')}}
    ,{'He_normal': init.kaiming_normal_, 'kwargs' : {'a': 0, 'mode': 'fan_in', 'non_linearity': 'relu'}}
    ,{'He_uniform': init.kaiming_uniform_, 'kwargs' : {'a': 0, 'mode': 'fan_in', 'non_linearity': 'relu'}}
    ,{'normal': init.normal_, 'kwargs' : {'mean': 0.0 , 'std': 0.1}}
    ,{'uniform' : init.uniform_, 'kwargs': {'a': -0.1, 'b': 0.1}}]

# Generate the models to prune and train them OR load if already done that

if os.path.isfile(Trained_models):
    # import ipdb
    # ipdb.set_trace()
    models = torch.load(Trained_models)

else:
    print("This file was not found %s" % Trained_models)
    models = {}
    print("==> Building and Training models...")
    for init_func in init_functions:
        method_name = list(init_func.keys())[0]
        models[method_name] = []
        for i in range(N):
            model = SNN(args, kwargs, init_method=init_func[method_name], **init_func['kwargs'])
            if args.cuda:
                model.cuda()
                cudnn.benchmark = True

            model.Dataloader()
            for epoch in range(args.epochs):
                model.trainn(epoch)
                model.test(epoch)
            models[method_name].append(model)
    print("Training done!")
    torch.save(models,Trained_models)


# Pruning, Training, updating, evaluating robustness for each attack along with computing structural properties


