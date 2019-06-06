from attacks.attack_methods import *
from models.models import *
import configargparse
import torch
from torch.nn import init
import torch.backends.cudnn as cudnn
from utils.logger import Logger
# from models.models import models
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
parser.add('--pruning', type=str, default="random",
                    help='pruning method (default: random)')
parser.add('--alpha', type=float, default=0.1,
                    help='parameter of the pruning (default: 0.1)')
parser.add('--saved_models', type=str, default="tests/results/Trained_models_pruning.pkl",
                    help='path to saved trained models (default: tests/results/Trained_models_pruning.pkl)')
parser.add('--resume', '-r', action='store_true', help='resume training from checkpoint')
parser.add('--save', action='store_true', help='save checkpoints when training')
parser.add('--cuda', action='store_true', help='build the model on GPU')
parser.add('--gpu', type=int, default=0,
                    help='on which gpu to run (default: 0')
parser.add('--log-interval', type=int, default=128,
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
parser.add('--popsize', default=400, type=int, help='The number of adversarial examples in each iteration.')
parser.add('--samples', default=100, type=int, help='The number of image samples to attack.')
parser.add('--targeted', action='store_true', help='Set this switch to test for targeted attacks.')
parser.add('--V', action='store_true', default=False,
                    help='visualize generated adversarial examples')

args = parser.parse_args()
args.cuda = torch.cuda.is_available() and args.cuda

torch.manual_seed(args.seed)
kwargs = {'num_workers' : 4} if args.cuda else {}

if args.cuda:
    torch.cuda.set_device(args.gpu)
    torch.cuda.manual_seed(args.seed)

#logger
# logger = Logger('./logs')


#Experiment Hyper parameters
Experiment = args.experiment
N = 10 #number of repetitions
attacks_ = ['One_Pixel']
path_to_results = args.path + Experiment + '/pruning_experiment'
Trained_models = args.saved_models





models = torch.load(Trained_models)




# Pruning, Training, updating, evaluating robustness for each attack along with computing structural properties
# import ipdb
# ipdb.set_trace()
import time
Pruning_steps = 20
N=0
Results = {}

while N < 10:
    Results = {}

    models_time = time.time()
    for name in models:
        model_time = time.time()
        Results[name] = {}

        for model in models[name][N:N+1]:
            run = N
            Results[name]['run_'+str(run)] = {}
            step=0
            logger = Logger('./logs/' + Experiment + '/' + name + '_run' + str(run) + '/pruning_steps')
            Results[name]['run_' + str(run)]['Pruning_step_'+str(step)] = {}

            for tag, value in model.named_parameters():
                tag = tag.replace('.', '/')
                # import ipdb
                # ipdb.set_trace()
                logger.histo_summary(tag, value.data.cpu().numpy(), step + 1)
                try:
                    logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), step + 1)
                except AttributeError:
                    print('No gradient data for this parameter')
            # evaluate Robustness (FGSM, FGSM_eps, One_Pixel)
            for attack in attacks_:
                print("==> Attacking {} __ run {} with {} ".format(name, run, attack))
                net = model
                Results[name]['run_' + str(run)]['Pruning_step_' + str(step)][attack] = {}
                Results[name]['run_' + str(run)]['Pruning_step_' + str(step)][attack].update(
                    model.get_structural_properties())
                Results[name]['run_' + str(run)]['Pruning_step_' + str(step)][attack].update(
                    {'Accuracy': model.best_acc })
                attacker = attacks[attack](args, Net=net,logger=logger)
                if args.cuda:
                    attacker.cuda()

                dta = attacker.attack()

                Results[name]['run_' + str(run)]['Pruning_step_' + str(step)][attack].update(
                    {'Robustness': dta['Success_Rate'],
                     'Avg_confidence': np.mean(dta['Confidences']),
                     'Max_confidence': np.max(dta['Confidences'])})
                if attack == 'FGSM':
                    dta_ep = attacker.attack_eps()
                    Results[name]['run_' + str(run)]['Pruning_step_' + str(step)][attack].update(
                        {'Avg_epsilon': dta_ep['Avg_epsilon'],
                         'Max_epsilon': dta_ep['Max_epsilon'],
                         'Min_epsilon': dta_ep['Min_epsilon']})

        print("Attacking this model took {} minutes".format((time.time()-model_time)/60.))
    print("Attacking all models for one run took {}".format((time.time()-models_time)/60.))

    df = pd.DataFrame.from_dict(Results, orient='index')
    df.to_csv(path_to_results+'_run_'+str(N)+'_initial.csv')

    N += 1









