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
attacks_ = ['FGSM', 'One_Pixel']
path_to_results = args.path + Experiment + '/pruning_experiment'
Trained_models = args.saved_models


# Init functions to use when initializing weights
init_functions = [{'xavier_normal': init.xavier_normal_, 'kwargs': {'gain': init.calculate_gain('relu')}}
    ,{'xavier_uniform_': init.xavier_uniform_, 'kwargs' : {'gain': init.calculate_gain('relu')}}
    ,{'He_normal': init.kaiming_normal_, 'kwargs' : {'a': 0, 'mode': 'fan_in', 'nonlinearity': 'relu'}}
    ,{'He_uniform': init.kaiming_uniform_, 'kwargs' : {'a': 0, 'mode': 'fan_in', 'nonlinearity': 'relu'}}
    ,{'normal': init.normal_, 'kwargs' : {'mean': 0.0 , 'std': 0.1}}
    ,{'uniform' : init.uniform_, 'kwargs': {'a': -0.1, 'b': 0.1}}]

# Generate the models to prune and train them OR load if already done that

if os.path.isfile(Trained_models):

    models = torch.load(Trained_models)

else:
    print("This file was not found %s" % Trained_models)
    models = {}
    print("==> Building and Training models...")
    for init_func in init_functions[:10]:
        method_name = list(init_func.keys())[0]
        models[method_name] = []
        for i in range(N):
            model = SNN(args, kwargs, logger=Logger('./logs/'+Experiment+'/'+method_name+'_run'+ str(i)), init_method=init_func[method_name], **init_func['kwargs'])
            if args.cuda:
                model.cuda()
                cudnn.benchmark = True
            model.Dataloader()
            model.structural_properties()
            model.count_parameters()
            for epoch in range(args.epochs):
                model.trainn(epoch)
                model.test(epoch)
            model.del_logger()
            models[method_name].append(model)
    print("Training done!")
    torch.save(models,Trained_models)


# Pruning, Training, updating, evaluating robustness for each attack along with computing structural properties
# import ipdb
# ipdb.set_trace()
import time
Pruning_steps = 20
N=0
while N < 10:
    Results = {}
    attacks_data = {}
    models_time = time.time()
    for name in models:
        model_time = time.time()
        Results[name] = {}
        attacks_data[name] = {}
        for model in models[name][N:N+1]:
            run = N
            Results[name]['run_'+str(run)] = {}
            attacks_data[name]['run' + str(run)] = {}
            logger = Logger('./logs/'+Experiment+'/'+name+'_run'+str(run)+'/pruning_steps') # TODO add logging
            step = 0
            epochs = 2
            pruned_pct = 0
            stop = False
            while (step < Pruning_steps) and not stop:
                Results[name]['run_' + str(run)]['Pruning_step_'+str(step)] = {}
                attacks_data[name]['run' + str(run)] ['Pruning_step_'+str(step)] = {}
                if args.pruning == 'magnitude':
                    model.prune(args.alpha)
                else:
                    model.prune_random(args.alpha)
                print('Pruning step {}'.format(step))
                print('previously pruned : {:.3f}%'.format(100*pruned_pct))
                print('number_pruned: {:.3f}%'.format(100*(model.num_pruned/model.num_weights)))
                print('newly_pruned : {:.3f}%'.format(100*(model.num_pruned/model.num_weights - pruned_pct)))

                new_pruned = model.num_pruned / model.num_weights - pruned_pct
                pruned_pct = model.num_pruned/model.num_weights
                f1_score = model.validate()

                model.stats['num_pruned'].append(pruned_pct)
                model.stats['new_pruned'].append(new_pruned)
                model.stats['f1_score'].append(f1_score)

                model.structural_properties()
                print('model weights : {}'.format(model.count_parameters()))

                #Retraining
                for e in range(epochs):
                    model.train_pruned()
                    f1_score= model.validate()

                    print('Retraining epoch {} : F1_score : {:.5f} % '.format(
                        e,100*f1_score
                    ))

                # Log values and gradients of the parameters (histogram summary) after each pruning step
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
                        {'Accuracy': f1_score , 'Num_pruned': pruned_pct, 'New_pruned' : new_pruned})
                    attacker = attacks[attack](args, Net=net , logger=logger)
                    if args.cuda:
                        attacker.cuda()

                    dta = attacker.attack()
                    attacks_data[name]['run' + str(run)]['Pruning_step_' + str(step)][attack] = dta
                    Results[name]['run_' + str(run)]['Pruning_step_' + str(step)][attack].update(
                        {'Robustness': dta['Success_Rate'],
                         'Avg_confidence': np.mean(dta['Confidences']),
                         'Max_confidence': np.max(dta['Confidences'])})
                    if attack == 'FGSM':
                        dta_ep = attacker.attack_eps()
                        attacks_data[name]['run' + str(run)]['Pruning_step_' + str(step)]['FGSM_eps'] = dta_ep
                        Results[name]['run_' + str(run)]['Pruning_step_' + str(step)][attack].update(
                            {'Avg_epsilon': dta_ep['Avg_epsilon'],
                             'Max_epsilon': dta_ep['Max_epsilon'],
                             'Min_epsilon': dta_ep['Min_epsilon']})

                # Stopping criterion
                if new_pruned <= 0.001:
                    print('stopping pruning')
                    stop = True

                step += 1
        df1 = pd.DataFrame.from_dict(Results[name], orient='index')
        df1.to_csv(path_to_results + '_'+ name + '_run_' + str(N)+'.csv')
        print("Attacking this model took {} minutes".format((time.time()-model_time)/60.))
    print("Attacking all models for one run took {}".format((time.time()-models_time)/60.))

    df = pd.DataFrame.from_dict(Results, orient='index')
    df.to_csv(path_to_results+'_run_'+str(N)+'.csv')
    with open("utils/adv_examples/" + Experiment + "_gpu_" + str(args.gpu) + "_run_"+str(N)+".pkl", "wb") as f:
        pickle.dump(attacks_data, f)

    N += 1



# import ipdb
# ipdb.set_trace()
# df = pd.DataFrame.from_dict(Results, orient='index')
# df.to_csv(path_to_results)
# with open("utils/adv_examples/" + Experiment + "_gpu_"+ str(args.gpu) +".pkl", "wb") as f:
#     pickle.dump(attacks_data, f)







