from attacks.attack_methods import *
import configargparse
import torch
import torch.backends.cudnn as cudnn
from models.models import models
from utils.common import generate_SNNs
import torch.nn.init as init
import os
import pandas as pd
import numpy as np
from utils.logger import Logger


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
parser.add('--saved_models', type=str, default="tests/results/Trained_FFNs",
                    help='path to saved trained models (default: tests/results/)')
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
    torch.cuda.manual_seed(args.seed)

init_functions = [{'xavier_normal': init.xavier_normal_, 'kwargs': {'gain': init.calculate_gain('relu')}}
    ,{'xavier_uniform_': init.xavier_uniform_, 'kwargs' : {'gain': init.calculate_gain('relu')}}
    ,{'He_normal': init.kaiming_normal_, 'kwargs' : {'a': 0, 'mode': 'fan_in', 'nonlinearity': 'relu'}}
    ,{'He_uniform': init.kaiming_uniform_, 'kwargs' : {'a': 0, 'mode': 'fan_in', 'nonlinearity': 'relu'}}
    ,{'normal': init.normal_, 'kwargs' : {'mean': 0.0 , 'std': 0.1}}
    ,{'uniform' : init.uniform_, 'kwargs': {'a': -0.1, 'b': 0.1}}]


class Experiment(object):
    def __init__(self,args,kwargs, experiment, path):
        # Experiment setting
        self.args = args
        self.kwargs = kwargs
        self.experiment = experiment
        self.attacks = ['FGSM','One_Pixel']
        self.path_to_results = path + experiment + "/"+ experiment + ".csv"
        self.nodes = [150, 200, 250, 300, 350, 400, 500]  # for FFN experiment
        self.ks = [2,4,6,8,10,20]
        self.ps = [0.5,0.6,0.7,0.8,0.9]
        # self.N = 10  # number of repetitions
        self.params = range(50000,91000)  #range of parameters for SNNs generation
        self.nb_SNNs = 100
        self.Trained_models={}
        self.Results = {}
        self.attacks_data = {}

    def load_models(self, path = None):
        if path is not None and os.path.isfile(path):
            self.Trained_models = torch.load(path)
        elif os.path.isfile(self.args.path + self.experiment + "/Trained_SNNs_{}.pkl".format(str(self.params))):
            # import ipdb
            # ipdb.set_trace()
            self.Trained_models = torch.load(self.args.path + self.experiment + "/Trained_SNNs_{}.pkl".format(str(self.params)))
        else:
            print("This file {} was not found ".format(self.args.path + self.experiment +"/Trained_SNNs_{}.pkl".format(str(self.params))))
            print("Starting training...")
            self.train_models()

    def train_models(self):
        if self.args.model == 'FFN':
            if os.path.isfile(self.args.path + self.experiment +"/Trained_FFNs.pkl"):
                self.Trained_models = torch.load(self.args.path + self.experiment +"/Trained_FFNs.pkl")
            else:
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
                torch.save(self.Trained_models, self.args.path + self.experiment +"/Trained_FFNs.pkl")
        elif self.args.model == 'SNN':
            if os.path.isfile(self.args.path + self.experiment +"/Generated_SNNS_graphs_{}.pkl".format(str(self.params))):
                graphs = torch.load(self.args.path + self.experiment +"/Generated_SNNS_graphs_{}.pkl".format(str(self.params)))
            else:
                #generate SNNs with parameters in range of (79500,89500) parameters
                #TODO save generated graph structures / Done
                graphs = generate_SNNs(self.params, self.args, self.kwargs, self.nb_SNNs, self.nodes, self.ks,self.ps)
                #save generated graphs
                torch.save(graphs, self.args.path + self.experiment +"/Generated_SNNS_graphs_{}.pkl".format(str(self.params)))
            for idx ,(params, graph) in enumerate(graphs):
                for init_func in init_functions:
                    method_name = list(init_func.keys())[0]
                    snn = SNN(args, kwargs, graph,
                              logger=Logger('logs/'+self.experiment+'/'+'SNN_'+str(idx) + '/' + method_name),
                              nodes=params['nodes'], k=params['k'], p=params['p'],
                              init_method=init_func[method_name], **init_func['kwargs'])

                    if self.args.cuda:
                        snn.cuda()
                        cudnn.benchmark = True
                    self.Trained_models[self.experiment + "_" +
                                        str(snn.count_parameters()) + "_" +
                                        str(snn.args.nodes) + "_" +
                                        str(snn.args.k) + "_" +
                                        str(snn.args.p)] = {}
                    # for i in range(self.N):
                    print("==> Training model.." + self.args.experiment + "_" +
                          str(snn.count_parameters()) + "_" +
                          str(snn.args.nodes) + "_" +
                          str(snn.args.k) + "_" +
                          str(snn.args.p) + "_" +
                          method_name)
                    snn.Dataloader()
                    snn.structural_properties()
                    for epoch in range(self.args.epochs):
                        snn.trainn(epoch)
                        snn.test(epoch)
                    snn.del_logger()
                    self.Trained_models[self.experiment + "_" +
                                        str(snn.count_parameters()) + "_" +
                                        str(snn.args.nodes) + "_" +
                                        str(snn.args.k) + "_" +
                                        str(snn.args.p)][method_name] = snn.best_state
            # save trained models
            torch.save(self.Trained_models , self.args.path + self.experiment +
                       "/Trained_SNNs_{}.pkl".format(str(self.params)))


    def attack_models(self):
        # Attack all trained models and store the results
        self.Results = {}
        self.attacks_data = {}

        for model in self.Trained_models.keys():
            self.attacks_data[model] = {}
            self.Results[model] = {}
            for init_func in self.Trained_models[model].keys():
                self.attacks_data[model][init_func]= {}
                self.Results[model][init_func] = {}
                Logger
                for attack in self.attacks:
                    print("==> Attacking {} __ {} with {} ".format(model, init_func, attack))
                    self.Results[model][init_func][attack] = {}
                    net = self.Trained_models[model][init_func]['model']
                    self.Results[model][init_func][attack].update(net.get_structural_properties())
                    attacker = attacks[attack](self.args, Net=net, logger=Logger('logs/' + self.experiment +
                                                                                 '/' + model +
                                                                                 '/' + init_func+'/'+attack))
                    if self.args.cuda:
                        attacker.cuda()
                    dta = attacker.attack()
                    self.attacks_data[model][init_func][attack] = dta
                    if attack == "FGSM":
                        dta_ep = attacker.attack_eps()
                        self.Results[model][init_func][attack].update({'Avg_epsilon' : dta_ep['Avg_epsilon'] ,
                                                                       'Max_epsilon' : dta_ep['Max_epsilon'],
                                                                       'Min_epsilon': dta_ep['Min_epsilon']})
                        self.attacks_data[model][init_func]['FGSM_eps'] = dta_ep
                    # self.Results[model]['Robustness'].append(dta['Success_Rate'])
                    self.Results[model][init_func][attack].update({'Robustness': dta['Success_Rate'],
                                                                   'Avg_confidence': np.mean(dta['Confidences']),
                                                                   'Max_confidence': np.max(dta['Confidences']),
                                                                   'Accuracy': net.best_acc})

        df = pd.DataFrame.from_dict(self.Results, orient='index')
        df.to_csv(self.path_to_results)
        with open("utils/adv_examples/"+ self.experiment + str(self.params)+ ".pkl","wb") as f:
            pickle.dump(self.attacks_data, f)
        return df

# models = args.saved_models


exp = Experiment(args,kwargs, args.experiment, args.path)
exp.load_models()
import ipdb
ipdb.set_trace()
results = exp.attack_models()

