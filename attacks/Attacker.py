import torch
from torch.backends import cudnn
import os
from models import Net


class Attack(Net):
    def __init__(self,args,kwargs=None,Net=None,logger=None):
        super(Attack,self).__init__(args,kwargs)
        if Net is not None:
            self.Net = Net
        else:
            self.Net = self.load_model(self.args.config_file+'.ckpt')
        self.best_acc = self.Net.best_acc
        if args.cuda:
            self.Net.cuda()
            cudnn.benchmark = True
        self._logger=logger
    def load_model(self,path=None):
        #TODO invoke training in case the model has no checkpoint?
        print('==> Loading the model..')
        assert os.path.isfile(path), 'Error: no checkpoint found for this model. Do you want to train it ?(Y/N)'
        checkpoint = torch.load(path)
        net = checkpoint['net']
        print('==> Model loaded successfully')
        return net
        # assert os.path.isfile(weights) , "Error: weight file {} is invalid, try training the model first".format(weights)
        # #Load pre_trained model weigths
        # with open(weights , "rb") as f:
        #     weights_dict = pickle.load(f)
        # for param in self.Net.named_parameters():
        #     if param[0] in weights_dict.keys():
        #         print("Copying: ", param[0])
        #         param[1].data = weights_dict[param[0]].data
        # print("Weights loaded!")







