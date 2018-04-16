#TODO write down adverasarail attacks method scripts in here
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import *
import pickle

import os
from source.models import Net
from common import generate_samples


class Attack(Net):
    def __init__(self,args,kwargs=None):
        super(Attack,self).__init__(args,kwargs)
    def load_weights(self,weights=None):
        assert os.path.isfile(weights) , "Error: weight file {} is invalid, try training the model first".format(weights)
        #Load pre_trained model weigths
        weights_dict = {}
        with open(weights , "rb") as f:
            weights_dict = pickle.load(f)
        for param in self.named_parameters():
            if param[0] in weights_dict.keys():
                print("Copying: ", param[0])
                param[1].data = weights_dict[param[0]].data
        print("Weights loaded!")

    def fgsm(self):
        #TODO
        generate_samples(self.model)
        #Load Generated samples
        with open("../utils/adv_examples/"+self.model+"_samples.pkl", "rb") as f:
            samples_5k = pickle.load(f)
        xs = samples_5k["images"]
        y_trues = samples_5k["labels"]
        noises = []
        y_preds = []
        y_preds_adversarial = []
        xs_clean = []
        y_trues_clean = []
        totalMisclassification = 0
        for x, y_true in tqdm(zip(xs, y_trues)):

            #make x as Variable
            x = Variable(torch.FloatTensor(x.reshape(1,784)), requires_grad=True) if self.model=="FFN" else Variable(torch.FloatTensor(x).unsqueeze(0), requires_grad=True)
            y_true = Variable(torch.LongTensor(np.array([y_true])), requires_grad=False)

            #Classify x before Adv_attack
            y_pred = np.argmax(self(x).data.numpy())

            #Generate Adv Image
            outputs = self(x)
            loss = F.nll_loss(outputs, y_true)
            loss.backward() # to obtain gradients of x

            #Add small perturbation
            epsilon = 0.1
            x_grad = torch.sign(x.grad.data)
            x_adversarial = torch.clamp(x.data + epsilon*x_grad,0,1)

            #Classify after Adv_attack
            y_pred_adversarial = np.argmax(self(Variable(x_adversarial)).data.numpy())

            if y_true.data.numpy() != y_pred:
                print("MISCLASSIFICATION")
                totalMisclassification +=1
            else:
                y_preds.append(y_pred)
                y_preds_adversarial.append(y_pred_adversarial)
                noises.append((x_adversarial - x.data).numpy())
                xs_clean.append(x.data.numpy())
                y_trues_clean.append(y_true.data.numpy())
        print("Total missclassifications: ", totalMisclassification , " out of :", len(xs))

        with open("../utils/adv_examples/bulk_mnist_fgsm_"+self.model+".pkl", "wb") as f:
            adv_dta_dict = {
                "xs" : xs_clean,
                "y_trues" : y_trues_clean,
                "y_preds" : y_preds,
                "noised" : noises,
                "y_preds_adversarial" : y_preds_adversarial
            }
            pickle.dump(adv_dta_dict, f)
