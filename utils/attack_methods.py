#TODO write down adverasarail attacks method scripts in here
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from tqdm import *
import pickle
import os
from source.models import Net
from utils.common import generate_samples
import random

class Attack(Net):
    def __init__(self,args,kwargs=None):
        super(Attack,self).__init__(args,kwargs)
        if args.method == "L_BFGS":
            self.r = nn.Parameter(data=torch.zeros(1, 784),requires_grad=True) if args.model == "FFN" else nn.Parameter(data=torch.zeros(1, 1, 28 , 28), requires_grad=True)
            self.Optimizer = optim.SGD(params=[self.r], lr=0.008)

    def forward(self, x):
        if self.args.method == "L_BFGS":
            x = x + self.r
            x = torch.clamp(x, 0, 1)
        x = super(Attack, self).forward(x)
        return(x)
    def load_weights(self,weights=None):
        assert os.path.isfile(weights) , "Error: weight file {} is invalid, try training the model first".format(weights)
        #Load pre_trained model weigths
        with open(weights , "rb") as f:
            weights_dict = pickle.load(f)
        for param in self.named_parameters():
            if param[0] in weights_dict.keys():
                print("Copying: ", param[0])
                param[1].data = weights_dict[param[0]].data
        print("Weights loaded!")

    def FGSM(self):
        #TODO
        generate_samples(self.model)
        #Load Generated samples
        with open("utils/adv_examples/"+self.model+"_samples.pkl", "rb") as f:
            samples_5k = pickle.load(f)
        xs = samples_5k["images"]
        y_trues = samples_5k["labels"]
        noises = []
        y_preds = []
        y_preds_adversarial = []
        xs_clean = []
        y_trues_clean = []
        totalMisclassification = 0
        Adv_misclassification = 0
        for x, y_true in tqdm(zip(xs, y_trues)):

            #make x as Variable
            if self.args.cuda:
                x = Variable(torch.cuda.FloatTensor(x.reshape(1, 784)),
                             requires_grad=True) if self.model == "FFN" else Variable(torch.cuda.FloatTensor(x).unsqueeze(0),
                                                                                      requires_grad=True)
                y_true = Variable(torch.cuda.LongTensor(np.array([y_true])), requires_grad=False)
            else:
                x = Variable(torch.cuda.FloatTensor(x.reshape(1,784)), requires_grad=True) if self.model=="FFN" else Variable(torch.cuda.FloatTensor(x).unsqueeze(0), requires_grad=True)
                y_true = Variable(torch.cuda.LongTensor(np.array([y_true])), requires_grad=False)

            #Classify x before Adv_attack
            y_pred = np.argmax(self(x).cpu().data.numpy()) if self.args.cuda else np.argmax(self(x).data.numpy())

            #Generate Adv Image
            outputs = self(x)
            loss = self.SoftmaxWithXent(outputs, y_true)
            loss.backward() # to obtain gradients of x

            #Add small perturbation
            epsilon = 0.1
            x_grad = torch.sign(x.grad.data)
            x_adversarial = torch.clamp(x.data + epsilon*x_grad,0,1)

            #Classify after Adv_attack
            y_pred_adversarial = np.argmax(self(Variable(x_adversarial)).cpu().data.numpy()) if self.args.cuda else np.argmax(self(Variable(x_adversarial)).data.numpy())

            if self.args.cuda:
                y_true = y_true.cpu()
                x = x.cpu()
                x_adversarial = x_adversarial.cpu()
            if y_true.data.numpy() != y_pred:
                print("MISCLASSIFICATION")
                totalMisclassification +=1
            else:
                if y_pred != y_pred_adversarial:
                    Adv_misclassification += 1
                y_preds.append(y_pred)
                y_preds_adversarial.append(y_pred_adversarial)
                noises.append((x_adversarial - x.data).numpy())
                xs_clean.append(x.data.numpy())
                y_trues_clean.append(y_true.data.numpy())
        print("Total misclassifications: ", totalMisclassification , " out of :", len(xs))
        print('\nTotal misclassified adversarial examples : {} out of {}\nError_Rate is {:.0f}%'.format(
            Adv_misclassification, len(y_preds_adversarial),
            100. * Adv_misclassification / len(
                y_preds_adversarial)))
        with open("utils/adv_examples/bulk_mnist_fgsm_"+self.model+".pkl", "wb") as f:
            adv_dta_dict = {
                "xs" : xs_clean,
                "y_trues" : y_trues_clean,
                "y_preds" : y_preds,
                "noised" : noises,
                "y_preds_adversarial" : y_preds_adversarial
            }
            pickle.dump(adv_dta_dict, f)

    def L_BFGS(self, norm="l2"):
        # TODO validate results
        generate_samples(self.model)
        # Load Generated samples
        with open("utils/adv_examples/" + self.model + "_samples.pkl", "rb") as f:
            samples_5k = pickle.load(f)
        images = samples_5k["images"]
        labels = samples_5k["labels"]
        noises = []
        y_preds = []
        y_preds_adversarial = []
        xs_clean = []
        y_trues_clean = []
        totalMisclassification = 0
        Adv_misclassification = 0
        for x, l in tqdm(zip(images, labels)):

            # Random wrong label to fool the model with
            l_target = random.choice(list(set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) - set([l])))
            if self.args.cuda :
                _x = Variable(torch.cuda.FloatTensor(x)) if self.model == "FFN" else Variable(
                    torch.cuda.FloatTensor(x).unsqueeze(0))
                _l_target = Variable(torch.cuda.LongTensor(np.array([l_target])))
            else:
                _x = Variable(torch.FloatTensor(x)) if self.model == "FFN" else Variable(
                    torch.FloatTensor(x).unsqueeze(0))
                _l_target = Variable(torch.LongTensor(np.array([l_target])))

            # reset value of r
            if self.args.cuda:
                self.r.data = torch.zeros(1, 784).cuda() if self.model == "FFN" else torch.zeros(1, 1, 28 , 28).cuda()
            else:
                self.r.data = torch.zeros(1, 784) if self.model == "FFN" else torch.zeros(1, 1, 28, 28)

            # classify x before Adv_attack
            y_pred = np.argmax(self(_x).cpu().data.numpy()) if self.args.cuda else np.argmax(self(_x).data.numpy())

            if l.data.numpy() != y_pred:
                print("Image was not classified correctly")

            # Optimitzation box contrained
            for i in range(1000):
                self.Optimizer.zero_grad()
                output = self(_x)
                loss = self.SoftmaxWithXent(output, _l_target)

                # Norm used
                if norm == "l1":
                    adv_loss = loss + torch.mean(torch.abs(self.r))
                elif norm == "l2":
                    adv_loss = loss + torch.mean(torch.pow(self.r, 2))
                else:
                    adv_loss == loss

                adv_loss.backward()
                self.Optimizer.step()

                # Until output == y_target
                y_pred_adversarial = np.argmax(self(_x).cpu().data.numpy()) if self.args.cuda else np.argmax(self(_x).data.numpy())

                if y_pred_adversarial == l_target:
                    break

                if i == 999:
                    print("Results may be incorrect, Optimization run for 1000 iteration")

            if y_pred == l:
                if y_pred_adversarial != y_pred:
                    Adv_misclassification += 1
                xs_clean.append(x)
                y_trues_clean.append(l)
                y_preds.append(y_pred)
                y_preds_adversarial.append(y_pred_adversarial)
                noises.append(self.r.cpu().data.numpy().squeeze() if self.args.cuda else self.r.data.numpy().squeeze)
            else:
                print("y_pred != y_true, wrongly classified before attack -> not stored ")
                totalMisclassification += 1
        print("Total misclassifications: ", totalMisclassification, " out of :", len(images))
        print('\nTotal misclassified adversarial examples : {} out of {}\nError_Rate is {:.0f}%'.format(Adv_misclassification,len(y_preds_adversarial),100. * Adv_misclassification / len(y_preds_adversarial)))
        with open("utils/adv_examples/bulk_mnist_lbfgs_" + self.model + ".pkl", "wb") as f:
            adv_dta_dict = {
                "xs": xs_clean,
                "y_trues": y_trues_clean,
                "y_preds": y_preds,
                "noised": noises,
                "y_preds_adversarial": y_preds_adversarial
            }
            pickle.dump(adv_dta_dict, f)