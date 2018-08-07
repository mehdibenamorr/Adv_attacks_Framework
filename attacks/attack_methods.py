import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.backends import cudnn
import pickle
import os
from models.models import Net,FFN,CNN,SNN
from utils.common import generate_samples,vis_adv_org
from utils.attacks import one_pixel,fgsm , l_bfgs
import random
import matplotlib.pyplot as plt

class Attack(Net):
    def __init__(self,args,kwargs=None):
        super(Attack,self).__init__(args,kwargs)
        # if self.model == "FFN":
        #     self.Net = FFN(args,kwargs)
        # elif self.model == "CNN":
        #     self.Net = CNN(args,kwargs)
        # elif self.model == "SNN":
        #     self.Net = SNN(args,kwargs)
        self.Net = self.load_model(self.args.config_file+'.ckpt')
        if args.cuda:
            self.Net.cuda()
            cudnn.benchmark = True
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


class FGSM(Attack):
    def __init__(self,args,kwargs=None):
        super(FGSM,self).__init__(args,kwargs)

    def forward(self, x):
        return self.Net(x)

    def attack(self):
        test_loader = generate_samples(self.model)
        # Load Generated samples
        # with open("data/" + self.model + "/10k_samples.pkl", "rb") as f:
        #     samples_10k = pickle.load(f)
        print("=> Samples loaded. Starting FGSM attack....")
        # xs = samples_10k["images"]
        # y_trues = samples_10k["labels"]
        noises = []
        y_preds = []
        y_preds_adversarial = []
        xs_clean = []
        y_trues_clean = []
        totalMisclassification = 0
        Adv_misclassification = 0
        for batch_idx , (x, y_true) in enumerate(test_loader):

            # make x as Variable
            if self.args.cuda:
                x = Variable(x.cuda(),requires_grad=True) if self.model != "CNN" else \
                    Variable(x.unsqueeze(0).cuda(),requires_grad=True)
                y_true = Variable(y_true, requires_grad=False).cuda()
            else:
                x = Variable(x,requires_grad=True) if self.model != "CNN" else \
                    Variable(x.unsqueeze(0), requires_grad=True)
                y_true = Variable(y_true, requires_grad=False)

            # Classify x before Adv_attack

            y_pred = np.argmax(self(x).cpu().data.numpy()) if self.args.cuda else np.argmax(self(x).data.numpy())

            if y_true.data.item() != y_pred :
                #print("MISCLASSIFICATION")
                totalMisclassification += 1
                continue

            #generate an adversarial example
            x_adversarial = fgsm(self,x,y_true,epsilon=0.1)

            # Classify after Adv_attack
            y_pred_adversarial = np.argmax(self(Variable(x_adversarial)).cpu().data.numpy()) if self.args.cuda else np.argmax(
                self(Variable(x_adversarial)).data.numpy())

            if self.args.cuda:
                y_true = y_true.cpu()
                x = x.cpu()
                x_adversarial = x_adversarial.cpu()

            if y_pred != y_pred_adversarial:
                Adv_misclassification += 1
                if self.args.V :
                    vis_adv_org(x,x_adversarial,y_pred,y_pred_adversarial)
            y_preds.append(y_pred)
            y_preds_adversarial.append(y_pred_adversarial)
            noises.append((x_adversarial - x.data).numpy())
            xs_clean.append(x.data.numpy())
            y_trues_clean.append(y_true.data.numpy())
        print("Total misclassifications: ", totalMisclassification, " out of :", len(test_loader.dataset))
        print('\nTotal misclassified adversarial examples : {} out of {}\nSuccess_Rate is {:.3f}%'.format(
            Adv_misclassification, len(y_preds_adversarial),
            100. * Adv_misclassification / len(
                y_preds_adversarial)))
        with open("utils/adv_examples/mnist_fgsm_" + self.args.config_file + ".pkl", "wb") as f:
            adv_dta_dict = {
                "xs": xs_clean,
                "y_trues": y_trues_clean,
                "y_preds": y_preds,
                "noised": noises,
                "y_preds_adversarial": y_preds_adversarial
            }
            pickle.dump(adv_dta_dict, f)


class One_Pixel(Attack):
    def __init__(self,args,kwargs=None):
        super(One_Pixel,self).__init__(args,kwargs)

    def forward(self, x):
        return self.Net(x)

    def attack(self):
        test_loader = generate_samples(self.model)
        # # Load Generated samples
        # with open("data/" + self.model + "/10k_samples.pkl", "rb") as f:
        #     samples_10k = pickle.load(f)
        print("=> Samples loaded. Starting One pixel attack....")
        # xs = samples_10k["images"]
        # y_trues = samples_10k["labels"]
        noises = []
        y_preds = []
        y_preds_adversarial = []
        xs_clean = []
        y_trues_clean = []
        totalMisclassification = 0
        correct = 0
        Adv_misclassification = 0
        success_rate = 0
        for batch_idx, (x, y_true) in enumerate(test_loader):

            if self.model == "CNN":
                x = x.unsqueeze(0)
            with torch.no_grad():
                img_var = Variable(x).cuda() if self.args.cuda else Variable(x)
            prior_probs = F.softmax(self.Net(img_var),dim=1)
            y_pred = np.argmax(prior_probs.cpu().data.numpy()) if self.args.cuda else np.argmax(prior_probs.data.numpy())

            if y_true.data.item() != y_pred :
                totalMisclassification+=1
                continue

            correct += 1
            y_true = y_true.data.item()
            targets = [None] if not self.args.targeted else range(10)

            for target_class in targets:
                if self.args.targeted:
                    if target_class==y_true:
                        continue

                flag, x, adv_img = one_pixel(x,y_true, self.Net,self.model,target_class, pixels=self.args.pixels,
                                    maxiter=self.args.max_iter, popsize= self.args.popsize,cuda=self.args.cuda)
                Adv_misclassification += flag
                if self.args.targeted:
                    success_rate = float(Adv_misclassification)/(9*correct)
                else:
                    success_rate = float(Adv_misclassification)/correct

                if flag == 1:
                    print("success rate : {:.4f}  ({}/{}) [(x,y) = ({},{}) and I = {}]".format(success_rate,Adv_misclassification,
                                                                                               correct,x[0],x[1],x[2]))
            if correct == self.args.samples:
                break

        print("Total misclassifications: ", totalMisclassification, " out of :", self.args.samples)
        print('\nTotal misclassified adversarial examples : {}/{} \nSuccess_Rate is {:.3f}%'.format(
            Adv_misclassification,correct,
            100. * success_rate))


class L_BFGS(Attack):

    def __init__(self,args,kwargs=None):
        super(L_BFGS,self).__init__(args,kwargs)
        self.r = nn.Parameter(data=torch.zeros(1, 784), requires_grad=True) if args.model == "FFN" else nn.Parameter(
            data=torch.zeros(1, 1, 28, 28), requires_grad=True)
        if self.args.cuda:
            self.r.cuda()
        self.Optimizer = optim.SGD(params=[self.r], lr=0.008)

    def forward(self, x):
        x = x + self.r
        x = torch.clamp(x, 0, 1)
        x = self.Net(x)
        return (x)

    def attack(self):
        # TODO validate results
        test_loader = generate_samples(self.model)
        # # Load Generated samples
        # with open("data/" + self.model + "/10k_samples.pkl", "rb") as f:
        #     samples_10k = pickle.load(f)
        # images = samples_10k["images"]
        # labels = samples_10k["labels"]
        noises = []
        y_preds = []
        y_preds_adversarial = []
        xs_clean = []
        y_trues_clean = []
        totalMisclassification = 0
        Adv_misclassification = 0
        for batch_idx , (x, l) in enumerate(test_loader):

            # if self.args.targeted:
            # Random wrong label to fool the model with
            l_target = random.choice(list(set([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) - set([l])))
            if self.args.cuda:
                _x = Variable(x.cuda()) if self.model != "CNN" else Variable(
                    x.unsqueeze(0).cuda())
                _l_target = Variable(torch.cuda.LongTensor(np.array([l_target])))
            else:
                _x = Variable(x) if self.model != "CNN" else Variable(
                    x.unsqueeze(0))
                _l_target = Variable(torch.LongTensor(np.array([l_target])))

            # reset value of r
            if self.args.cuda:
                self.r.data = torch.zeros(1, 784).cuda() if self.model != "CNN" else torch.zeros(1, 1, 28, 28).cuda()
            else:
                self.r.data = torch.zeros(1, 784) if self.model != "CNN" else torch.zeros(1, 1, 28, 28)

            # classify x before Adv_attack
            y_pred = np.argmax(self(_x).cpu().data.numpy()) if self.args.cuda else np.argmax(self(_x).data.numpy())

            if l.data.item() != y_pred:
                print("Image was not classified correctly")
                print("y_pred != y_true, wrongly classified before attack -> not stored ")
                totalMisclassification += 1
                continue
            #generate adversarial example
            x_adversarial , y_pred_adversarial = l_bfgs(self,_x,_l_target,self.args.norm,self.args.max_iter)

            if y_pred_adversarial != y_pred:
                Adv_misclassification += 1
                if self.args.V :
                    vis_adv_org(_x.cpu(),x_adversarial.cpu(),y_pred,y_pred_adversarial,l_target)
            xs_clean.append(x)
            y_trues_clean.append(l)
            y_preds.append(y_pred)
            y_preds_adversarial.append(y_pred_adversarial)
            noises.append(self.r.cpu().data.numpy().squeeze() if self.args.cuda else self.r.data.numpy().squeeze)

        print("Total misclassifications: ", totalMisclassification, " out of :", len(test_loader.dataset))
        print('\nTotal misclassified adversarial examples : {} out of {}\nSuccess_Rate is {:.3f}%'.format(
            Adv_misclassification, len(y_preds_adversarial), 100. * Adv_misclassification / len(y_preds_adversarial)))
        with open("utils/adv_examples/mnist_lbfgs_" + self.model + ".pkl", "wb") as f:
            adv_dta_dict = {
                "xs": xs_clean,
                "y_trues": y_trues_clean,
                "y_preds": y_preds,
                "noised": noises,
                "y_preds_adversarial": y_preds_adversarial
            }
            pickle.dump(adv_dta_dict, f)

attacks = {'FGSM' : FGSM, 'L_BFGS' : L_BFGS, 'One_Pixel': One_Pixel}