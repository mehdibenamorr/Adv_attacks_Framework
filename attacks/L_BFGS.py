import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import pickle
from utils.common import generate_samples,vis_adv_org
from utils.attacks import  l_bfgs
import random
from .Attacker import Attack


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
