import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from utils.common import generate_samples,vis_adv_org
from utils.attacks import fgsm
from .Attacker import Attack


class FGSM(Attack):
    def __init__(self,args,kwargs=None,Net=None, logger=None):
        super(FGSM,self).__init__(args,kwargs,Net=Net,logger=logger)
        self.epsilon = args.epsilon
    def forward(self, x):
        return self.Net(x)

    def attack(self,epsilon=None):
        if epsilon is not None:
            self.epsilon = epsilon
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
        adversarial_confidences = []
        xs_clean = []
        y_trues_clean = []
        totalMisclassification = 0
        Adv_misclassification = 0
        correct = 0
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
            correct += 1
            #generate an adversarial example
            x_adversarial = fgsm(self,x,y_true,self.epsilon)

            # Classify after Adv_attack
            outputs = self(Variable(x_adversarial))
            y_pred_adversarial = np.argmax(outputs.data.cpu().numpy()) if self.args.cuda else np.argmax(
                outputs.data.numpy())
            adversarial_confidence = np.max(F.softmax(outputs, dim=1).data.cpu().numpy()) if self.args.cuda else \
                np.max(F.softmax(outputs, dim=1).data.numpy())

            if self.args.cuda:
                y_true = y_true.cpu()
                x = x.cpu()
                x_adversarial = x_adversarial.cpu()

            if y_pred != y_pred_adversarial:
                Adv_misclassification += 1

                vis_adv_org(x,x_adversarial,y_pred,y_pred_adversarial)
                y_preds.append(y_pred)
                y_preds_adversarial.append(y_pred_adversarial)
                noises.append((x_adversarial.data - x.data).cpu().numpy())
                xs_clean.append(x.data.cpu().numpy())
                y_trues_clean.append(y_true.data.cpu().numpy())
                adversarial_confidences.append(adversarial_confidence)
                # 3. Log adversarial images (image summary)
                info = {'Adv_image': x_adversarial.view(-1, 28, 28).numpy()}

                for tag, image in info.items():
                    self._logger.image_summary(tag, image, Adv_misclassification)

        print("Total misclassifications: ", totalMisclassification, " out of :", len(test_loader.dataset))
        print('\nTotal misclassified adversarial examples : {} out of {}\nSuccess_Rate is {:.3f}%  espsilon : {}'.format(
            Adv_misclassification, correct,
            100. * Adv_misclassification / correct ,self.epsilon))
        print("Avg_conf : {:.3f}% ; Max_conf : {:.3f}%".format(np.mean(adversarial_confidences) * 100
                                                               , np.max(adversarial_confidences) * 100))
        adv_dta_dict = {
            "xs": xs_clean,
            "y_trues": y_trues_clean,
            "y_preds": y_preds,
            "noises": noises,
            "y_preds_adversarial": y_preds_adversarial,
            "epsilon": self.epsilon,
            "Success_Rate": 100. * Adv_misclassification / correct,
            "model_acc": self.best_acc,
            "Confidences" : adversarial_confidences
        }
        # with open("utils/adv_examples/FGSM_"+str(self.epsilon) + "_" + self.args.config_file.split('/')[1] + ".pkl", "wb") as f:
        #     pickle.dump(adv_dta_dict, f)

        return adv_dta_dict

    def attack_eps(self):
        test_loader = generate_samples(self.model)
        # Load Generated samples
        # with open("data/" + self.model + "/10k_samples.pkl", "rb") as f:
        #     samples_10k = pickle.load(f)
        print("=> Samples loaded. Starting FGSM attack on epsilon....")
        # xs = samples_10k["images"]
        # y_trues = samples_10k["labels"]
        noises = []
        y_preds = []
        y_preds_adversarial = []
        xs_clean = []
        y_trues_clean = []
        totalMisclassification = 0
        Adv_misclassification = 0
        epsilons = []
        correct = 0
        for batch_idx , (x, y_true) in enumerate(test_loader):
            epsilon = 0.001
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
            correct+=1
            #generate an adversarial example
            x_adversarial = fgsm(self,x,y_true, epsilon)

            # Classify after Adv_attack
            y_pred_adversarial = np.argmax(self(Variable(x_adversarial)).cpu().data.numpy()) if self.args.cuda else np.argmax(
                self(Variable(x_adversarial)).data.numpy())

            while y_pred == y_pred_adversarial and epsilon < 0.26:
                epsilon += 0.01
                # generate an adversarial example
                x_adversarial = fgsm(self, x, y_true, epsilon)

                # Classify after Adv_attack
                y_pred_adversarial = np.argmax(
                    self(Variable(x_adversarial)).cpu().data.numpy()) if self.args.cuda else np.argmax(
                    self(Variable(x_adversarial)).data.numpy())


            if self.args.cuda:
                y_true = y_true.cpu()
                x = x.cpu()
                x_adversarial = x_adversarial.cpu()

            if y_pred != y_pred_adversarial:
                Adv_misclassification += 1
                epsilons.append(epsilon)
                y_preds.append(y_pred)
                y_preds_adversarial.append(y_pred_adversarial)
                noises.append((x_adversarial.data - x.data).cpu().numpy())
                xs_clean.append(x.data.cpu().numpy())
                y_trues_clean.append(y_true.data.cpu().numpy())

        print("Total misclassifications: ", totalMisclassification, " out of :", len(test_loader.dataset))
        print('\nTotal misclassified adversarial examples : {} out of {}'
              '\nSuccess_Rate is {:.3f}%  Average_espsilon : {:.4f}  Max_epsilon : {:.4f} Min_epsilon : {:.4f}'.format(
            Adv_misclassification, correct,
            100. * Adv_misclassification / correct, np.mean(epsilons), np.max(epsilons), np.min(epsilons)))


        adv_dta_dict = {
            "xs": xs_clean,
            "y_trues": y_trues_clean,
            "y_preds": y_preds,
            "noises": noises,
            "y_preds_adversarial": y_preds_adversarial,
            "Avg_epsilon": np.mean(epsilons),
            "Max_epsilon": np.max(epsilons),
            "Min_epsilon": np.min(epsilons),
            "Success_Rate": 100. * Adv_misclassification / correct,
            "epsilons": epsilons,
            "model_acc": self.best_acc
        }
        # with open("utils/adv_examples/FGSM_"+str(self.epsilon) + "_" + self.args.config_file.split('/')[1] + ".pkl", "wb") as f:
        #     pickle.dump(adv_dta_dict, f)

        return adv_dta_dict
