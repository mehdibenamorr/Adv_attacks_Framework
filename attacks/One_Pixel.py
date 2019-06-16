import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
from utils.common import generate_samples,vis_adv_org
from utils.attacks import one_pixel
from .Attacker import Attack

class One_Pixel(Attack):
    def __init__(self,args,kwargs=None, Net=None, logger=None):
        super(One_Pixel,self).__init__(args,kwargs,Net=Net, logger=logger)

        self.max_iter = self.args.max_iter
        self.popsize = self.args.popsize
        self.samples = self.args.samples
        self.pixles = self.args.pixels
        self.targeted = self.args.targeted

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
        adversarial_confidences = []
        for batch_idx, (x, y_true) in enumerate(test_loader):

            num_adv = 0 # number of successful generated adversarial examples from a sample
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

                flag, x_, adv_img = one_pixel(x,y_true, self.Net,self.model,target_class, pixels=self.args.pixels,
                                    maxiter=self.args.max_iter, popsize= self.args.popsize,cuda=self.args.cuda)

                Adv_misclassification += flag
                num_adv += flag

                if flag:
                    outputs = self(adv_img)
                    y_pred_adversarial = np.argmax(outputs.data.cpu().numpy()) if self.args.cuda else np.argmax(
                        outputs.data.numpy())
                    adversarial_confidence = np.max(F.softmax(outputs, dim=1).data.cpu().numpy()) if self.args.cuda else \
                        np.max(F.softmax(outputs, dim=1).data.numpy())
                    # y_pred_adversarial = np.argmax(self(adv_img).cpu().data.numpy()) if self.args.cuda else np.argmax(
                    #     self(self(adv_img)).data.numpy())

                    y_preds.append(y_pred)
                    y_preds_adversarial.append(y_pred_adversarial)
                    noises.append((adv_img.data - img_var.data).cpu().numpy())
                    xs_clean.append(img_var.data.cpu().numpy())
                    y_trues_clean.append(y_true)
                    adversarial_confidences.append(adversarial_confidence)
                    vis_adv_org(x, adv_img, y_pred, y_pred_adversarial)
                    # 3. Log adversarial images (image summary)
                    info = {'Adv_image': adv_img.view(-1,28,28).cpu().numpy()}

                    for tag, image in info.items():
                        self._logger.image_summary(tag, image, Adv_misclassification)

            if correct == self.args.samples:
                try:
                    break
                except ConnectionResetError:
                    break


        success_rate = Adv_misclassification / correct
        print("Total misclassifications: ", totalMisclassification, " out of :", self.args.samples)
        print('\nTotal misclassified adversarial examples : {}/{} \nSuccess_Rate is {:.3f}%'.format(
            Adv_misclassification,correct,
            100. * success_rate))
        print("Avg_conf : {:.3f}% ; Max_conf : {:.3f}%".format(np.mean(adversarial_confidences)*100
                                                               ,np.max(adversarial_confidences)*100))



        adv_dta_dict = {
            "xs": xs_clean,
            "y_trues": y_trues_clean,
            "y_preds": y_preds,
            "noises": noises,
            "y_preds_adversarial": y_preds_adversarial,
            "Success_Rate": 100.*success_rate,
            "model_acc": self.best_acc,
            "Confidences": adversarial_confidences
        }

        return adv_dta_dict

