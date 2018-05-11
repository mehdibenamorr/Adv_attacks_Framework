import torch
from torchvision import datasets, transforms
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np


def flat_trans(x):
    x.resize_(28*28)
    return x

def generate_samples(model):
    if  os.path.isfile("data/"+model+"/5k_samples.pkl"):
        pass
    else:
        mnist_transform = transforms.Lambda(flat_trans) if model == "FFN" else transforms.Normalize((0.1307,),
                                                                                                    (0.3081,))
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/'+model, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                mnist_transform
            ])),
            batch_size=100, shuffle=True)
        images, labels = [], []
        for idx, data in enumerate(test_loader):
            x_lots, y_lots = data
            for x, y in zip(x_lots, y_lots):
                images.append(x.numpy())
                labels.append(y)

            if idx == 49:
                break
        with open("data/"+model+"/5k_samples.pkl", "wb") as f:
            data_dict = {"images": images, "labels": labels}
            pickle.dump(data_dict, f)


def vis_adv_org(x_org, x_adv,y_pred,y_adv,target=None):
    noise = (x_adv - x_org).data.numpy().reshape(28,28)
    x_org = x_org.data.numpy().reshape(28,28)
    x_adv = x_adv.data.numpy().reshape(28,28)
    disp_im = np.concatenate((x_org, x_adv,noise),axis=1)
    if target:
        plt.title("Original : {}    Target/Adv : {}/{}    Perturbation : {:.5f}".format(y_pred,target,y_adv,noise.mean()))
    else:
        plt.title("Original : {}    Adversarial : {}    Perturbation : {:.5f}".format(y_pred,y_adv,noise.mean()))
    plt.imshow(disp_im,cmap='gray')
    plt.show()

#Adversarial attacks methods :
def fgsm(Net,x,y_true,epsilon=0.1):
    # Generate Adv Image
    outputs = Net(x)
    loss = Net.SoftmaxWithXent(outputs, y_true)
    loss.backward()  # to obtain gradients of x
    # Add small perturbation
    x_grad = torch.sign(x.grad.data)
    x_adversarial = torch.clamp(x.data + epsilon * x_grad, 0, 1)

    return x_adversarial

def l_bfgs(self,_x,_l_target,norm,max_iter):

    # Optimitzation box contrained
    for i in range(max_iter):
        self.Optimizer.zero_grad()
        output = self(_x)
        loss = self.SoftmaxWithXent(output, _l_target)

        # Norm used
        if norm == "l1":
            adv_loss = loss + torch.mean(torch.abs(self.r))
        elif norm == "l2":
            adv_loss = loss + torch.mean(torch.pow(self.r, 2))
        else:
            adv_loss = loss

        adv_loss.backward()
        self.Optimizer.step()

        # Until output == y_target
        y_pred_adversarial = np.argmax(self(_x).cpu().data.numpy()) if self.args.cuda else np.argmax(
            self(_x).data.numpy())

        if y_pred_adversarial == _l_target.data.item():
            break

        if i == max_iter - 1:
            print("Results may be incorrect, Optimization run for {} iteration".format(max_iter))
    x_adversarial = _x + self.r

    return x_adversarial, y_pred_adversarial

methods ={'FGSM':fgsm,'L_BFGS':l_bfgs}