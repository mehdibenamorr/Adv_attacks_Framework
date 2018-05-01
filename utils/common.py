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

def vis_adv_org(x_org, x_adv,y_pred,y_adv):
    noise = (x_adv - x_org).data.numpy().reshape(28,28)
    x_org = x_org.data.numpy().reshape(28,28)
    x_adv = x_adv.data.numpy().reshape(28,28)
    disp_im = np.concatenate((x_org, x_adv,noise),axis=1)
    plt.title("Original : {}    Adversarial : {}    Perturbation".format(y_pred,y_adv))
    plt.imshow(disp_im)
    plt.show()
