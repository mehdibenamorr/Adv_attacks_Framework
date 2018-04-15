import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import os
import pickle

def flat_trans(x):
    x.resize_(28*28)
    return x

def generate_samples(model):
    if not os.path.isfile(".adv_examples/"+model+"_samples.pkl"):
        pass
    else:
        mnist_transform = transforms.Lambda(flat_trans) if model == "FFN" else transforms.Normalize((0.1307,),
                                                                                                    (0.3081,))
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data/'+model, train=False, transform=transforms.Compose([
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
        with open(".adv_examples/"+model+"_samples.pkl", "wb") as f:
            data_dict = {"images": images, "labels": labels}
            pickle.dump(data_dict, f)
