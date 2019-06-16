from .Net import Net
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torch.nn.functional as F
import torch.optim as optim
from utils.common import flat_trans


class FFN(Net):

    def __init__(self,args,kwargs,logger=None):
        super(FFN,self).__init__(args,kwargs,logger)
        if self.args.layers == 1:
            self.fc1 = nn.Linear(28*28,self.args.nodes)
            self.fc2 = nn.Linear(self.args.nodes,10)
        elif self.args.layers == 2:
            self.fc1 = nn.Linear(28 * 28, int(self.args.nodes/2))
            self.fc2 = nn.Linear(int(self.args.nodes/2), int(self.args.nodes/2))
            self.fc3 = nn.Linear(int(self.args.nodes/2), 10)
        else:
            self.fc1 = nn.Linear(28 * 28, 300)
            self.fc2 = nn.Linear(300, 100)
            self.fc3 = nn.Linear(100, 10)

    def Dataloader(self):
        # self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum,
        #                            weight_decay=self.args.weight_decay)
        self.optimizer = optim.Adam(self.parameters())
        mnist_transform = transforms.Compose(
            [transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,)), transforms.Lambda(flat_trans)]
        )
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/FFN', train=True, download=True,
                           transform=mnist_transform),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/FFN', train=False, transform=mnist_transform),
            batch_size=self.args.test_batch_size, shuffle=False, **self.kwargs)

    def forward(self,x):
        if self.args.layers == 1:
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
        return (x)