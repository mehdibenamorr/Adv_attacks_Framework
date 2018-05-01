import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import *
from torch.autograd import Variable
from utils.common import flat_trans

#Define different deep learning models to attack

class Net(nn.Module):
    def __init__(self,args,kwargs=None):
        super(Net, self).__init__()
        self.args=args
        self.kwargs=kwargs
        self.model = args.model
        self.SoftmaxWithXent = nn.CrossEntropyLoss()

    def trainn(self,epoch):
        if self.model=="CNN":
            super(Net,self).train()
        for batch_idx, (data, target) in tqdm(enumerate(self.train_loader)):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.SoftmaxWithXent(output, target)
            loss.backward()
            self.optimizer.step()
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.data.item()))

    def test(self):
        self.eval()
        SoftmaxWithXent = nn.CrossEntropyLoss(size_average=False)
        test_loss = 0
        correct = 0
        for data, target in self.test_loader:
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = self.forward(data)
            test_loss =test_loss +   SoftmaxWithXent(output, target).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
    def save(self):
        print ("Dumping weights to disk")
        weights_dict = {}
        for param in list(self.named_parameters()):
            print ("Serializing Param" , param[0])
            weights_dict[param[0]]= param[1]
        with open("models/trained/"+self.model+"_weights.pkl", "wb") as f:
            pickle.dump(weights_dict, f)
        print ("Finished dumping to disk...")

class FFN(Net):
    def __init__(self,args,kwargs):
        super(FFN,self).__init__(args,kwargs)
        self.fc1 = nn.Linear(28 * 28, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 10)


    def Dataloader(self):
        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay)
        mnist_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(flat_trans)]
        )
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/FFN', train=True, download=True,
                           transform=mnist_transform),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/FFN', train=False, transform=mnist_transform),
            batch_size=self.args.test_batch_size, shuffle=True, **self.kwargs)
    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return (x)

class CNN(Net):
    def __init__(self,args,kwargs):
        super(CNN,self).__init__(args,kwargs)
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)


    def Dataloader(self):
        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum)
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/CNN', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/CNN', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=self.args.test_batch_size, shuffle=True, **self.kwargs)
    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)