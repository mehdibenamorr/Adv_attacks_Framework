import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import *
from torch.autograd import Variable
from utils.common import flat_trans, methods, generate_random_dag, layer_indexing
from igraph import *
from tensorboardX import SummaryWriter

#Define different deep learning models to attack


class Net(nn.Module):
    def __init__(self,args,kwargs=None):
        super(Net, self).__init__()
        self.args=args
        self.kwargs=kwargs
        self.model = args.model
        # self.writer = SummaryWriter(comment=args.model + '_training_epochs_' + str(args.epochs) + '_lr_' + str(args.lr))
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
            #logging
            # self.writer.add_scalar('loss',loss.data.item(),(epoch*len(self.train_loader)))
            if batch_idx % self.args.log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.data.item()))

    def Adv_train(self,epoch,method="FGSM"):
        for batch_idx, (data,target) in tqdm(enumerate(self.train_loader)):
            if self.args.cuda:
                data, target = data.cuda(),target.cuda()
            data, target = Variable(data,requires_grad=True), Variable(target)
            data_adv = methods[method](self, data, target, epsilon=0.1)
            self.optimizer.zero_grad()
            output = self(data_adv)
            loss = self.SoftmaxWithXent(output, target)
            loss.backward()
            self.optimizer.step()
            # logging
            # self.writer.add_scalar('Adv_loss', loss.data.item(), (epoch * len(self.train_loader)))
            if batch_idx % self.args.log_interval == 0:
                print('Adv_Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(self.train_loader.dataset),
                           100. * batch_idx / len(self.train_loader), loss.data.item()))

    def test(self,epoch):
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
        # logging
        # self.writer.add_scalar('test_loss', test_loss, epoch)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset),
            100. * correct / len(self.test_loader.dataset)))
        acc = 100. * correct / len(self.test_loader.dataset)
        if epoch == self.args.epochs and acc >= self.args.threshold:
            self.save()

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


class Layer(nn.Module):
    def __init__(self, in_dims, out_dim, vertices, predecessors, cuda, bias=True):
        super(Layer,self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        self.predecessors = predecessors
        self.vertices = vertices
        self.cuda = cuda
        weights = []
        self.act_masks = []
        self.w_masks = []
        for i,pred in enumerate(self.predecessors):
            mask = torch.zeros((out_dim, in_dims[i]))
            if cuda:
                mask = mask.cuda()
            act_mask = torch.zeros(in_dims[i])
            for j,v in enumerate(self.vertices):
                for p in v.predecessors():
                    if p in pred:
                        ind = pred.index(p)
                        mask[j, ind] = 1
                        act_mask[ind] = 1
            self.act_masks.append(act_mask)
            self.w_masks.append(mask)
            # import ipdb
            # ipdb.set_trace()
            weights.append(nn.Parameter(torch.normal(mean=torch.zeros(out_dim,in_dims[i]), std=0.1)))
        self.weights = nn.ParameterList(weights)
        if bias:
            self.bias = nn.Parameter(torch.normal(mean=torch.zeros(out_dim), std=0.1))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        output = torch.zeros(self.out_dim)
        if self.cuda:
            output = output.cuda()
        for i, inp in enumerate(inputs):
            output = output.add(inp.matmul(self.weights[i].mul(self.w_masks[i]).t()) + self.bias)
        return output


class SNN(Net):
    def __init__(self,args,kwargs):
        super(SNN,self).__init__(args,kwargs)
        self.graph = generate_random_dag(args.nodes, args.k, args.p)
        vertex_by_layers = layer_indexing(self.graph)
        l = self.graph.layout('fr')
        plot(self.graph, layout=l)
        # Using matrix multiplactions
        self.input_layer = nn.Linear(784, len(vertex_by_layers[0]))
        self.output_layer1 = nn.Linear(len(vertex_by_layers[-1]), 10)
        self.output_layer2 = nn.Linear(len(vertex_by_layers[-2]), 10)
        layers = []
        for i in range(1, len(vertex_by_layers)):
            layers.append(Layer([len(layer) for layer in vertex_by_layers[:i]],
                                len(vertex_by_layers[i]), vertex_by_layers[i], vertex_by_layers[:i], self.args.cuda))
        self.layers = nn.ModuleList(layers)

    def Dataloader(self):
        # self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum,
        #                            weight_decay=self.args.weight_decay)
        self.optimizer = optim.Adam(self.parameters())
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

    def forward(self, x):
        activations = []
        x = F.relu(self.input_layer(x))
        activations.append(x)
        for layer in self.layers:
            activations.append(F.relu(layer(activations)))
        x = (self.output_layer1(activations[-1]) + self.output_layer2(activations[-2]))

        return x

    def save(self):
        torch.save(self.state_dict(), "tests/"+self.model+"_"+self.args.config_file.split('/')[1]+".pt")
        # print ("Dumping weights to disk")
        # weights_dict = {}
        # for param in list(self.named_parameters()):
        #     print ("Serializing Param" , param[0])
        #     weights_dict[param[0]]= param[1]
        # with open("models/trained/"+self.model+"_weights.pkl", "wb") as f:
        #     pickle.dump(weights_dict, f)
        # print ("Finished dumping to disk...")

models={'FFN' : FFN, 'CNN' : CNN, 'SNN' : SNN}