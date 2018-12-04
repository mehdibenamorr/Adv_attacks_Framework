import pickle
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from tqdm import *
from torch.autograd import Variable
from utils.common import flat_trans, generate_random_dag, layer_indexing
from igraph import *
from tensorboardX import SummaryWriter
from paddll.graphs import *
import itertools
import sklearn.metrics as metrics
from igraph import *
from utils.logger import Logger


class Net(nn.Module):
    def __init__(self,args,kwargs=None,logger=None):
        super(Net, self).__init__()
        self.args=args
        self.kwargs=kwargs
        self._logger = logger
        self.model = args.model
        # self.writer = SummaryWriter(comment=args.model + '_training_epochs_' + str(args.epochs) + '_lr_' + str(args.lr))
        self.SoftmaxWithXent = nn.CrossEntropyLoss().cuda() if args.cuda else nn.CrossEntropyLoss()
        self.best_acc = 0
        self.best_state = {}

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)




    def trainn(self,epoch,):
        #TODO early stopping for training
        self.train()
        y_trues = []
        y_preds = []
        train_loss = 0
        correct = 0
        total =0
        for batch_idx, (data, target) in enumerate(self.train_loader):
            y_trues += target.tolist()
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self(data)
            loss = self.SoftmaxWithXent(output, target)
            loss.backward()
            self.optimizer.step()


            train_loss += loss.data.item()
            pred = output.data.max(1, keepdim=True)[1]
            y_preds += pred.reshape(pred.size(0)).tolist()
            total += target.size(0)
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()


            if batch_idx % self.args.log_interval == 0:
                print('Epoch: {} [{}/{}\tLoss: {:.4f} | Acc: {:.3f}%]'.format(
                    epoch, batch_idx, len(self.train_loader), train_loss/(batch_idx+1), 100.*correct.data.item()/total))

        train_loss /= len(self.train_loader.dataset)

        print('f1_scores : Mirco: {:.5f}, Macro: {:.5f}, Weighted: {:.5f} \n'.format(
            metrics.f1_score(y_trues, y_preds, average='micro'),
            metrics.f1_score(y_trues, y_preds, average='macro'),
            metrics.f1_score(y_trues, y_preds, average='weighted')))
        # logging with Tensorboard
        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'train_loss': train_loss, 'train_accuracy': 100.*correct.data.item()/total,
                'train_f1_score_micro':metrics.f1_score(y_trues, y_preds, average='micro'),
                'train_f1_score_macro':metrics.f1_score(y_trues,y_preds,average='macro')}

        for tag, value in info.items():
            self._logger.scalar_summary(tag, value, epoch + 1)

        # 2. Log values and gradients of the parameters (histogram summary)
        for tag, value in self.named_parameters():
            tag = tag.replace('.', '/')
            # import ipdb
            # ipdb.set_trace()
            self._logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
            try:
                self._logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
            except AttributeError:
                print('No gradient data for this parameter')
                import ipdb
                ipdb.set_trace()




    def test(self,epoch):
        self.eval()
        SoftmaxWithXent = nn.CrossEntropyLoss(size_average=False)
        test_loss = 0
        correct = 0
        y_trues = []
        y_preds = []
        for data, target in self.test_loader:
            y_trues += target.tolist()
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = self.forward(data)
            test_loss += SoftmaxWithXent(output, target).data.item() # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            y_preds += pred.reshape(pred.size(0)).tolist()
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

        test_loss /= len(self.test_loader.dataset)
        # logging
        # self.writer.add_scalar('test_loss', test_loss, epoch)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {:.3f}% ({}/{}) \n'.format(
            test_loss, 100. * correct.data.item() / len(self.test_loader.dataset) ,correct, len(self.test_loader.dataset) ))
        print('f1_scores : Mirco: {:.5f}, Macro: {:.5f}, Weighted: {:.5f} \n'.format(
            metrics.f1_score(y_trues,y_preds,average='micro'),metrics.f1_score(y_trues,y_preds,average='macro'),
            metrics.f1_score(y_trues,y_preds,average='weighted')))

        # logging with Tensorboard
        # ================================================================== #
        #                        Tensorboard Logging                         #
        # ================================================================== #

        # 1. Log scalar values (scalar summary)
        info = {'test_loss': test_loss, 'test_accuracy': 100. * correct.data.item() / len(self.test_loader.dataset),
                'test_f1_score_micro': metrics.f1_score(y_trues, y_preds, average='micro'),
                'test_f1_score_macro': metrics.f1_score(y_trues, y_preds, average='macro')}

        for tag, value in info.items():
            self._logger.scalar_summary(tag, value, epoch + 1)



        #save checkpoint
        acc = 100. * correct.data.item() / len(self.test_loader.dataset)
        # import ipdb
        # ipdb.set_trace()
        if acc > self.best_acc:
            self.best_acc = acc
            self.best_state = {'model': self}
            if self.args.save:
                print('Saving..')
                state = {
                    'net': self,
                    'acc': acc,
                    'epoch': epoch,
                }
                # if not os.path.isdir('checkpoint'):
                #     os.mkdir('checkpoint')
                torch.save(state, self.args.config_file+'.ckpt')


    def del_logger(self):
        del self._logger

    def save(self):
        # TODO Get rid of this method
        # print ("Dumping weights to disk")
        # weights_dict = {}
        # for param in list(self.named_parameters()):
        #     print ("Serializing Param" , param[0])
        #     weights_dict[param[0]]= param[1]
        # with open("models/trained/"+self.model+"_weights.pkl", "wb") as f:
        #     pickle.dump(weights_dict, f)
        # print ("Finished dumping to disk...")
        state = {
            'net': self,
        }
        torch.save(state, self.args.config_file + '.ckpt')


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
        self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr, momentum=self.args.momentum,
                                   weight_decay=self.args.weight_decay)
        # self.optimizer = optim.Adam(self.parameters())
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


class CNN(Net):

    def __init__(self,args,kwargs,logger=None):
        super(CNN,self).__init__(args,kwargs,logger)
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
            batch_size=self.args.test_batch_size, shuffle=False, **self.kwargs)

    def forward(self,x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Layer(nn.Module):
    def __init__(self, in_dims, out_dim, vertices, predecessors, cuda, init_method, bias=True,  **kwargs):
        """

        :param in_dims:
        :param out_dim:
        :param vertices:
        :param predecessors:
        :param cuda:
        :param init_method:
        :param bias:
        :param kwargs:
        """
        super(Layer,self).__init__()
        self.in_dims = in_dims
        self.out_dim = out_dim
        predecessors = predecessors
        vertices = vertices
        self.cuda = cuda
        weights = []
        self.act_masks = []
        self.w_masks = []
        for i,pred in enumerate(predecessors):
            mask = Variable(torch.zeros((out_dim, in_dims[i])))
            if cuda:
                mask = mask.cuda()
            act_mask = Variable(torch.zeros(in_dims[i]))
            for j,v in enumerate(vertices):
                for p in v.predecessors():
                    if p in pred:
                        ind = pred.index(p)
                        mask[j, ind] = 1
                        act_mask[ind] = 1
            self.act_masks.append(act_mask)
            self.w_masks.append(mask)
            # import ipdb
            # ipdb.set_trace()
            # weights.append(nn.Parameter(torch.normal(mean=torch.zeros(out_dim,in_dims[i]),
            #                                          std=torch.ones(out_dim,in_dims[i])*0.1)))
            weights.append(nn.Parameter(torch.Tensor(out_dim,in_dims[i])))

        self.weights = nn.ParameterList(weights)
        self.bias_mask = torch.ones(out_dim).cuda() if cuda else torch.ones(out_dim)
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_dim))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters(init_method, **kwargs)

    def reset_parameters(self, init_method, **kwargs):
        for weight in self.weights:
            init_method(weight.data, **kwargs)
        if self.bias is not None:
            init.normal_(self.bias, 0.,0.1)

    def forward(self, inputs):
        
        output = []
        for i, inp in enumerate(inputs):
            output.append(inp.matmul(self.weights[i].mul(self.w_masks[i]).t()))
        return sum(output) + self.bias*self.bias_mask if self.bias is not None else sum(output)


class SNN(Net):
    def __init__(self,args,args1, Graph=None, logger=None, nodes=None, k=None, p=None, init_method=init.normal_, **kwargs):
        """

        :param args:
        :param args1:
        :param Graph:
        :param logger:
        :param nodes:
        :param k:
        :param p:
        :param init_method:
        :param kwargs:
        """
        super(SNN,self).__init__(args,args1, logger)
        if Graph is not None:
            graph = Graph
            self.args.nodes = nodes
            self.args.k = k
            self.args.p = p
        elif (nodes is not None) and (k is not None) and (p is not None):
            graph = generate_random_dag(nodes, k, p, self.args.layers)
            self.args.nodes = nodes
            self.args.k = k
            self.args.p = p
        else:
            graph = generate_random_dag(self.args.nodes, self.args.k, self.args.p, self.args.layers)
        self._structure_graph = graph
        self._structural_properties = {}
        vertex_by_layers = layer_indexing(self._structure_graph)
        # Using matrix multiplications
        self.input_layer = nn.Linear(784, len(vertex_by_layers[0]))
        self.output_layer = nn.Linear(len(vertex_by_layers[-1]), 10)
        # self.output_layer2 = nn.Linear(len(vertex_by_layers[-2]), 10)
        layers = []
        for i in range(1, len(vertex_by_layers)):
            layers.append(Layer([len(layer) for layer in vertex_by_layers[:i]],
                                len(vertex_by_layers[i]), vertex_by_layers[i], vertex_by_layers[:i],
                                self.args.cuda, init_method, **kwargs))
        self.layers = nn.ModuleList(layers)


        # Pruning parameters
        self.weight_masks = []
        self.bias_masks = []
        self.pruned_book = {}
        self.stats = {'num_pruned': [], 'new_pruned': [], 'f1_score': [], 'Robustness': []}


    def count_parameters(self):
        #TODO include masks in counting/ Done
        num_params = 0
        num_params += sum(p.numel() for p in self.input_layer.parameters() if p.requires_grad)
        num_params += sum(p.numel() for p in self.output_layer.parameters() if p.requires_grad)
        for module in self.layers:
            for i in range(len(module.weights)):
                num_params += torch.nonzero(module.weights[i].mul(module.w_masks[i])).size(0)
            num_params += torch.nonzero(module.bias.mul(module.bias_mask)).size(0)

        return num_params

    def count_layers(self):
        num_layers=0
        for module in self.children():
            if isinstance(module, nn.ModuleList):
                num_layers += len(module) +1
        return num_layers

    def Dataloader(self):
        # self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr)
        self.optimizer = optim.Adam(self.parameters()) #lr = 0.001, eps = 1e-8, weight_decay = L2 penalty (0)
        mnist_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,)),transforms.Lambda(flat_trans)]
        )
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/SNN', train=True, download=True,
                           transform=mnist_transform),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/SNN', train=False, transform=mnist_transform),
            batch_size=self.args.test_batch_size, shuffle=False, **self.kwargs)

    def forward(self, x):
        activations = []
        x = F.relu(self.input_layer(x))
        activations.append(x)
        for layer in self.layers:
            activations.append(F.relu(layer(activations)))
        x = self.output_layer(activations[-1]) #+ self.output_layer2(activations[-2])

        return x

    def structure_graph(self):
        return self._structure_graph

    def structural_properties(self):
        self._structural_properties['#params'] = self.count_parameters()
        self._structural_properties['#nodes'] = self._structure_graph.vcount() # excluding input and output nodes (784,10)
        self._structural_properties['#edges'] = self._structure_graph.ecount() # excluding connections from input and output layers
        self._structural_properties['avg_path_length'] = self._structure_graph.average_path_length() # average geodesic length
        self._structural_properties['diameter'] = self._structure_graph.diameter() #longest geodesic
        self._structural_properties['eccentricity_distribution'] = self._structure_graph.eccentricity()
        self._structural_properties['avg_eccentricity'] = mean(self._structure_graph.eccentricity())
        self._structural_properties['avg_betweenness'] = mean(self._structure_graph.betweenness())
        self._structural_properties['avg_closeness'] = mean(self._structure_graph.closeness())
        self._structural_properties['closeness_distribution'] = self._structure_graph.closeness()
        self._structural_properties['radius'] = self._structure_graph.radius()
        self._structural_properties['avg_edge_betweenness'] = mean(self._structure_graph.edge_betweenness())
        self._structural_properties['degree_distribution'] = self._structure_graph.degree() #degree distribution
        self._structural_properties['density'] = self._structure_graph.density() #density of the graph


    def get_structural_properties(self):
        return self._structural_properties

    def prune(self, alpha = 0.25):
        """

        :param alpha: sensitivity or quality of the weight pruning (Threshold : alpha * std(weights))
        :return:
        """
        self.index = 0
        self.num_pruned = 0
        self.num_weights = 0
        self.alpha = alpha
        vertex_by_layers = layer_indexing(self._structure_graph)
        for l, module in enumerate(self.layers):
            weight = module.weights[-1].mul(module.w_masks[-1])
            weight_num = torch.numel(weight.data)
            weight_mask = torch.ge(weight.data.abs(), alpha * weight.data.std()).type('torch.FloatTensor')
            if len(self.bias_masks) <= self.index:
                bias_mask = torch.ones(module.bias.data.size())
            else:
                bias_mask = self.bias_masks[self.index]
            if self.args.cuda:
                weight_mask = weight_mask.cuda()
                bias_mask = bias_mask.cuda()

            if len(self.weight_masks) <= self.index:
                self.weight_masks.append(weight_mask)
            else:
                self.weight_masks[self.index] = weight_mask

            for i in range(bias_mask.size(0)):
                if len(torch.nonzero(weight_mask[i]).size()) == 0:
                    bias_mask[i] = 0
            if len(self.bias_masks) <= self.index:
                self.bias_masks.append(bias_mask)
            else:
                self.bias_masks[self.index] = bias_mask

            self.index += 1

            # TODO transfer these mask to the graph structure outside of the prune function
            module.w_masks[-1] = weight_mask
            module.bias_mask = bias_mask
            deleted_connections = (weight_mask == 0).nonzero().data.cpu().numpy()
            for e in deleted_connections:
                try:
                    self._structure_graph.delete_edges(self._structure_graph.get_eid(vertex_by_layers[l][e[1]].index,
                                                                                 vertex_by_layers[l+1][e[0]].index))
                except igraph._igraph.InternalError:
                    pass

            for n in (bias_mask==0).nonzero().data.cpu().numpy():
                if vertex_by_layers[l+1][n[0]].outdegree() == 0:
                    try:
                        self._structure_graph.delete_vertices(vertex_by_layers[l+1][n[0]].index)
                    except igraph._igraph.InternalError:
                        pass

            layer_pruned = weight_num - torch.nonzero(weight_mask).size(0)
            print("{} pruned weights of layer {}".format(100*(layer_pruned/weight_num), self.index))
            bias_num = torch.nonzero(module.bias.data).size(0)
            bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
            print("{} pruned biases of layer {}".format(100*(bias_pruned/bias_num), self.index))

            if self.index not in self.pruned_book.keys():
                self.pruned_book[self.index] = [100*layer_pruned/weight_num]
            else:
                self.pruned_book[self.index].append(100*layer_pruned/weight_num)

            self.num_pruned += layer_pruned
            self.num_weights += weight_num

            module.weights[-1].data *= weight_mask
            module.bias.data *= bias_mask

    def prune_random(self, n = 0.2):
        """

        :param n: % percentage of weights to prune randomly from each layer
        :return:
        """
        self.index = 0
        self.num_pruned = 0
        self.num_weights = 0
        self.n = n
        vertex_by_layers = layer_indexing(self._structure_graph)
        for l, module in enumerate(self.layers):
            weight = module.weights[-1].mul(module.w_masks[-1])
            weight_num = torch.numel(weight.data)
            # weight_mask = torch.ge(weight.data.abs(), alpha * weight.data.std()).type('torch.FloatTensor')
            if len(self.weight_masks) <= self.index:
                weight_mask = torch.ones(weight.data.size())
                bias_mask = torch.ones(module.bias.data.size())
            else:
                weight_mask = self.weight_masks[self.index]
                bias_mask = self.bias_masks[self.index]
            # numpy random choice method to randomnly prune n weigths
            idx = weight_mask.nonzero()

            n_w = len(idx) - int(len(idx)*n)
            weight_mask[np.random.choice(idx[:,0], len(idx)- n_w, replace=False),
                        np.random.choice(idx[:,1], len(idx)- n_w, replace=False)] = 0

            if self.args.cuda:
                weight_mask = weight_mask.cuda()
                bias_mask = bias_mask.cuda()

            if len(self.weight_masks) <= self.index:
                self.weight_masks.append(weight_mask)
            else:
                self.weight_masks[self.index] = weight_mask

            for i in range(bias_mask.size(0)):
                if len(torch.nonzero(weight_mask[i]).size()) == 0:
                    bias_mask[i] = 0
            if len(self.bias_masks) <= self.index:
                self.bias_masks.append(bias_mask)
            else:
                self.bias_masks[self.index] = bias_mask

            self.index += 1

            # TODO transfer these mask to the graph structure outside of the prune function
            module.w_masks[-1] = weight_mask
            module.bias_mask = bias_mask
            deleted_connections = (weight_mask == 0).nonzero().data.cpu().numpy()
            for e in deleted_connections:
                try:
                    self._structure_graph.delete_edges(self._structure_graph.get_eid(vertex_by_layers[l][e[1]].index,
                                                                                 vertex_by_layers[l+1][e[0]].index))
                except igraph._igraph.InternalError:
                    pass

            for n in (bias_mask==0).nonzero().data.cpu().numpy():
                if vertex_by_layers[l+1][n[0]].outdegree() == 0:
                    try:
                        self._structure_graph.delete_vertices(vertex_by_layers[l+1][n[0]].index)
                    except igraph._igraph.InternalError:
                        pass

            layer_pruned = weight_num - torch.nonzero(weight_mask).size(0)
            print("{}% pruned weights of layer {}".format(100*(layer_pruned/weight_num), self.index))
            bias_num = torch.numel(module.bias.data)
            bias_pruned = bias_num - torch.nonzero(bias_mask).size(0)
            print("{}% pruned biases of layer {}".format(100*(bias_pruned/bias_num), self.index))

            if self.index not in self.pruned_book.keys():
                self.pruned_book[self.index] = [100*layer_pruned/weight_num]
            else:
                self.pruned_book[self.index].append(100*layer_pruned/weight_num)

            self.num_pruned += layer_pruned
            self.num_weights += weight_num

            module.weights[-1].data *= weight_mask
            module.bias.data *= bias_mask

    def set_grad(self):
        for module in self.layers:
            module.weights[-1].grad.data *= self.weight_masks[self.index]
            module.bias.grad.data *= self.bias_masks[self.index]
            self.index += 1

    def train_pruned(self):
        y_trues = []
        y_preds = []
        for data, target in self.train_loader:
            self.index = 0
            y_trues += target.tolist()
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.forward(data)
            loss = self.SoftmaxWithXent(output, target)
            loss.backward()
            self.set_grad() # mask the gradient data of the weights

            self.optimizer.step()
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            y_preds += pred.reshape(pred.size(0)).tolist()

        # print('f1_scores : Mirco: {:.5f}, Macro: {:.5f}, Weighted: {:.5f} \n Precision_score : {:.5f} %'.format(
        #     metrics.f1_score(y_trues, y_preds, average='micro'), metrics.f1_score(y_trues, y_preds, average='macro'),
        #     metrics.f1_score(y_trues, y_preds, average='weighted'),
        #     100*metrics.precision_score(y_trues, y_preds, average='micro')))



    def validate(self):
        y_trues = []
        y_preds = []
        for data, target in self.test_loader:
            y_trues += target.tolist()
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            with torch.no_grad():
                data, target = Variable(data), Variable(target)
            output = self.forward(data)
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            y_preds += pred.reshape(pred.size(0)).tolist()

        # print('f1_scores : Mirco: {:.5f}, Macro: {:.5f}, Weighted: {:.5f} \n Precision_score : {:.5f} %'.format(
        #     metrics.f1_score(y_trues, y_preds, average='micro'), metrics.f1_score(y_trues, y_preds, average='macro'),
        #     metrics.f1_score(y_trues, y_preds, average='weighted'), metrics.precision_score(y_trues,y_preds, average='micro')))

        return metrics.f1_score(y_trues,y_preds, average='micro')




    def save(self):
        # del self._structure_graph
        torch.save(self.state_dict(), "tests/"+self.model+"_"+self.args.config_file.split('/')[1]+".pt")
        # print ("Dumping weights to disk")
        # weights_dict = {}
        # for param in list(self.named_parameters()):
        #     print ("Serializing Param" , param[0])
        #     weights_dict[param[0]]= param[1]
        # with open("models/trained/"+self.model+"_weights.pkl", "wb") as f:
        #     pickle.dump(weights_dict, f)
        # print ("Finished dumping to disk...")


class _SparseTorch(nn.Module):
    def __init__(self, input_size, output_size, structure_graph,cuda):
        """
        :param input_size:
        :type input_size int
        :param output_size:
        :type output_size int
        :param structure_graph: A graph object specifying the structure of your arbitrary designed graph.
        :type structure_graph igraph.Graph
        """
        super(_SparseTorch, self).__init__()
        self._input_size = input_size
        self._output_size = output_size
        self._structure_graph = structure_graph

        layer_index, vertices_by_layer = build_layer_index(self._structure_graph)
        self._layer_index = layer_index
        self._vertices_by_layer = vertices_by_layer

        self._fully_input_to_sources = nn.Linear(self._input_size, len(vertices_by_layer[0]))

        # Contains variables for each layer in size of number of its vertices
        self._layers = {}
        # Contains variables for each vertex in size of the number of its incoming connections
        self._weights_per_vertice = {}
        vertices_with_highway_to_output = []
        for layer in vertices_by_layer:
            if layer is 0:
                pass
            else:
                #self._layers[layer] = nn.Parameter(torch.zeros(len(vertices_by_layer[layer])))
                self._layers[layer] = Variable(torch.normal(
                    mean=torch.zeros(len(vertices_by_layer[layer])),
                    std=torch.ones(len(vertices_by_layer[layer])) * 0.1)).cuda() if cuda else Variable(torch.normal(
                    mean=torch.zeros(len(vertices_by_layer[layer])),
                    std=torch.ones(len(vertices_by_layer[layer])) * 0.1))
                #self._layers[layer] = Variable(torch.zeros(len(vertices_by_layer[layer])))
                for vertice in vertices_by_layer[layer]:
                    incoming = self._structure_graph.es.select(_target=vertice)
                    ordered_sources = sorted(edge.source for edge in incoming)
                    incoming_size = len(ordered_sources)
                    self._weights_per_vertice[vertice] = nn.Parameter(torch.normal(
                        mean=torch.zeros(incoming_size),
                        std=torch.ones(incoming_size) * 0.1
                    ))
                    '''self._weights_per_vertice[vertice] = Variable(torch.normal(
                        mean=torch.zeros(incoming_size),
                        std=torch.ones(incoming_size) * 0.1))'''

                    # Add to list of vertices with no outgoing edges
                    successors = self._structure_graph.vs[vertice].successors()
                    if len(successors) is 0:
                        # We have no outgoing connections, so this vertice should be connected to last FF layer
                        vertices_with_highway_to_output.append(vertice)

        last_hidden_layer_index = max(layer_index.values())
        vertices_to_fully_layer = [vertice for vertice in vertices_by_layer[last_hidden_layer_index]]
        vertices_to_fully_layer.extend(vertices_with_highway_to_output)
        self._vertices_to_fully_layer = sorted(vertices_to_fully_layer)
        last_layer_to_output_dim = [self._output_size, len(vertices_to_fully_layer)]
        self._weights_to_output_layer = nn.Parameter(torch.normal(
            mean=torch.zeros(last_layer_to_output_dim),
            std=torch.ones(last_layer_to_output_dim) * 0.1
        ))
        #self._weights_to_output_layer = Variable(torch.normal(mean=torch.zeros(last_layer_to_output_dim), std=torch.ones(last_layer_to_output_dim) * 0.1))

        self._param_list = nn.ParameterList([param for param in itertools.chain(
            [self._weights_to_output_layer],
            self._weights_per_vertice.values()
        )])
        self._linear_dummy = nn.Linear(len(self._vertices_to_fully_layer), self._output_size)

    def forward(self, x):
        activation = F.relu
        #print('X:')
        #print(x.size())
        #print()
        layer_index, vertices_by_layer = self._layer_index, self._vertices_by_layer

        self._layers[0] = activation(self._fully_input_to_sources(x)).transpose(0, 1)
        vertex_to_index_by_layer = {
            0: {vertice: idx for vertice, idx in zip(vertices_by_layer[0], range(len(vertices_by_layer[0])))}
        }

        for layer in vertices_by_layer:
            if layer is 0:
                pass
            else:
                vertices_in_layer = len(vertices_by_layer[layer])
                vertex_to_index_by_layer[layer] = {vertice: idx for vertice, idx in zip(vertices_by_layer[layer], range(vertices_in_layer))}

                layer_results = []
                for vertice, current_vertex_idx in zip(vertices_by_layer[layer], range(vertices_in_layer)):
                    incoming = self._structure_graph.es.select(_target=vertice)
                    collected_inputs = []
                    ordered_sources = sorted(edge.source for edge in incoming)
                    incoming_size = len(ordered_sources)
                    for source in ordered_sources:
                        source_layer = layer_index[source]
                        source_idx = vertex_to_index_by_layer[source_layer][source]
                        #print('source layer %s, source vertex %s [idx#%s]:\n\n %s' % (source_layer, source, source_idx, self._layers[source_layer]))
                        #collected_inputs.append(self._layers[source_layer].index_select(0, Variable(torch.LongTensor([idx]))))
                        #print(self._layers[source_layer].index_select(0, Variable(torch.LongTensor([idx]))))
                        #print(self._layers[source_layer][idx])
                        #print()
                        propagation_value = self._layers[source_layer][source_idx].unsqueeze(0)
                        collected_inputs.append(propagation_value)
                    #print([x.size() for x in collected_inputs])
                    vertex_input = torch.cat(collected_inputs)
                    #vertex_input = torch.stack(collected_inputs, dim=1)
                    #print('Vertex input dim: %s' % vertex_input.size())

                    # activation(sum(W * x))
                    """print()
                    print('Vertice %s [#%s] in layer %s' % (vertice, current_vertex_idx, layer))
                    print('-'*10)
                    print('ordered_sources %s' % ordered_sources)
                    print('vertex_input:')
                    print(vertex_input.size())
                    print('weights:')
                    print(self._weights_per_vertice[vertice].size())"""
                    wx = self._weights_per_vertice[vertice].view([1, incoming_size]).mm(vertex_input)
                    vertex_result = activation(torch.sum(wx, dim=0))
                    #print('vertex_result:')
                    #print(vertex_result.size())
                    layer_results.append(vertex_result)

                #print()
                self._layers[layer] = torch.stack(layer_results, 0)
                #self._layers[layer] = torch.cat(layer_results)
                #print('Layer %s resulted in: ' % layer)
                #print(self._layers[layer].size())
                #print()

        collected_input = []
        for vertice in self._vertices_to_fully_layer:
            source_layer = layer_index[vertice]
            if source_layer not in vertex_to_index_by_layer:
                raise RuntimeError('Source layer was not found in indices for previous layers - structural error.')
            idx = vertex_to_index_by_layer[source_layer][vertice]
            #collected_input.append(self._layers[source_layer].index_select(0, Variable(torch.LongTensor([idx]))))
            propagation_value = self._layers[source_layer][idx].unsqueeze(0)
            collected_input.append(propagation_value)
        input_to_fully_layer = torch.cat(collected_input)

        result = input_to_fully_layer.transpose(0, 1)
        return self._linear_dummy(result)


class JSNN(Net):
    def __init__(self,args,kwargs=None):
        super(JSNN, self).__init__(args,kwargs)
        self._torch_model = _SparseTorch(784,10,generate_random_dag(self.args.nodes, self.args.k, self.args.p),self.args.cuda)

    def Dataloader(self):
        # self.optimizer = optim.SGD(self.parameters(), lr=self.args.lr)
        self.optimizer = optim.Adam(self._torch_model.parameters())
        mnist_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Lambda(flat_trans)]
        )
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/SNN', train=True, download=True,
                           transform=mnist_transform),
            batch_size=self.args.batch_size, shuffle=True, **self.kwargs)
        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/SNN', train=False, transform=mnist_transform),
            batch_size=self.args.test_batch_size, shuffle=False, **self.kwargs)

    def cuda(self, device=None):
        self._torch_model.cuda()

    def forward(self, x):
        x = self._torch_model(x)

        return x

models={'FFN' : FFN, 'CNN' : CNN, 'SNN' : SNN, 'JSNN': JSNN}