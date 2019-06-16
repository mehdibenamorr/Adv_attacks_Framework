import torch
import torch.nn as nn
from torch.autograd import Variable
import sklearn.metrics as metrics


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

