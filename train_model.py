import configargparse
import torch
from models.models import models

# Training settings
parser = configargparse.ArgParser()
parser.add('-c','--config-file', required=True, is_config_file= True,help='config file path')
parser.add('--model', type=str, default="SNN",
                    help='model to train (default: SNN)')
parser.add('--batch-size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add('--test-batch-size', type=int, default=256,
                    help='input batch size for testing (default: 256)')
parser.add('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add('--weight_decay', type=float, default=1e-04,
                    help='weigth_decay rate')
parser.add('--nodes', type=int, default=350,
                    help='number of nodes (default: 10)')
parser.add('--m', type=int, default=10,
                    help='number of edges to attach a new node to existing ones (default: 3)')
parser.add('--p', type=float, default=0.8,
                    help='probability for edge creation (default: 0.5)')
parser.add('--k', type=int, default=6,
                    help='Each node is joined with its k nearest neighbors in a ring topology (default: 1)')
parser.add('--threshold', type=float, default=95,
                    help='accuracy threshold for saving the model (default: 95%)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers' : 1 , 'pin_memory': True} if args.cuda else {}

#write the trainng and generation




if __name__=="__main__":
    model = models[args.model](args,kwargs)
    if args.cuda:
        model.cuda()
    model.Dataloader()

    for epoch in range(1,args.epochs+1):
        model.trainn(epoch)
        model.test(epoch)