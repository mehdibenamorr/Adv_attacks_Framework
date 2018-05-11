import argparse
import torch
from models.models import models

# Training settings
parser = argparse.ArgumentParser(description='Train Mnist models')
parser.add_argument('--model', type=str, default="FFN",
                    help='model to train (default: FFN)')
parser.add_argument('--batch-size', type=int, default=256,
                    help='input batch size for training (default: 256)')
parser.add_argument('--test-batch-size', type=int, default=256,
                    help='input batch size for testing (default: 256)')
parser.add_argument('--epochs', type=int, default=100,
                    help='number of epochs to train (default: 100)')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=50,
                    help='how many batches to wait before logging training status')
parser.add_argument('--weight_decay', type=float, default=1e-04,
                    help='weigth_decay rate')
parser.add_argument('--adv_train', action='store_true', default=False,
                    help='Train the model on adversarial examples ')
parser.add_argument('--method', type=str, default="FGSM",
                    help='attack method to train with (default: FGSM)')


args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()


torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

kwargs = {'num_workers' : 1 , 'pin_memory': True} if args.cuda else {}

#write the trainng and generation




if __name__=="__main__":
    model = models[args.model](args,kwargs)
    model.Dataloader()
    if args.cuda:
        model.cuda()

    for epoch in range(1,args.epochs+1):
        model.trainn(epoch)
        if args.adv_train:
            model.Adv_train(epoch)
        model.test(epoch)
    # model.save()