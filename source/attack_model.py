from utils.attack_methods import *
import argparse
import torch

# Training settings
parser = argparse.ArgumentParser(description='Attack Mnist models')
parser.add_argument('--model', type=str, default="FFN", metavar='MD',
                    help='model to attack (default: FFN)')
parser.add_argument('--method', type=str, default="FGSM", metavar='A',
                    help='method to use for the adversarial attack (default: FGSM)')
parser.add_argument('--attack', action='store_true', default=True,
                    help='boolean that im gonna get rid off')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

if __name__=="__main__":
    attacker = Attack(args)
    if args.cuda:
        attacker.cuda()
    attacker.load_weights("utils/trained/"+attacker.model+"_weights.pkl")
    if args.method == "FGSM":
        attacker.FGSM()
    elif args.method == "L_BFGS":
        attacker.L_BFGS("l2")