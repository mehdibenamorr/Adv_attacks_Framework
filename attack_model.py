from attacks.attack_methods import *
import argparse
import torch

# Training settings
parser = argparse.ArgumentParser(description='Attack Mnist models')
parser.add_argument('--model', type=str, default="FFN",
                    help='model to attack (default: FFN)')
parser.add_argument('--method', type=str, default="FGSM",
                    help='method to use for the adversarial attack (default: FGSM)')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--norm', type=str, default="l2",
                    help='regularization norm (default: l2)')
parser.add_argument('--max_iter', type=int, default=1000,
                    help='maximum iter for l_bfgs optimization (default: 1000)')
parser.add_argument('--V', action='store_true', default=False,
                    help='visualize generated adversarial examples')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

if __name__=="__main__":
    attacker = attacks[args.method](args)
    if args.cuda:
        attacker.cuda()
    attacker.load_weights("models/trained/"+attacker.model+"_weights.pkl")
    attacker.attack()