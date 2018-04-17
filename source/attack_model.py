from utils.attack_methods import *
import argparse


# Training settings
parser = argparse.ArgumentParser(description='Attack Mnist models')
parser.add_argument('--model', type=str, default="FFN", metavar='MD',
                    help='model to attack (default: FFN)')
parser.add_argument('--method', type=str, default="FGSM", metavar='A',
                    help='method to use for the adversarial attack (default: FGSM)')
parser.add_argument('--attack', action='store_true', default=True,
                    help='boolean that im gonna get rid off')
args = parser.parse_args()

if __name__=="__main__":
    attacker = Attack(args)
    attacker.load_weights("../utils/trained/"+attacker.model+"_weights.pkl")
    if args.method == "FGSM":
        attacker.FGSM()
    elif args.method == "L_BFGS":
        attacker.L_BFGS("l2")