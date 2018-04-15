from utils.attack_methods import *
import argparse


# Training settings
parser = argparse.ArgumentParser(description='Attack Mnist models')
parser.add_argument('--model', type=str, default="FFN", metavar='MD',
                    help='model to attack (default: FFN)')
parser.add_argument('--method', type=str, default="fgsm", metavar='A',
                    help='method to use for the adversarial attack (default: FGSM)')
parser.add_argument('--attack', action='store_true', default=True,
                    help='method to use for the adversarial attack (default: FGSM)')
args = parser.parse_args()

if __name__=="__main__":
    attacker = Attack(args)
    attacker.load_weights("../utils/trained/"+attacker.model+"_weights.pkl")
    attacker.fgsm()