import numpy as np

def flat_trans(x):
    x.resize_(28*28)
    return x