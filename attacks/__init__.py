
from .Attacker import Attack
from .FGSM import FGSM
from .L_BFGS import L_BFGS
from  .One_Pixel import One_Pixel


attacks = {'FGSM' : FGSM, 'L_BFGS' : L_BFGS, 'One_Pixel': One_Pixel}