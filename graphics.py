import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
# sns.set(style='ticks', palette='Set2')
#colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
"""
1.5 columns (5,3.75) ; 1 column (3.54,2.65) ; 2 columns (7.25,5.43)
"""
style = {"figure.figsize": (3.54 , 2.65),
         "figure.titlesize" : 11,
         "legend.frameon": False,
         "legend.loc" : 'upper left',
         "legend.fontsize" : 11,
         "axes.labelsize" : 11,
         "savefig.bbox" : 'tight',
         "savefig.pad_inches" : 0.05,
         "savefig.dpi" : 300,
         "xtick.direction" : 'in',
         "xtick.major.size" : 4,
         "xtick.major.width" : 2,
         "xtick.minor.size" : 2,
         "xtick.minor.width" : 1,
         "xtick.minor.visible" : True,
         "xtick.top": False,
         "ytick.direction" : 'in',
         "ytick.major.size" : 4,
         "ytick.major.width" : 2,
         "ytick.minor.size" : 2,
         "ytick.minor.width" : 1,
         "ytick.minor.visible" : True,
         "ytick.right": False,
         "text.usetex" : True
         }
sns.set(context='paper',style='white',font_scale=1.5,color_codes=True, rc=style)


# Implt_justf experiment
plots_path = 'plots/'


#threshold act func
x = [-3, -2, -1, 0, 1, 2, 3]
y = [0, 0, 0, 0, 1, 1, 1]
fig = plt.step(x,y)
plt.grid(True)
plt.savefig(plots_path + 'threshold.pdf')
plt.close()


#Tanh
x = np.arange(-5,5,0.1)
plt.plot(np.tanh(x))
plt.grid(True)
plt.savefig(plots_path + 'tanh.pdf')
plt.close()
#Linear
plt.plot(x,x)
plt.grid(True)
plt.savefig(plots_path + 'linear.pdf')
plt.close()



#Sigmoid

plt.plot(1/(1+np.exp(-x)))
plt.grid(True)
plt.savefig(plots_path+'sigmoid.pdf')
plt.close()
#ReLU
plt.plot(np.maximum(0,x))
plt.grid(True)
plt.savefig(plots_path+'ReLU.pdf')
plt.close()

#LeakyReLU

z = np.arange(-55,5,1)
plt.plot(np.maximum(0.01*z,z))
plt.grid(True)
plt.savefig(plots_path+'Leaky_ReLU.pdf')
plt.close()

#Softmax
plt.plot(np.exp(x)/np.sum(np.exp(x)))
plt.grid(True)
plt.savefig(plots_path+'softmax.pdf')
plt.close()