import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns



# Implt_justf experiment
path_to_results = "tests/results/Implementation_exp_200.csv"
df = pd.read_csv(path_to_results, encoding='utf-8', index_col=0)
df['acc_diff'] = df['FFN_acc'].values - df['SNN_acc'].values
sns.distplot(df['acc_diff'].values)
df['acc_diff'].plot.kde()
plt.show()


#FGSM 1Layer experiment
path_to_results = "tests/results/experiment_FFN_1Layer.csv"
df1 = pd.read_csv(path_to_results, index_col=0)
import ipdb
ipdb.set_trace()