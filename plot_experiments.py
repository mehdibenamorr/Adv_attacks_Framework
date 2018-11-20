import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


plt.style.use(['thesis','subfigure'])





# Implt_justf experiment
results = "tests/results/Implementation_exp_200.csv"
results1 = "tests/results/Implementation_exp_100.csv"
df = pd.read_csv(results, encoding='utf-8', index_col=0)
df = df.drop(['FFN_Time','SNN_Time'], axis=1)
df1 = pd.read_csv(results1, encoding='utf-8', index_col=0)
df1 = df1.drop(['FFN_Time','SNN_Time'], axis=1)
df['acc_diff'] = df['FFN_acc'].values - df['SNN_acc'].values
df1['acc_diff'] = df1['FFN_acc'].values - df1['SNN_acc'].values
fig , ax = plt.subplots(2,1, sharey=True)

#plot a subplot containing implementation experiment distribution for both density and counts histograms
sns.distplot(df['acc_diff'].values, label='Implementation Validation for 784-100-10 : \n Accuracy difference distribution', ax=ax[0])

plt.show()


#FGSM 1Layer experiment
results3 = "tests/results/FFN1L_experiment_FGSM_0.1.csv"
results4 = "tests/results/SNN**.csv"
df2 = pd.read_csv(results3, index_col=0)
import ipdb
ipdb.set_trace()




plt.show()