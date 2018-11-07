import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.offline as py
import plotly.figure_factory as ff



# Implt_justf experiment
results = "tests/results/Implementation_exp_200.csv"
results1 = "tests/results/Implementation_exp_100.csv"
df = pd.read_csv(results, encoding='utf-8', index_col=0)
df = df.drop(['FFN_Time','SNN_Time'], axis=1)
df1 = pd.read_csv(results1, encoding='utf-8', index_col=0)
df1 = df1.drop(['FFN_Time','SNN_Time'], axis=1)
df['acc_diff'] = df['FFN_acc'].values - df['SNN_acc'].values
df1['acc_diff'] = df1['FFN_acc'].values - df1['SNN_acc'].values
fig = ff.create_distplot([df[['FFN_acc','SNN_acc']][c] for c in df[['FFN_acc','SNN_acc']].columns], df[['FFN_acc','SNN_acc']].columns,
                            bin_size=0.25, show_hist = False )
fig['layout'].update(title= 'Implementation Validation for 784-100-10 <br> Performances Using SNN implemenation vs Using Pytorch Linear module')
fig1 = ff.create_distplot([df1[['FFN_acc','SNN_acc']][c] for c in df1[['FFN_acc','SNN_acc']].columns], df1[['FFN_acc','SNN_acc']].columns,
                            bin_size=0.25, show_hist = False )
fig1['layout'].update(title= 'Implementation Validation for 784-100-100-10 <br> Performances Using SNN implemenation vs Using Pytorch Linear module')

fig2 = ff.create_distplot([df[['acc_diff']][c] for c in df[['acc_diff']].columns], df[['acc_diff']].columns , bin_size=0.05)
fig2['layout'].update(title = 'Implementation Validation for 784-100-10 <br> Accuracy difference distribution')
fig3 = ff.create_distplot([df1[['acc_diff']][c] for c in df1[['acc_diff']].columns], df1[['acc_diff']].columns , bin_size=0.05)
fig3['layout'].update(title = 'Implementation Validation for 784-100-100-10 <br> Accuracy difference distribution')
# py.plot(fig3, filename = 'Accuracy difference distribution for 784-100-100-10')
# import ipdb
# ipdb.set_trace()
# py.plot( fig , filename= 'Implementation Validation for 784-100-10_performance')
# py.plot( fig1 , filename= 'Implementation Validation for 784-100-100-10_performance')
#
# sns.distplot(df['acc_diff'].values)
# df['acc_diff'].plot.kde()
# plt.show()


#FGSM 1Layer experiment
results3 = "tests/results/FFN1L_experiment_FGSM_0.1.csv"
results4 = "tests/results/SNN**.csv"
df2 = pd.read_csv(results3, index_col=0)
import ipdb
ipdb.set_trace()

