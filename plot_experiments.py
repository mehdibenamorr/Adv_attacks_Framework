import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
# sns.set(style='ticks', palette='Set2')
sns.set(color_codes=True)
plt.style.use(['thesis'])
# Implt_justf experiment
plots_path = 'plots/'

def impl_just():
    results = "tests/results/Implementation_exp_200.csv"
    results1 = "tests/results/Implementation_exp_100.csv"
    df = pd.read_csv(results, encoding='utf-8', index_col=0)
    df = df.drop(['FFN_Time','SNN_Time'], axis=1)
    df1 = pd.read_csv(results1, encoding='utf-8', index_col=0)
    df1 = df1.drop(['FFN_Time','SNN_Time'], axis=1)
    df['acc_diff'] = df['FFN_acc'].values - df['SNN_acc'].values
    df1['acc_diff'] = df1['FFN_acc'].values - df1['SNN_acc'].values


    #plot a subplot containing implementation experiment distribution for both density and counts histograms
    # Cut the window in 2 parts
    f, (ax_box, ax_hist) = plt.subplots(2,  gridspec_kw={"height_ratios": (.15, .85)})
    f.suptitle('Implementation Validation for 784-100-100-10 \n Performance differences distribution')
    # Add a graph in each part
    sns.boxplot(df['acc_diff'] , ax=ax_box)
    sns.distplot(df['acc_diff'],bins=10, ax=ax_hist, label= 'Accuracy differences distribution')
    sns.distplot(df['acc_diff'],fit=norm,hist=False, kde=False , ax=ax_hist, label='Normal fit')

    ax_hist.set(xlabel='Accuracy difference in $(\%)$',ylabel='Density')
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    plt.legend()
    f.savefig(plots_path + 'Impl_validation_784_100_100_10.jpg')


    #second exp
    f, (ax_box, ax_hist) = plt.subplots(2,  gridspec_kw={"height_ratios": (.15, .85)})
    f.suptitle('Implementation Validation for 784-100-10 \n Performance differences distribution')
    # Add a graph in each part
    sns.boxplot(df1['acc_diff'], ax=ax_box)
    sns.distplot(df1['acc_diff'], bins=10, ax=ax_hist, label='Accuracy differences distribution')
    sns.distplot(df1['acc_diff'], fit=norm, hist=False, kde=False, ax=ax_hist, label='Normal fit')

    ax_hist.set(xlabel='Accuracy difference in $(\%)$', ylabel='Density')
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    plt.legend()
    f.savefig(plots_path + 'Impl_validation_784_100_10.jpg')

impl_just()


#FGSM 1Layer experiment
results3 = "tests/results/FFN1L_experiment_FGSM_0.1.csv"
results4 = "tests/results/SNN**.csv"
df2 = pd.read_csv(results3, index_col=0)
import ipdb
ipdb.set_trace()




plt.show()