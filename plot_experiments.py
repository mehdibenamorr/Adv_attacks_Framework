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
style = {"figure.figsize": (7.25 , 5.43),
         "figure.titlesize" : 11,
         "legend.frameon": False,
         "legend.loc" : 'upper right',
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
         "ytick.right": False
         }
sns.set(context='paper',style='white',font_scale=1.5,color_codes=True, rc=style)


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
    # f.suptitle('Implementation Validation for 784-100-100-10 \n Performance differences distribution')
    # Add a graph in each part
    sns.boxplot(df['acc_diff'] , ax=ax_box)
    sns.distplot(df['acc_diff'], ax=ax_hist, kde_kws={'label' : 'kernel density estimate', 'color' : '#ffa756'}, hist_kws={'edgecolor': 'black'})
    sns.distplot(df['acc_diff'],fit=norm,hist=False, kde=False , ax=ax_hist,
                 label='PDF estimate', fit_kws={'linestyle' : '--'})

    ax_hist.set(xlabel='Accuracy difference in $(\%)$',ylabel='Density')
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    plt.legend()
    # f.savefig(plots_path + 'Impl_validation_784_100_100_10.jpg')
    f.savefig(plots_path + 'Impl_validation_784_100_100_10.svg')
    f.savefig(plots_path + 'Impl_validation_784_100_100_10.pdf')


    #second exp
    f, (ax_box, ax_hist) = plt.subplots(2,  gridspec_kw={"height_ratios": (.15, .85)})
    # f.suptitle('Implementation Validation for 784-100-10 \n Performance differences distribution')
    # Add a graph in each part
    sns.boxplot(df1['acc_diff'], ax=ax_box)
    sns.distplot(df1['acc_diff'], ax=ax_hist, kde_kws={'label' : 'kernel density estimate', 'color' : '#ffa756'},hist_kws={'edgecolor': 'black'})
    sns.distplot(df1['acc_diff'], fit=norm, hist=False, kde=False, ax=ax_hist,
                 label='PDF estimate',fit_kws={'linestyle' : '--'})

    ax_hist.set(xlabel='Accuracy difference in $(\%)$', ylabel='Density')
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    plt.legend()
    # f.savefig(plots_path + 'Impl_validation_784_100_10.jpg')
    f.savefig(plots_path + 'Impl_validation_784_100_10.svg')
    f.savefig(plots_path + 'Impl_validation_784_100_10.pdf')

    #both performances boxplots
    dff= df
    dff.columns = ['Linear_module', 'SNN_module', 'acc_diff']
    f, ax = plt.subplots(1)
    # f.suptitle('Accuracy boxplot of the two algorithms used \n for the 784_100_10 network')
    sns.boxplot(data= dff[['Linear_module','SNN_module']], ax=ax, palette="Set2", width=0.3)
    ax.set(xlabel='Module used', ylabel='Accuracy')
    f.savefig(plots_path + 'Boxplot_Impl_validation_784_100_10.svg')
    f.savefig(plots_path + 'Boxplot_Impl_validation_784_100_10.pdf')

    #second exp
    dff = df1
    dff.columns = ['Linear_module', 'SNN_module', 'acc_diff']
    f, ax = plt.subplots(1)
    # f.suptitle('Accuracy boxplot of the two algorithms used \n for the 784_100_10 network')
    sns.boxplot(data=dff[['Linear_module', 'SNN_module']], ax=ax, palette="Set2", width=0.3)
    ax.set(xlabel='Module used', ylabel='Accuracy')
    f.savefig(plots_path + 'Boxplot_Impl_validation_784_100_100_10.svg')
    f.savefig(plots_path + 'Boxplot_Impl_validation_784_100_100_10.pdf')


# impl_just()

def ffn_experiment():
    #FGSM 1Layer experiment
    results = "tests/results/FFN1L_experiment_FGSM_0.1.csv"
    df = pd.read_csv(results, encoding='utf-8', index_col=0)
    df.columns = ['Avg_epsilon', 'Max_epsilon', 'Min_epsilon', 'Success Rate', 'Accuracy', '#params']
    df['Robustness'] = 100 - df['Success Rate'].values
    g = sns.pairplot(df[['Avg_epsilon', 'Max_epsilon', 'Accuracy' , 'Success Rate', 'Robustness', '#params']],
                     palette="Set2", x_vars= ['#params', 'Accuracy'],
                     y_vars=['Avg_epsilon','Max_epsilon','Robustness','Success Rate'], kind = 'reg' , markers= '+')
    g.savefig(plots_path + 'Pairplot_FFN1L_experiment_FGSM.pdf')
    g.savefig(plots_path + 'Pairplot_FFN1L_experiment_FGSM.svg')
    f , (ax_acc, ax_params) = plt.subplots(1, 2, sharey=True, gridspec_kw={"width_ratios": (.50, .50)})
    sns.regplot(x=df['#params'], y=df['Robustness'], ax=ax_params, marker='+')
    sns.regplot(x=df['Accuracy'], y=df['Robustness'], ax=ax_acc, marker='+')
    ax_params.set(xlabel='number of parameters', ylabel='')
    ax_acc.set(xlabel='Accuracy', ylabel='Robustness $(\%)$')
    f.savefig(plots_path + 'Scatter_plot_FFN1L_experiment_FGSM.pdf')
    f.savefig(plots_path + 'Scatter_plot_FFN1L_experiment_FGSM.svg')



def snn_experiment():
    #SNN experiment
    results = "tests/results/SNN_experiment_FGSM_0.1.csv"
    df = pd.read_csv(results, encoding='utf-8', index_col=0)



def pruning_experiment():
    pass

ffn_experiment()

# plt.show()