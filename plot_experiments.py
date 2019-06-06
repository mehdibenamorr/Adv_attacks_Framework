import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import seaborn as sns
from scipy.stats import norm
# sns.set(style='ticks', palette='Set2')
#colors = ['#E69F00', '#56B4E9', '#F0E442', '#009E73', '#D55E00']
"""
1.5 columns (5,3.75) ; 1 column (3.54,2.65) ; 2 columns (7.25,5.43)
"""
style = {"figure.figsize": (5,3.75),
         "figure.titlesize" : 11,
         "legend.frameon": False,
         "legend.loc" : 'upper right',
         "legend.fontsize" : 11,
         "axes.labelsize" : 11,
         "axes.titlesize": 11,
         "savefig.bbox" : 'tight',
         "savefig.pad_inches" : 0.05,
         "savefig.dpi" : 300,
         "xtick.direction" : 'in',
         "xtick.labelsize": 11,
         "xtick.major.size" : 4,
         "xtick.major.width" : 2,
         "xtick.minor.size" : 2,
         "xtick.minor.width" : 1,
         "xtick.minor.visible" : True,
         "xtick.top": False,
         "xtick.bottom": True,
         "ytick.direction" : 'in',
         "ytick.labelsize": 11,
         "ytick.major.size" : 4,
         "ytick.major.width" : 2,
         "ytick.minor.size" : 2,
         "ytick.minor.width" : 1,
         "ytick.minor.visible" : True,
         "ytick.right": False,
         "ytick.left":True
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
    f, (ax_box, ax_hist) = plt.subplots(2,  sharex=True,gridspec_kw={"height_ratios": (.20, .80)},figsize=(3.54,2.65))
    # f.suptitle('Implementation Validation for 784-100-100-10 \n Performance differences distribution')
    # Add a graph in each part
    sns.boxplot(df['acc_diff'] , ax=ax_box, width=0.4)
    sns.distplot(df['acc_diff'], ax=ax_hist, kde_kws={'label' : 'KDE', 'color' : '#ffa756'}, hist_kws={'edgecolor': 'black'})
    sns.distplot(df['acc_diff'],fit=norm,hist=False, kde=False , ax=ax_hist,
                 label='PDF', fit_kws={'linestyle' : '--'})

    ax_hist.set(xlabel='f1-scores difference in $(\%)$',ylabel='Density')
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    # f.savefig(plots_path + 'Impl_validation_784_100_100_10.jpg')
    f.savefig(plots_path + 'Impl_validation_784_100_100_10.svg')
    f.savefig(plots_path + 'Impl_validation_784_100_100_10.pdf')


    #second exp
    f, (ax_box, ax_hist) = plt.subplots(2,  sharex=True,gridspec_kw={"height_ratios": (.20, .80)}, figsize=(3.54,2.65))
    # f.suptitle('Implementation Validation for 784-100-10 \n Performance differences distribution')
    # Add a graph in each part
    sns.boxplot(df1['acc_diff'], ax=ax_box, width=0.4)
    sns.distplot(df1['acc_diff'], ax=ax_hist, kde_kws={'label' : 'KDE', 'color' : '#ffa756'},hist_kws={'edgecolor': 'black'})
    sns.distplot(df1['acc_diff'], fit=norm, hist=False, kde=False, ax=ax_hist,
                 label='PDF',fit_kws={'linestyle' : '--'})

    ax_hist.set(xlabel='f1-scores difference in $(\%)$', ylabel='Density')
    # Remove x axis name for the boxplot
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    # f.savefig(plots_path + 'Impl_validation_784_100_10.jpg')
    f.savefig(plots_path + 'Impl_validation_784_100_10.svg')
    f.savefig(plots_path + 'Impl_validation_784_100_10.pdf')

    #both performances boxplots
    dff= df
    dff.columns = ['Linear', 'Graph Based', 'acc_diff']
    f, ax = plt.subplots(1, figsize=(3,4))
    # f.suptitle('Accuracy boxplot of the two algorithms used \n for the 784_100_10 network')
    sns.boxplot(data= dff[['Linear','Graph Based']], ax=ax, color="w", width=0.3)
    ax.set(xlabel='Method used', ylabel='f1-score')
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')
    f.savefig(plots_path + 'Boxplot_Impl_validation_784_100_10.svg')
    f.savefig(plots_path + 'Boxplot_Impl_validation_784_100_10.pdf')

    #second exp
    dff = df1
    dff.columns = ['Linear', 'Graph Based', 'acc_diff']
    f, ax = plt.subplots(1, figsize=(3,4))
    # f.suptitle('Accuracy boxplot of the two algorithms used \n for the 784_100_10 network')
    sns.boxplot(data=dff[['Linear', 'Graph Based']], ax=ax, color="w", width=0.3)
    ax.set(xlabel='Method used', ylabel='f1-score')
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')
    f.savefig(plots_path + 'Boxplot_Impl_validation_784_100_100_10.svg')
    f.savefig(plots_path + 'Boxplot_Impl_validation_784_100_100_10.pdf')
#
#
impl_just()
#
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
    import glob
    import ast
    results = glob.glob("/home/mehdi/Desktop/Thesis-repo/results/SNNs/*.csv")
    data_dict= dict()
    for file in results:
        df_dummy = pd.read_csv(file, encoding='utf-8', index_col=0)
        data_dict.update(df_dummy.to_dict(orient='index'))

    structural_properties = []
    degree_dists = []
    for model in data_dict.keys():
        ids = model.split('_')
        idx = ids[-1]
        values = []
        columns = []
        values.append(idx)
        columns.append('Graph')
        #structural properties
        fixed_dict = ast.literal_eval(data_dict[model]['He_normal'])
        prop = fixed_dict['One_Pixel']
        degree_dist = prop.pop('degree_distribution')
        degree_dists.append(degree_dist)
        prop['std_degree'] = np.std(degree_dist)
        prop.pop('radius')
        prop['avg_degree'] = np.mean(degree_dist)
        prop['max_degree'] = np.max(degree_dist)
        # prop['min_degree'] = np.min(degree_dist)
        prop['k'] = int(ids[3])
        prop['p'] = float(ids[4])
        prop.pop('Robustness')
        prop.pop('Avg_confidence')
        prop.pop('Max_confidence')
        prop.pop('Accuracy')
        for key, item in prop.items():
            values.append(item)
            columns.append(key)
        structural_properties.append(values)
    df_structural_prop = pd.DataFrame(structural_properties,columns=columns)
    df_structural_prop = df_structural_prop.set_index('Graph')




    #FGSM

    data = {'xavier_normal': [], 'xavier_uniform_':[], 'He_normal':[], 'He_uniform':[], 'normal':[], 'uniform':[]}
    for model in data_dict.keys():
        ids = model.split('_')
        idx = ids[-1]
        for init in data_dict[model].keys():
            fixed_dict = ast.literal_eval(data_dict[model][init])
            values = []
            columns = []
            values.append(idx)
            columns.append('Graph')
            prop = fixed_dict['FGSM']
            degree_dist = prop.pop('degree_distribution')
            prop.pop('radius')
            prop['std_degree'] = np.std(degree_dist)
            prop['avg_degree'] = np.mean(degree_dist)
            prop['max_degree'] = np.max(degree_dist)
            # prop['min_degree'] = np.min(degree_dist)
            prop['k'] = int(ids[3])
            prop['p'] = float(ids[4])
            for key, item in prop.items():
                values.append(item)
                columns.append(key)
            data[init].append(values)
    dfs_fgsm = dict()
    for init in data:
        dfs_fgsm[init] = pd.DataFrame(data[init], columns=columns).set_index('Graph')
    frames= []
    for init in dfs_fgsm.keys():
        dfs_fgsm[init]['init']=init
        frames.append(dfs_fgsm[init])
    df_fgsm = pd.concat(frames)
    df_fgsm['Success Rate'] = df_fgsm['Robustness']
    df_fgsm['Robustness'] = 100-df_fgsm['Success Rate']
    #One_pixel

    data = {'xavier_normal': [], 'xavier_uniform_': [], 'He_normal': [], 'He_uniform': [], 'normal': [], 'uniform': []}
    for model in data_dict.keys():
        ids = model.split('_')
        idx = int(ids[-1])
        for init in data_dict[model].keys():
            fixed_dict = ast.literal_eval(data_dict[model][init])
            values = []
            columns = []
            values.append(idx)
            columns.append('Graph')
            prop = fixed_dict['One_Pixel']
            degree_dist = prop.pop('degree_distribution')
            prop.pop('radius')
            prop['std_degree'] = np.std(degree_dist)
            prop['avg_degree'] = np.mean(degree_dist)
            prop['max_degree'] = np.max(degree_dist)
            # prop['min_degree'] = np.min(degree_dist)
            prop['k'] = int(ids[3])
            prop['p'] = float(ids[4])
            for key, item in prop.items():
                values.append(item)
                columns.append(key)
            data[init].append(values)
    dfs_onepixel = dict()
    for init in data:
        dfs_onepixel[init] = pd.DataFrame(data[init], columns=columns).set_index('Graph')
    frames = []
    for init in dfs_onepixel.keys():
        dfs_onepixel[init]['init']=init
        frames.append(dfs_onepixel[init])
    df_onepixel = pd.concat(frames)
    df_onepixel['Success Rate'] = df_onepixel['Robustness']
    df_onepixel['Robustness'] = 100 - df_onepixel['Success Rate']

    frames = []
    df_fgsm['attack']='FGSM'
    frames.append(df_fgsm)
    df_onepixel['attack']='OnePixel'
    frames.append(df_onepixel)
    df = pd.concat(frames,sort=True)
    # mapping = {'xavier_normal':1, 'xavier_uniform_':4, 'He_normal':2, 'He_uniform':5, 'normal':3, 'uniform':6}
    mapping = {'xavier_normal': 'G_N', 'xavier_uniform_': 'G_U', 'He_normal': 'He_N', 'He_uniform': 'He_U',
               'normal': 'N', 'uniform': 'U'}
    df['init'] = df['init'].map(mapping)
    #plotting

    #f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)}, figsize=(3.54, 2.65))
    #     # f.suptitle('Implementation Validation for 784-100-100-10 \n Performance differences distribution')
    #     # Add a graph in each part
    #     sns.boxplot(df['acc_diff'] , ax=ax_box, width=0.4)
    #     sns.distplot(df['acc_diff'], ax=ax_hist, kde_kws={'label' : 'KDE', 'color' : '#ffa756'}, hist_kws={'edgecolor': 'black'})
    #     sns.distplot(df['acc_diff'],fit=norm,hist=False, kde=False , ax=ax_hist,
    #                  label='PDF', fit_kws={'linestyle' : '--'})
    #
    #     ax_hist.set(xlabel='F1 scores difference in $(\%)$',ylabel='Density')
    #     # Remove x axis name for the boxplot
    #     ax_box.set(xlabel='')
    #     plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    #     plt.setp(ax_box.lines, color='k')
    #     plt.legend()
    #     # f.savefig(plots_path + 'Impl_validation_784_100_100_10.jpg')
    #     f.savefig(plots_path + 'Impl_validation_784_100_100_10.svg')
    #     f.savefig(plots_path + 'Impl_validation_784_100_100_10.pdf')

    #parameters distribution
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)},
                                            figsize=(5,3.75))
    sns.boxplot(df_structural_prop['#params'] , ax=ax_box, width = 0.4)
    sns.distplot(df_structural_prop['#params'], ax=ax_hist, kde=False, rug=True , hist_kws={'edgecolor':'black'})
    ax_hist.set(xlabel='number of parameters', ylabel='counts')
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    f.savefig(plots_path + 'SNNs_parameters_distribution.pdf')
    plt.close(f)
    #avg_path_length distribution
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)},
                                        figsize=(3.54,2.65))
    sns.boxplot(df_structural_prop['avg_path_length'], ax=ax_box, width=0.4)
    sns.distplot(df_structural_prop['avg_path_length'], ax=ax_hist, kde=False, rug=True, hist_kws={'edgecolor': 'black'})
    ax_hist.set(xlabel='average path length', ylabel='counts')
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    f.savefig(plots_path + 'SNNs_pathlength_distribution.pdf')
    plt.close(f)
    #Diameter
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)},
                                        figsize=(3.54,2.65))
    sns.boxplot(df_structural_prop['diameter'], ax=ax_box, width=0.4)
    sns.distplot(df_structural_prop['diameter'], ax=ax_hist, kde=False, rug=True,
                 hist_kws={'edgecolor': 'black'})
    ax_hist.set(xlabel='diameter', ylabel='counts')
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    f.savefig(plots_path + 'SNNs_diameter_distribution.pdf')
    plt.close(f)
    #density
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)},
                                        figsize=(3.54,2.65))
    sns.boxplot(df_structural_prop['density'], ax=ax_box, width=0.4)
    sns.distplot(df_structural_prop['density'], ax=ax_hist, kde=False, rug=True,
                 hist_kws={'edgecolor': 'black'})
    ax_hist.set(xlabel='density', ylabel='counts')
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    f.savefig(plots_path + 'SNNs_density_distribution.pdf')
    plt.close(f)
    #avg_betweeness
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)},
                                        figsize=(3.54,2.65))
    sns.boxplot(df_structural_prop['avg_betweenness'], ax=ax_box, width=0.4)
    sns.distplot(df_structural_prop['avg_betweenness'], ax=ax_hist, kde=False, rug=True,
                 hist_kws={'edgecolor': 'black'})
    ax_hist.set(xlabel='average node betweenness', ylabel='counts')
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    f.savefig(plots_path + 'SNNs_avgbetweenness_distribution.pdf')
    plt.close(f)
    #avg closeness
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)},
                                        figsize=(3.54,2.65))
    sns.boxplot(df_structural_prop['avg_closeness'], ax=ax_box, width=0.4)
    sns.distplot(df_structural_prop['avg_closeness'], ax=ax_hist, kde=False, rug=True,
                 hist_kws={'edgecolor': 'black'})
    ax_hist.set(xlabel='average node closeness', ylabel='counts')
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    f.savefig(plots_path + 'SNNs_avgcloseness_distribution.pdf')
    plt.close(f)
    #average eccentricity
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)},
                                        figsize=(3.54,2.65))
    sns.boxplot(df_structural_prop['avg_eccentricity'], ax=ax_box, width=0.4)
    sns.distplot(df_structural_prop['avg_eccentricity'], ax=ax_hist, kde=False, rug=True,
                 hist_kws={'edgecolor': 'black'})
    ax_hist.set(xlabel='average node eccentricity', ylabel='counts')
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    f.savefig(plots_path + 'SNNs_avgeccentricity_distribution.pdf')
    plt.close()
    #average edge betweenness
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)},
                                        figsize=(3.54,2.65))
    sns.boxplot(df_structural_prop['avg_edge_betweenness'], ax=ax_box, width=0.4)
    sns.distplot(df_structural_prop['avg_edge_betweenness'], ax=ax_hist, kde=False, rug=True,
                 hist_kws={'edgecolor': 'black'})
    ax_hist.set(xlabel='average edge betweenness', ylabel='counts')
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    f.savefig(plots_path + 'SNNs_avgedgebetweeen_distribution.pdf')
    plt.close(f)

    # std degrees
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.20, .80)},
                                        figsize=(3.54, 2.65))
    sns.boxplot(df_structural_prop['std_degree'], ax=ax_box, width=0.4)
    sns.distplot(df_structural_prop['std_degree'], ax=ax_hist, kde=False, rug=True,
                 hist_kws={'edgecolor': 'black'})
    ax_hist.set(xlabel='degrees standard deviation', ylabel='counts')
    ax_box.set(xlabel='')
    plt.setp(ax_box.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_box.lines, color='k')
    plt.legend()
    f.savefig(plots_path + 'SNNs_std_degree_distribution.pdf')
    plt.close(f)

    #f= df_structural_prop[['avg_path_length', 'diameter','avg_eccentricity', 'avg_betweenness', 'avg_closeness','avg_edge_betweenness', 'density']].hist(figsize=(7.25,5.43),xlabelsize=11,ylabelsize=11,sharey=True,**{'edgecolor':'black'})

    #Investigation: size, k, p distribution for 50-60 k group
    df_group1 = df_structural_prop.loc[(df_structural_prop['#params'] <= 60000) & (df_structural_prop['#params'] >= 50000)]
    df_group2 = df_structural_prop.loc[(df_structural_prop['#params'] >= 60000)]
    f, axes = plt.subplots(2,2, figsize=(7.25,5.43))
    sns.distplot(df_structural_prop['#nodes'], ax=axes[0,0], kde=False, rug=True, hist_kws={'edgecolor': 'black'})
    sns.distplot(df_structural_prop['p'], ax=axes[0, 1], kde=False, rug=True, hist_kws={'edgecolor': 'black'})
    sns.distplot(df_structural_prop['k'], ax=axes[1, 0], kde=False, rug=True, hist_kws={'edgecolor': 'black'})

    axes[0, 0].set(xlabel='size', ylabel='counts',title='(1)')
    axes[0, 1].set(xlabel='p', ylabel='',title='(2)')
    axes[1, 0].set(xlabel='k', ylabel='counts',title='(3)')

    plt.legend()
    plt.tight_layout()
    f.savefig(plots_path + 'Inverstigation.pdf')
    plt.close(f)
    #boxplots
    # sns.set(palette='Set2')
    f, axes = plt.subplots(2, 2, figsize=(7.25, 5.43))
    sns.boxplot(x=df_group1['#nodes'],y=df_group1['#params'] ,ax=axes[0, 0],width=0.4)
    sns.swarmplot(x='#nodes', y='#params',data=df_group1, ax=axes[0, 0],hue='k')
    sns.boxplot(x=df_group1['p'],y=df_group1['#params'], ax=axes[0, 1],width=0.4)
    sns.swarmplot(x='p', y='#params',data=df_group1, ax=axes[0, 1], hue='k')
    sns.boxplot(x=df_group1['k'],y=df_group1['#params'],ax=axes[1, 0],width=0.4)
    sns.swarmplot(x='k', y='#params',data=df_group1, ax=axes[1, 0], hue='k')
    sns.boxplot(x=df_group2['k'],y=df_group2['#params'], ax=axes[1, 1],width=0.4)
    sns.swarmplot(x='k', y='#params',data=df_group2, ax=axes[1, 1],hue='k')
    axes[0, 0].set(xlabel='size (50k-60k)', ylabel='number of parameters', title='(1)')
    axes[0, 1].set(xlabel='p (50k-60k)', ylabel='', title='(2)')
    axes[1, 0].set(xlabel='k (50k-60k)', ylabel='number of parameters', title='(3)')
    axes[1, 1].set(xlabel='k (>60k)', ylabel='', title='(4)')
    for i in range(2):
        for j in range(2) :
            axes[i,j].legend().set_visible(False)
    axes[1,0].legend().set(visible=True)
    plt.setp(axes[0, 0].artists, edgecolor='k', facecolor='w')
    plt.setp(axes[0, 1].artists, edgecolor = 'k', facecolor = 'w')
    plt.setp(axes[1, 0].artists, edgecolor = 'k', facecolor = 'w')
    plt.setp(axes[1, 1].artists, edgecolor = 'k', facecolor = 'w')
    plt.setp(axes[0, 0].lines, color='k')
    plt.setp(axes[0, 1].lines, color = 'k')
    plt.setp(axes[1, 0].lines, color = 'k')
    plt.setp(axes[1, 1].lines, color = 'k')
    # plt.legend()
    plt.tight_layout()
    f.savefig(plots_path + 'Inverstigation_boxplots.pdf')
    plt.close(f)
    #
    # #fgsm result plots
    flierprops = dict(marker='o',markersize=4,markerfacecolor='w')
    f, (ax_fgsm,ax_onepixel) = plt.subplots(2, figsize=(5,3.75),sharex=True)
    sns.boxplot(x='init',y='Robustness',data=df[df['attack']=='FGSM'], ax=ax_fgsm,width=0.4,flierprops=flierprops)
    sns.boxplot(x='init', y='Robustness', data=df[df['attack'] == 'OnePixel'], width=0.4, ax=ax_onepixel,flierprops=flierprops)
    ax_fgsm.set(xlabel='',ylabel='robustness $(\%)$',title='(FGSM)')
    plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_fgsm.lines, color='k')
    ax_onepixel.set(xlabel='initialization', ylabel='robustness $(\%)$', title='(OnePixel)')
    plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_robustness_init.pdf')
    plt.close(f)
    #
    f, (ax_fgsm, ax_onepixel) = plt.subplots(2, figsize=(5,3.75), sharex=True)
    sns.boxplot(x='init', y='Avg_confidence', data=df[df['attack'] == 'FGSM'], width=0.4, ax=ax_fgsm,flierprops=flierprops)
    sns.boxplot(x='init', y='Avg_confidence', data=df[df['attack'] == 'OnePixel'], width=0.4, ax=ax_onepixel,flierprops=flierprops)
    ax_fgsm.set(xlabel='', ylabel='average confidence', title='(FGSM)')
    plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_fgsm.lines, color='k')
    ax_onepixel.set(xlabel='initialization', ylabel='average confidence', title='(OnePixel)')
    plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_avgconf_init.pdf')
    plt.close(f)

    f,ax = plt.subplots(1,figsize=(5,3.75))
    sns.boxplot(x='init',y='Avg_epsilon',data=df[df['attack']=='FGSM'],width=0.4,ax=ax,flierprops=flierprops)
    ax.set(xlabel='initialization',ylabel='average epsilon',title='(FGSM)')
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_avgeps_init.pdf')
    plt.close(f)

    f, ax = plt.subplots(1, figsize=(5, 3.75))
    sns.boxplot(x='init', y='Accuracy', data=df[df['attack'] == 'FGSM'], width=0.4, ax=ax,flierprops=flierprops)
    ax.set(xlabel='initialization', ylabel='f1-score $(\%)$')
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_accuracy_init.pdf')
    plt.close(f)

    f, ax = plt.subplots(1, figsize=(5, 3.75))
    sns.boxplot(x='init', y='Robustness', data=df[df['attack'] == 'FGSM'], width=0.4, ax=ax,flierprops=flierprops)
    ax.set(xlabel='initialization', ylabel='robustness')
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_robust_init.pdf')
    plt.close(f)



    print('mehdi')
    #Correlation analysis
    #aggregate the values and remove the outliers
    # df = df.reset_index()
    fgsm = df[df['attack']=='FGSM'][['Robustness','Avg_confidence','Avg_epsilon']]
    op = df[df['attack'] == 'OnePixel'][['Robustness', 'Avg_confidence']]
    Q1= fgsm.quantile(0.25)
    Q3 = fgsm.quantile(0.75)
    IQR = Q3 - Q1
    A = df[df['attack'] == 'FGSM'][~((fgsm < (Q1 - 1.5 * IQR)) | (fgsm > (Q3 + 1.5 * IQR))).any(axis=1)]

    q1 = op.quantile(0.25)
    q3 = op.quantile(0.75)
    iqr = q3 - q1
    B = df[df['attack'] == 'OnePixel'][~((op < (q1 - 1.5 * iqr)) | (op > (q3 + 1.5 * iqr))).any(axis=1)]

    df1 = pd.concat([A,B])

    df1_med = df1.reset_index().groupby(['Graph','attack']).median().reset_index()
    df1_mean = df1.reset_index().groupby(['Graph','attack']).mean().reset_index()


    corr_mean_fgsm_spearman = df1_mean[df1_mean['attack']=='FGSM'][['Robustness', 'Avg_confidence', 'Avg_epsilon', '#params', 'std_degree', 'avg_edge_betweenness',
                       'avg_eccentricity', 'avg_closeness', 'avg_betweenness', 'density', 'diameter',
                       'avg_path_length']].corr(method='spearman')
    corr_mean_onepixel_spearman = df1_mean[df1_mean['attack'] == 'OnePixel'][
        ['Robustness', 'Avg_confidence', '#params', 'std_degree', 'avg_edge_betweenness',
         'avg_eccentricity', 'avg_closeness', 'avg_betweenness', 'density', 'diameter',
         'avg_path_length']].corr(method='spearman')
    corr_mean_fgsm_kendall = df1_mean[df1_mean['attack'] == 'FGSM'][
        ['Robustness', 'Avg_confidence', 'Avg_epsilon', '#params', 'std_degree', 'avg_edge_betweenness',
         'avg_eccentricity', 'avg_closeness', 'avg_betweenness', 'density', 'diameter',
         'avg_path_length']].corr(method='kendall')
    corr_mean_onepixel_kendall = df1_mean[df1_mean['attack'] == 'OnePixel'][
        ['Robustness', 'Avg_confidence', '#params', 'std_degree', 'avg_edge_betweenness',
         'avg_eccentricity', 'avg_closeness', 'avg_betweenness', 'density', 'diameter',
         'avg_path_length']].corr(method='kendall')

    corr_med_fgsm_spearman = df1_med[df1_med['attack'] == 'FGSM'][['Robustness', 'Avg_confidence', 'Avg_epsilon', '#params', 'std_degree', 'avg_edge_betweenness',
                       'avg_eccentricity', 'avg_closeness', 'avg_betweenness', 'density', 'diameter',
                       'avg_path_length']].corr(method='spearman')
    corr_med_onepixel_spearman = df1_med[df1_med['attack'] == 'Onepixel'][
        ['Robustness', 'Avg_confidence', 'Avg_epsilon', '#params', 'std_degree', 'avg_edge_betweenness',
         'avg_eccentricity', 'avg_closeness', 'avg_betweenness', 'density', 'diameter',
         'avg_path_length']].corr(method='spearman')

    corr_med_fgsm_kendall = df1_med[df1_med['attack'] == 'FGSM'][
        ['Robustness', 'Avg_confidence', 'Avg_epsilon', '#params', 'std_degree', 'avg_edge_betweenness',
         'avg_eccentricity', 'avg_closeness', 'avg_betweenness', 'density', 'diameter',
         'avg_path_length']].corr(method='kendall')
    corr_med_onepxel_kendall = df1_med[df1_med['attack'] == 'OnePixel'][
        ['Robustness', 'Avg_confidence', 'Avg_epsilon', '#params', 'std_degree', 'avg_edge_betweenness',
         'avg_eccentricity', 'avg_closeness', 'avg_betweenness', 'density', 'diameter',
         'avg_path_length']].corr(method='kendall')

    print('mehdi')
    """
    1.5 columns (5,3.75) ; 1 column (3.54,2.65) ; 2 columns (7.25,5.43)
    """

    f, ax_fgsm = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='density', y='Robustness', data=df1_mean[df1_mean['attack']=='FGSM'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='density', ylabel='robustness $(\%)$')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_robust_density.pdf')
    plt.close(f)

    f, ax_fgsm = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='#params', y='Robustness', data=df1_mean[df1_mean['attack'] == 'FGSM'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='number of parameters', ylabel='robustness $(\%)$')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_robust_params.pdf')
    plt.close(f)

    f, ax_fgsm = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='avg_path_length', y='Avg_confidence', data=df1_mean[df1_mean['attack'] == 'FGSM'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='Average path length', ylabel='average confidence')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_conf_path.pdf')
    plt.close(f)

    f, ax_fgsm = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='density', y='Avg_confidence', data=df1_mean[df1_mean['attack'] == 'FGSM'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_conf_density.pdf')
    plt.close(f)

    f, ax_fgsm = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='#params', y='Avg_epsilon', data=df1_mean[df1_mean['attack'] == 'FGSM'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='number of parameters', ylabel='average epsilon')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_eps_params.pdf')
    plt.close(f)

    f, ax_fgsm = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='avg_path_length', y='Avg_epsilon', data=df1_mean[df1_mean['attack'] == 'FGSM'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='average path length', ylabel='average epsilon')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_eps_path.pdf')
    plt.close(f)

    f, ax_fgsm = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='#params', y='Robustness', data=df1_mean[df1_mean['attack'] == 'OnePixel'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='number of parameters', ylabel='robustness $(\%)$')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_robust_params_op.pdf')
    plt.close(f)

    f, ax_fgsm = plt.subplots(1, figsize=(3.54, 2.65))
    sns.regplot(x='avg_eccentricity', y='Robustness', data=df1_mean[df1_mean['attack'] == 'OnePixel'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='average eccentricity', ylabel='robustness $(\%)$')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_robust_avgecc_op.pdf')
    plt.close(f)

    f, ax_fgsm = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='density', y='Avg_confidence', data=df1_mean[df1_mean['attack'] == 'OnePixel'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_conf_density_op.pdf')
    plt.close(f)

    f, ax_fgsm = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='#params', y='Avg_confidence', data=df1_mean[df1_mean['attack'] == 'OnePixel'], ax=ax_fgsm,marker='+')
    ax_fgsm.set(xlabel='number of parameters', ylabel='average confidence')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='density', ylabel='average confidence')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'relation_conf_params_op.pdf')
    plt.close(f)

    print('mehdi')
    # sns.pairplot(df[df['attack']=='FGSM'][['Robustness','Avg_epsilon','Avg_confidence','Accuracy','avg_path_length', 'diameter','avg_eccentricity', 'avg_betweenness',
    #                                         'avg_closeness','avg_edge_betweenness', 'density','#params']]
    #              ,palette="Set2",x_vars=['avg_path_length', 'diameter','avg_eccentricity', 'avg_betweenness',
    #                                         'avg_closeness','avg_edge_betweenness', 'density','#params'],
    #              y_vars=['Robustness','Avg_epsilon','Avg_confidence',
    #                                         'Accuracy'],kind='reg')

    # g = sns.lmplot(x="#params", y="Robustness", data=df_fgsm, hue='init', fit_reg=False)
    # sns.regplot(x="#params", y="Robustness", data=df_fgsm,scatter=False, ax=g.axes[0, 0])
    # g.savefig(plots_path + 'Robust_params.pdf')
    # g = sns.lmplot(x="#params", y="Avg_epsilon", data=df_fgsm, hue='init', fit_reg=False)
    # sns.regplot(x="#params", y="Avg_epsilon", data=df_fgsm, scatter=False, ax=g.axes[0, 0])
    # g.savefig(plots_path+'avg_epsilon_params.pdf')
    # g = sns.lmplot(x="#params", y="Avg_confidence", data=df_fgsm, hue='init', fit_reg=False)
    # sns.regplot(x="#params", y="Avg_confidence", data=df_fgsm, scatter=False, ax=g.axes[0, 0])
    # g.savefig(plots_path + 'avg_confidence_params.pdf')
    # g = sns.lmplot(x="avg_path_length", y="Robustness", data=df_fgsm, hue='init', fit_reg=False)
    # sns.regplot(x="avg_path_length", y="Robustness", data=df_fgsm, scatter=False, ax=g.axes[0, 0])
    # g.savefig(plots_path + 'Robust_avg_path_length.pdf')





snn_experiment()

def pruning_experiment():
    import glob
    import ast
    # results = glob.glob("/home/mehdi/Desktop/results/randompruning/pruning_experiment_run_*.csv")
    # data_dict = dict()
    # for file in results:
    #     df_dummy = pd.read_csv(file, encoding='utf-8', index_col=0).transpose()
    #     data_dict.update(df_dummy.to_dict(orient='index'))
    #
    #
    # #FGSM
    # data = {'xavier_normal': [], 'xavier_uniform_': [], 'He_normal': [], 'He_uniform': [], 'normal': [], 'uniform': []}
    # for run in data_dict.keys():
    #
    #     for init in data_dict[run].keys():
    #         fixed_dict = ast.literal_eval(data_dict[run][init])
    #         for step in fixed_dict.keys():
    #             values = []
    #             columns = []
    #             values.append(int(run[-1]))
    #             columns.append('run')
    #             prop = fixed_dict[step]['FGSM']
    #             prop['pruning_step'] = int(step.split('_')[-1])
    #             degree_dist = prop.pop('degree_distribution')
    #             eccentricity = prop.pop('eccentricity_distribution')
    #             closeness = prop.pop('closeness_distribution')
    #             prop['std_closeness'] = np.std(closeness)
    #             prop['std_eccentricity'] = np.std(eccentricity)
    #             prop.pop('radius')
    #             prop['avg_degree'] = np.mean(degree_dist)
    #             prop['max_degree'] = np.max(degree_dist)
    #             # prop['min_degree'] = np.min(degree_dist)
    #             for key, item in prop.items():
    #                 values.append(item)
    #                 columns.append(key)
    #             data[init].append(values)
    # dfs_fgsm = dict()
    # for init in data:
    #     dfs_fgsm[init] = pd.DataFrame(data[init], columns=columns)
    # frames = []
    # for init in dfs_fgsm.keys():
    #     dfs_fgsm[init]['init'] = init
    #     frames.append(dfs_fgsm[init])
    # df_fgsm = pd.concat(frames)
    # df_fgsm['Success Rate'] = df_fgsm['Robustness']
    # df_fgsm['Robustness'] = 100 - df_fgsm['Success Rate']
    #
    #
    # #One Pixel
    # data = {'xavier_normal': [], 'xavier_uniform_': [], 'He_normal': [], 'He_uniform': [], 'normal': [], 'uniform': []}
    # for run in data_dict.keys():
    #
    #     for init in data_dict[run].keys():
    #         fixed_dict = ast.literal_eval(data_dict[run][init])
    #         for step in fixed_dict.keys():
    #             values = []
    #             columns = []
    #             values.append(int(run[-1]))
    #             columns.append('run')
    #             prop = fixed_dict[step]['One_Pixel']
    #             prop['pruning_step'] = int(step.split('_')[-1])
    #             degree_dist = prop.pop('degree_distribution')
    #             prop.pop('eccentricity_distribution')
    #             prop.pop('closeness_distribution')
    #             prop.pop('radius')
    #             prop['avg_degree'] = np.mean(degree_dist)
    #             prop['max_degree'] = np.max(degree_dist)
    #             # prop['min_degree'] = np.min(degree_dist)
    #             for key, item in prop.items():
    #                 values.append(item)
    #                 columns.append(key)
    #             data[init].append(values)
    # dfs_onepixel = dict()
    # for init in data:
    #     dfs_onepixel[init] = pd.DataFrame(data[init], columns=columns)
    # frames = []
    # for init in dfs_onepixel.keys():
    #     dfs_onepixel[init]['init'] = init
    #     frames.append(dfs_onepixel[init])
    # df_onepixel = pd.concat(frames)
    # df_onepixel['Success Rate'] = df_onepixel['Robustness']
    # df_onepixel['Robustness'] = 100 - df_onepixel['Success Rate']
    #
    # frames = []
    # df_fgsm['attack'] = 'FGSM'
    # frames.append(df_fgsm)
    # df_onepixel['attack'] = 'OnePixel'
    # frames.append(df_onepixel)
    # df3 = pd.concat(frames,sort=True)
    # # mapping = {'xavier_normal': 1, 'xavier_uniform_': 4, 'He_normal': 2, 'He_uniform': 5, 'normal': 3, 'uniform': 6}
    # mapping = {'xavier_normal': 'G_N', 'xavier_uniform_': 'G_U', 'He_normal': 'He_N', 'He_uniform': 'He_U',
    #            'normal': 'N', 'uniform': 'U'}
    # df3['init'] = df3['init'].map(mapping)

    results = glob.glob("/home/mehdi/Desktop/Thesis-repo/results/randompruning4/pruning_experiment_run_?.csv")
    data_dict = dict()
    for file in results:
        df_dummy = pd.read_csv(file, encoding='utf-8', index_col=0).transpose()
        data_dict.update(df_dummy.to_dict(orient='index'))

    # FGSM
    data = {'xavier_normal': [], 'xavier_uniform_': [], 'He_normal': [], 'He_uniform': [], 'normal': [], 'uniform': []}
    for run in data_dict.keys():

        for init in data_dict[run].keys():
            fixed_dict = ast.literal_eval(data_dict[run][init])
            for step in fixed_dict.keys():
                values = []
                columns = []
                values.append(int(run[-1]))
                columns.append('run')
                prop = fixed_dict[step]['FGSM']
                prop['pruning_step'] = int(step.split('_')[-1])
                degree_dist = prop.pop('degree_distribution')
                prop['std_degree'] = np.std(degree_dist)
                eccentricity = prop.pop('eccentricity_distribution')
                closeness = prop.pop('closeness_distribution')
                prop['std_closeness'] = np.std(closeness)
                prop['std_eccentricity'] = np.std(eccentricity)
                prop.pop('radius')
                prop['avg_degree'] = np.mean(degree_dist)
                prop['max_degree'] = np.max(degree_dist)
                # prop['min_degree'] = np.min(degree_dist)
                for key, item in prop.items():
                    values.append(item)
                    columns.append(key)
                data[init].append(values)
    dfs_fgsm = dict()
    for init in data:
        dfs_fgsm[init] = pd.DataFrame(data[init], columns=columns)
    frames = []
    for init in dfs_fgsm.keys():
        dfs_fgsm[init]['init'] = init
        frames.append(dfs_fgsm[init])
    df_fgsm = pd.concat(frames)
    df_fgsm['Success Rate'] = df_fgsm['Robustness']
    df_fgsm['Robustness'] = 100 - df_fgsm['Success Rate']

    # One Pixel
    data = {'xavier_normal': [], 'xavier_uniform_': [], 'He_normal': [], 'He_uniform': [], 'normal': [], 'uniform': []}
    for run in data_dict.keys():

        for init in data_dict[run].keys():
            fixed_dict = ast.literal_eval(data_dict[run][init])
            for step in fixed_dict.keys():
                values = []
                columns = []
                values.append(int(run[-1]))
                columns.append('run')
                prop = fixed_dict[step]['One_Pixel']
                prop['pruning_step'] = int(step.split('_')[-1])
                degree_dist = prop.pop('degree_distribution')
                prop['std_degree'] = np.std(degree_dist)
                eccentricity = prop.pop('eccentricity_distribution')
                closeness = prop.pop('closeness_distribution')
                prop['std_closeness'] = np.std(closeness)
                prop['std_eccentricity'] = np.std(eccentricity)
                prop.pop('radius')
                prop['avg_degree'] = np.mean(degree_dist)
                prop['max_degree'] = np.max(degree_dist)
                # prop['min_degree'] = np.min(degree_dist)
                for key, item in prop.items():
                    values.append(item)
                    columns.append(key)
                data[init].append(values)
    dfs_onepixel = dict()
    for init in data:
        dfs_onepixel[init] = pd.DataFrame(data[init], columns=columns)
    frames = []
    for init in dfs_onepixel.keys():
        dfs_onepixel[init]['init'] = init
        frames.append(dfs_onepixel[init])
    df_onepixel = pd.concat(frames)
    df_onepixel['Success Rate'] = df_onepixel['Robustness']
    df_onepixel['Robustness'] = 100 - df_onepixel['Success Rate']

    frames = []
    df_fgsm['attack'] = 'FGSM'
    frames.append(df_fgsm)
    df_onepixel['attack'] = 'OnePixel'
    frames.append(df_onepixel)
    df4 = pd.concat(frames,sort=True)
    # mapping = {'xavier_normal': 1, 'xavier_uniform_': 4, 'He_normal': 2, 'He_uniform': 5, 'normal': 3, 'uniform': 6}
    mapping = {'xavier_normal': 'G_N', 'xavier_uniform_': 'G_U', 'He_normal': 'He_N', 'He_uniform': 'He_U',
               'normal': 'N', 'uniform': 'U'}
    df4['init'] = df4['init'].map(mapping)


    #robustness vs pruning
    #df4[(df4['attack']=='FGSM')].groupby(['init']).mean()
    f, (ax_model1, ax_model2) = plt.subplots(2, figsize=(5, 3.75), sharex=True)
    sns.boxplot(x='pruning_step', y='Robustness', data=df4[df4['attack']=='FGSM'],    ax=ax_model1, width=0.4,fliersize=2)
    sns.boxplot(x='pruning_step', y='Robustness', data=df4[df4['attack']=='OnePixel'], width=0.4, ax=ax_model2,fliersize=2)
    ax_model1.set(xlabel='', ylabel='Robustness $(\%)$', title='(FGSM)')
    plt.setp(ax_model1.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_model1.lines, color='k')
    ax_model2.set(xlabel='pruning step', ylabel='Robustness $(\%)$', title='(OnePixel)')
    plt.setp(ax_model2.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_model2.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_robustness_pruning.pdf')
    plt.close(f)

    f, (ax_model1, ax_model2) = plt.subplots(2, figsize=(5, 3.75), sharex=True)
    sns.boxplot(x='pruning_step', y='Avg_confidence', data=df4[df4['attack']=='FGSM'],     ax=ax_model1, width=0.4,fliersize=2)
    sns.boxplot(x='pruning_step', y='Avg_confidence', data=df4[df4['attack']=='OnePixel'], width=0.4, ax=ax_model2,fliersize=2)
    ax_model1.set(xlabel='', ylabel='Avgerage confidence', title='(FGSM)')
    plt.setp(ax_model1.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_model1.lines, color='k')
    ax_model2.set(xlabel='pruning step', ylabel='Average confidence', title='(OnePixel)')
    plt.setp(ax_model2.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_model2.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_avgconf_pruning.pdf')
    plt.close(f)

    f, ax = plt.subplots(1, figsize=(5, 3.75), sharex=True)
    sns.boxplot(x='pruning_step', y='Accuracy', data=df4[df4['attack'] == 'FGSM'], width=0.4, ax=ax,fliersize=2)
    ax_model1.set(xlabel='', ylabel='f1-measure $(\%)$', title='Model(1)')
    plt.setp(ax_model1.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_model1.lines, color='k')
    ax_model2.set(xlabel='pruning step', ylabel='f1-measure $(\%)$', title='Model(2)')
    plt.setp(ax_model2.artists, edgecolor='k', facecolor='w')
    plt.setp(ax_model2.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_accuracies_pruning.pdf')


    f, ax = plt.subplots(1, figsize=(5, 3.75))
    sns.boxplot(x='pruning_step', y='Avg_epsilon', data=df4[df4['attack']=='FGSM'],     ax=ax, width=0.4,fliersize=2)
    ax.set(xlabel='pruning step', ylabel='Average $\epsilon$', title='FGSM')
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_avgeps_pruning.pdf')
    plt.close(f)

    #correlation analysis
    #aggregate variables
    df_agg = df4.groupby(['attack'])
    corr_mean_fgsm_spearman = df4[df4['attack'] == 'FGSM'][
        ['Robustness', 'Avg_confidence', 'Avg_epsilon', '#params', 'std_degree', 'avg_edge_betweenness',
         'std_eccentricity', 'std_closeness','avg_closeness', 'avg_betweenness', 'density', 'diameter',
         'avg_path_length']].corr(method='spearman')
    corr_mean_onepixel_spearman = df4[df4['attack'] == 'OnePixel'][
        ['Robustness', 'Avg_confidence', '#params', 'std_degree', 'avg_edge_betweenness',
         'std_eccentricity', 'std_closeness', 'avg_closeness', 'avg_betweenness', 'density', 'diameter',
         'avg_path_length']].corr(method='spearman')

    f, ax1 = plt.subplots(1, figsize=(3.54,2.65))
    sns.regplot(x='pruning_step',y='density',data=df4[df4['attack']=='FGSM'],ax=ax1,fit_reg=False,marker='+')

    ax1.set(xlabel='pruning step', ylabel='density')
    plt.tight_layout()
    f.savefig(plots_path+'density_pruning.pdf')


    f, ax2 = plt.subplots(1, figsize=(3.54, 2.65))
    sns.regplot(x='pruning_step', y='avg_path_length', data=df4[df4['attack'] == 'FGSM'], ax=ax2, fit_reg=False,marker='+')
    ax2.set(xlabel='pruning step', ylabel='avgerage path length')
    plt.tight_layout()
    f.savefig(plots_path + 'avg_path_length_pruning.pdf')

    f,ax3=plt.subplots(1,figsize=(3.54, 2.65))
    sns.regplot(x='pruning_step', y='#params', data=df4[df4['attack'] == 'FGSM'], ax=ax3, fit_reg=False,marker='+')
    ax3.set(xlabel='pruning step', ylabel='number of parameters')
    plt.tight_layout()
    f.savefig(plots_path + 'params_pruning.pdf')

    f, ax3 = plt.subplots(1, figsize=(3.54, 2.65))
    sns.regplot(x='pruning_step', y='avg_eccentricity', data=df4[df4['attack'] == 'FGSM'], ax=ax3, fit_reg=False, marker='+')
    ax3.set(xlabel='pruning step', ylabel='average eccentricity')
    plt.tight_layout()
    f.savefig(plots_path + 'avgecc_pruning.pdf')
    print('mehdi')

pruning_experiment()
def pruning_experiment4():
    import glob
    import ast
    results = glob.glob("/home/mehdi/Desktop/results/randompruning4/pruning_experiment_runs_per*.csv")
    dfz= []
    for file in results:
        data_dict = dict()
        df_dummy = pd.read_csv(file, encoding='utf-8', index_col=0).transpose()
        data_dict.update(df_dummy.to_dict(orient='index'))


        #FGSM
        data = {'xavier_normal': [], 'xavier_uniform_': [], 'He_normal': [], 'He_uniform': [], 'normal': [], 'uniform': []}
        for run in list(data_dict.keys())[:10]:

            for init in data_dict[run].keys():
                values = []
                columns = []
                fixed_dict = ast.literal_eval(data_dict[run][init])
                values.append(int(run[-1]))
                columns.append('run')
                values.append(fixed_dict['Accuracy'])
                columns.append('Accuracy')
                data[init].append(values)
        columns=['run','Accuracy']
        dfs = dict()
        for init in data:
            dfs[init] = pd.DataFrame(data[init], columns=columns)
        frames = []
        for init in dfs.keys():
            dfs[init]['init'] = init
            frames.append(dfs[init])
        df = pd.concat(frames)
        mapping = {'xavier_normal': 'G_N', 'xavier_uniform_': 'G_U', 'He_normal': 'He_N', 'He_uniform': 'He_U',
                   'normal': 'N', 'uniform': 'U'}
        df['init'] = df['init'].map(mapping)
        df = df.reset_index(drop=True)
        dfz.append(df)
    # df3 = dfz[0]
    df4 = dfz[0]

    # f, (ax_fgsm, ax_onepixel) = plt.subplots(2, figsize=(5, 3.75), sharex=True)
    # sns.boxplot(x='init', y='Accuracy', data=df3, ax=ax_fgsm, width=0.4, whis="range")
    # sns.boxplot(x='init', y='Accuracy', data=df4, width=0.4, ax=ax_onepixel, whis="range")
    # ax_fgsm.set(xlabel='', ylabel='f1-measure $(\%)$', title='Model(1)')
    # plt.setp(ax_fgsm.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_fgsm.lines, color='k')
    # ax_onepixel.set(xlabel='initialization', ylabel='f1-measure $(\%)$', title='Model(2)')
    # plt.setp(ax_onepixel.artists, edgecolor='k', facecolor='w')
    # plt.setp(ax_onepixel.lines, color='k')
    # plt.tight_layout()
    # f.savefig(plots_path + 'distribution_accuracy_pruning.pdf')
    # plt.close(f)
    f, ax = plt.subplots(1, figsize=(5, 3.75))
    sns.boxplot(x='init', y='Accuracy', data=df4, ax=ax, width=0.4, fliersize=2)
    ax.set(xlabel='initialization', ylabel='f1-measure $(\%)$')
    plt.setp(ax.artists, edgecolor='k', facecolor='w')
    plt.setp(ax.lines, color='k')
    plt.tight_layout()
    f.savefig(plots_path + 'distribution_accuracy_pruning.pdf')
    plt.close(f)
    print('mehdi')

# pruning_experiment4()
# ffn_experiment()

# plt.show()