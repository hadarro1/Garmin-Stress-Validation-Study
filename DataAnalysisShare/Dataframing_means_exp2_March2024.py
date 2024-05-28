# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 13:03:03 2022

@author: hadar
"""

#%%
import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import scipy.stats as stat
#import matplotlib.patches as mpatches
import seaborn as sns
import os
# from fun_data_organize import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from pydoc import help
from statistics import stdev

from scipy.stats.stats import pearsonr
import math 
import scipy
from scipy import stats

general_path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2'
data_path = f'{general_path}/Data_Analysis'
res_path = f'{data_path}/Results'


#   #   for smpeling shorter periods from the tasks:
# inp = input('press 1 if you want to sample the data, \n press 0 if you want to analyse the full data\n')
inp = 0

# samp_duration = int(input('type the sample length (3 or 6 minutes) ?' ))
samp_duration = 3
# samp_duration = 6
dur = 15

sub_error = [int(i) for i in['403','404', '406', '409','412','413', '414', '415', '418', '429', '431', '432', '433', '435', '439', '445', '446', '454', '457','458','459', '460', '463', '464','472','474','475','480','488','489']]

selected_metrics = ['Mean_HR_bpm','RMSSD_ms','SD2_SD1_ratio','LF_HF_ratio_AR','HFpow_AR_nu', 'garmin_stress']                 

df_all = pd.read_csv(f'{res_path}/df_all.csv')

#%%

"""
# ===========================================================================
###############    import subject info (physical and behavioral) ############
# ===========================================================================

subjects = pd.read_csv(f'{res_path}/subjects.csv')
print(subjects.head(3))

subjects = subjects[~subjects['sub'].isin(sub_error)]
subjects.columns
subjects = subjects[['sub','age','gender','bmi_Kg/m^2','final_quiz_score',
                     'PSS-14_score','reported_stress_1', 'reported_stress_2', 'reported_stress_3','missing_stress_points']]

# sub_error = [319,332,333]
# subjects = subjects[(subjects['sub'] != 319) & (subjects['sub'] != 332) & (subjects['sub'] != 333)].reset_index()

"""

#%%

# ===========================================================================
# ===============            create table of means            ===============
# ===========================================================================

def fun_means_wide(df_all):
    cols = ['sub', 'part', 'sample_num'] + selected_metrics
    df = df_all[cols]
    # df_mean_sd = df.groupby(['sub','part'], as_index=False).agg([np.mean, np.std])
    df_mean_wide = df_all.groupby(['sub','part'], as_index=False).mean()
    df_mean_wide = df_mean_wide.drop('sample_num', axis=1)
    df_mean_melt = df_mean_wide.melt(id_vars=['sub', 'part'])  
    return df_mean_wide, df_mean_melt                               


df_mean_wide, df_mean_melt = fun_means_wide(df_all)
# df_mean_wide_norm, df_mean_melt_norm = fun_means_wide(df_norm_all)

# df_mean_wide.to_csv(f'{data_path}/Results/df_mean_wide.csv', index=False)       
# df_mean_melt.to_csv(f'{data_path}/Results/df_mean_melt.csv', index=False)

#%%

##############  Edit df_mean_wide :

    # subset to the selected metrics only and rename some columns
df_mean_wide_edit = df_mean_wide.rename(columns={'sub': 'subject', "part": "condition"})[['subject', 'condition'] + selected_metrics]
# replace the parts names to condition names
df_mean_wide_edit['condition'] = df_mean_wide_edit['condition'].replace({'p1': 'Baseline', 'p2': 'Stress', 'p3': 'Recovery'})

#%%
###############  add reported stress to the df_mean_wide
############################################################
    
###  add reported stress from "subjects_full"
subjects_full = pd.read_csv(f'{res_path}/subjects_full.csv')

subjects_full.columns

reported_stress = subjects_full[['subject','reported_stress_1', 'reported_stress_2', 'reported_stress_3']]
reported_stress = reported_stress.rename(columns={'reported_stress_1': 'Baseline_reported_stress', 
                                                  'reported_stress_2':'Stress_reported_stress', 
                                                  'reported_stress_3': 'Recovery_reported_stress'})

# Merge the two DataFrames based on Subject and Condition
df_mean_wide_edit = pd.merge(df_mean_wide_edit, reported_stress, on='subject', how='right')
# Assign reported stress to the specific row of each condition
df_mean_wide_edit['reported_stress'] = df_mean_wide_edit.apply(lambda row: row[row['condition'] + '_reported_stress'], axis=1)
df_mean_wide_edit = df_mean_wide_edit.drop(columns=['Baseline_reported_stress', 'Stress_reported_stress', 'Recovery_reported_stress'])
   

print(df_mean_wide_edit)    

# Export the edited table
df_mean_wide_edit.to_csv(f'{res_path}/df_mean_wide_edit.csv', index=False)      

       

#%%

# *******   create a table with only one row per subject     **********
#           (from the df_mean_wide table)   
# ==========================================================================


#############   to change the columns names according to the new naming in the following part:

data = pd.read_csv(f'{res_path}/df_mean_wide_edit.csv')
subjects_full = pd.read_csv(f'{res_path}/subjects_full.csv')

# selected_metrics = ['Mean_HR_bpm','RMSSD_ms','SD2_SD1_ratio','LF_HF_ratio_AR','HFpow_AR_nu', 'garmin_stress']                 
data.rename(columns = {'Mean_HR_bpm':'HR', 'RMSSD_ms':'RMSSD', 'SD2_SD1_ratio':'SD2/SD1', 'HFpow_AR_nu':'HF_nu','LF_HF_ratio_AR':'LF/HF'}, inplace = True)

####  reshape table from 3 row per subject into 1 row per subject
df_means = data.pivot(index='subject', columns='condition')
df_means = df_means.reset_index(level=[0,0])
# df_means.columns.name = None
 
                 
df_means.columns = ['_'.join((col[0], str(col[1]))) for col in df_means.columns]
df_means.shape

# df_means.columns = df_means.columns.droplevel(0)
# df_subjects = subjects_full.drop(['sub'], axis = 1) 

display(df_means)
df_means.rename(columns = {'subject_':'subject'}, inplace = True)

#%%

#############  Megre physiological data (HR,HRV,Garmin) with subjects data (personal characteristics)
# ==========================================================================

df_means_all = pd.merge(df_means, subjects_full, on='subject', how='left', right_index=False)
df_means_all = df_means_all.drop(['missing_stress_points', 'reported_stress_1', 'reported_stress_2','reported_stress_3'], axis = 1) 


df_means_all.shape


#%%

################  Create Delta columns for all selected metrics

# selected_metrics = ['Mean_HR_bpm','RMSSD_ms','SD2_SD1_ratio','LF_HF_ratio_AR','HFpow_AR_nu', 'garmin_stress']                 
selected_metrics = ['HR', 'RMSSD', 'SD2/SD1', 'LF/HF', 'HF_nu', 'garmin_stress', 'reported_stress']

df = df_means_all
# Calculate deltas for each selected metric
for metric in selected_metrics:
    stress_col = f"{metric}_Stress"
    baseline_col = f"{metric}_Baseline"
    delta_col = f"{metric}_Delta"
    df[delta_col] = df[stress_col] - df[baseline_col]

print(df)

df_means_all = df

#%%

# Replace "-" with NaN
df_means_all = df_means_all.replace('-', pd.NA)

columns_to_convert = ['PSS-14_score', 'math_quiz_score']
df_means_all[columns_to_convert] = df_means_all[columns_to_convert].apply(pd.to_numeric, errors='coerce')
df_means_all.dtypes

df_means_all = df_means_all.drop(columns=['Unnamed: 0'])


#%%

df_means_all.to_csv(f'{data_path}/Results/df_means_all.csv', index=False)       




#%%









#%%

###############  old code
# ============================================================================
# ========   means per state and mean differenece between states:   ==========
# ============================================================================
""" calculate the differeneces between conditions (rest,stress,recovery) 
 reactivity and recovery for each sub and each parameter: """
 
### original code is in the older file 'data_orgenize_August_2022.py'

# ============================================================================
# ================   a shorter code to create the means table     ============
# ============================================================================
# select only p1 data
def fun_means_long(df_mean_melt):
    df1 = df_mean_melt.loc[df_mean_melt['part']=='p1'].reset_index(drop=True) 
    df = df1.rename(columns={'value':'p1_baseline'}).drop(columns={'part'})
    # add p2,p3 data 
    df['p2_stress'] = df_mean_melt.loc[df_mean_melt['part']=='p2', 'value'].reset_index(drop=True) 
    df['p3_recovery'] = df_mean_melt.loc[df_mean_melt['part']=='p3', 'value'].reset_index(drop=True)
    
    # add the deltas between states
    df['D1'] = df['p2_stress']-df['p1_baseline']
    df['D2'] = df['p2_stress']-df['p3_recovery'] 
    
    df_mean_long = df
    
    return df_mean_long


df_mean_long = fun_means_long(df_mean_melt)
df_mean_long.to_csv(f'{data_path}/Results/df_mean_long.csv', index=False)       
# df_mean_long_norm = fun_means_long(df_mean_melt_norm)
# df_mean_long_norm.to_csv(f'{data_path}/Results/df_mean_long_norm.csv', index=False)       


#%%

# ============================================================================
# calculate the mean value for each variable across subjects
# ============================================================================
def fun_mean_p_v(df_mean_long):
    df_mean_per_var = df_mean_long.groupby(['variable']).mean().drop(columns = ['sub'])
    df_mean_per_var = df_mean_long.groupby(['variable']).agg([np.mean, np.std]).drop(columns = ['sub'])
    
    # add efffect size:    eff1 = mean_reactivity/std_p2 ,     eff2 = mean_recovery/std_p3
    df_mean_per_var['eff_size_D1'] = df_mean_per_var.loc[:,('D1', 'mean')]/df_mean_per_var.loc[:,('p1_baseline', 'std')]
    df_mean_per_var['eff_size_D2'] = df_mean_per_var.loc[:,('D2', 'mean')]/df_mean_per_var.loc[:,('p3_recovery', 'std')]
    return df_mean_per_var

df_mean_per_var = fun_mean_p_v(df_mean_long)
df_mean_per_var.to_csv(f'{data_path}/Results/df_mean_per_var.csv', index=True)       

#%%

############################      remove p1    ############################
# df_mean_new = df_mean_long.drop(['p1', 'reactivity'], inplace=False, axis=1)
# df_mean_new.to_csv('C:/users/hadar/OneDrive/Documents/MASTER/2-RESEARCH/experiment_validation/Results/df_mean_no_p1.csv', index=True)       


#%%
# create means table of normalized data >>> in the older file 'data_orgenize_August_2022.py'
















#%%
# ###########################################################################
# ###########################################################################
# ###########################################################################
# ###########################################################################
# ======================          plots:          ===========================
# ###########################################################################


# 
#          plots of the mean difference: the reactivity and recovery          
# ===========================================================================

######### plot with p1: (each variable seperatly)
data = df_mean_long
for metric in selected_metrics:
    reactivity = data[data['variable'] ==metric].reactivity
    recovery = data[data['variable'] ==metric].recovery
    
    fig, ax = plt.subplots(figsize=(4, 3))
    plt.bar(1, np.mean(reactivity), edgecolor = 'r', facecolor = 'r', 
            label = 'Reactivity = Rest to Stress')
    plt.bar(2, np.mean(recovery), edgecolor = 'b', facecolor = 'b', 
            label = 'Recovery = Stress to Relax')
    plt.scatter([1 for _ in range(len(reactivity))],reactivity, color = 'r', 
                alpha = 0.25)
    plt.scatter([2 for _ in range(len(recovery))],recovery, color = 'b', 
                alpha = 0.25)
    plt.hlines(0, 0.5, 2.5, 'k', '--')
    # plt.legend(bbox_to_anchor = (0.75, -0.1))
    plt.title(f'{metric} - mean delta', fontweight='bold',size=14)
    plt.xticks(np.arange(1,3), ['Reactivity', 'Recovery'], color='black', 
               fontweight='bold', fontsize='14', horizontalalignment='center')
           
    plt.show()
    
    # plt.hist(recovery)
    # plt.title(f'recovery - {metric} - Histogram')
    # plt.show()

#%%

###  bar plot of the recovery only (without p1 and reactivity): (bar plot for all variables together)
data =  df_mean_long_norm  # it has to bo normalized sinceit put all variables on a single scale

###  create the expected change direction under stress for each variable 
###   (increase ('True') or not ('False'))
is_increase = [True,True, False,False,False,False,False, True, True,True,True, False,False,False, True,True,True,False,True]
exp_changepd = pd.DataFrame({'variable': all_hrv_features[4::], 'increase_at_stress': is_increase})
 

for i,metric in enumerate(selected_metrics):
    recovery = data[data['variable'] ==metric].recovery
    
    # set colors depend on the change direction and if it is as expected or not
    c = 'blue' if exp_changepd.loc[exp_changepd['variable'] ==metric, 'increase_at_stress'].bool()== True else 'lime'
    c = 'red' if c == 'blue' and np.mean(recovery) >= 0 else c
    c = 'red' if c == 'lime' and np.mean(recovery) <= 0 else c
   
    plt.bar(i, np.mean(recovery), edgecolor = 'grey', facecolor=c, label = f'{metric}')
    plt.scatter([i for _ in range(len(recovery))], recovery, color=c, alpha = 0.25)

plt.hlines(0, -1, 8, linewidth=1, color='black', linestyles='--')
plt.legend(bbox_to_anchor = (1.0, 1), fontsize='10')
plt.title('Recovery - mean difference between Stress to Relax', size=14)
plt.xticks(np.arange(len(selected_metrics)), selected_metrics, color='black', rotation=60, 
           fontweight='bold', fontsize='10', horizontalalignment='right')
plt.show()
    
  
    # colors = ['red', '#630A28', 'darkviolet', 'violet', 
    #           'blue','cyan', 'darkgreen', 'lime',
    #           '#B190FC' ,'magenta', 'orange','yellow']
    # '#41DFFB', colors names in this link:
    # https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib    
    


#%%
                       
# ============================================================================ 
# ====================  scatterplots - for each variable   ===================
# ============================================================================ 

#metric =metrics[0]  
# my_dat = df_norm_all  
# norm = ""
task_dur = 15

def plot_scatter_for_var(my_dat,metric, norm) :   
    # calculate the variable average for all subs in every time-point to show on the scattterplot:
    data = pd.DataFrame(data = my_dat, columns = ['sample_num',metric])
    mean_data_subs = data.groupby('sample_num').mean()
    mean_data_subs.rename(columns={metric: f'mean_{metric}'}, inplace = True)
    # plot the scatter with the mean:
    fig, ax = plt.subplots(figsize=(10, 6))              
    sns.scatterplot(x="sample_num", y=metric,  hue='sub',
                    data=my_dat, palette="bright")
    sns.lineplot(x = "sample_num", y=f'mean_{metric}', data=mean_data_subs, 
                 color='black', markers=True, label = "average on subs")
    
    plt.axvline(x=(task_dur/samp_duration)+0.5, color='r', linestyle='--') # add lines to define tasks borders
    plt.axvline(x=2*(task_dur/samp_duration)+0.5, color='r', linestyle='--')
    
    plt.xticks(range(int(45/samp_duration+1)))
    plt.xlabel(f'sample num ({samp_duration} minutes per sample)')
    
    plt.legend(bbox_to_anchor=(1.01,1.1)).remove()


    plt.title(f'{metric} {norm}')

    plt.show()
    # plt.savefig(f'{res_path}/subjects_{sub[0]}-{sub[-1]}_{metric}_in_time.png')     
    return
# =========================================================================== 



"""
selected_metrics = ['garmin_stress',
 'Mean_HR_bpm',
 'SD2_SD1_ratio',
 'LF_HF_ratio_AR',
 'RMSSD_ms',
 'pNN50',
 'HFpow_AR_ms2',
 'HFpow_AR_nu']
    """
# p_single_metric = plot_scatter_for_var(df_all, 'RMSSD_ms', '')

#%%
p = [plot_scatter_for_var(df_all, x, '') for x in selected_metrics]
# p_n = [plot_scatter_for_var(df_norm_all, x,' - normalized') for x in selected_metrics] 

#%%
# =========================================================================== 
# scatterplot for each sub:
# ============================================================================ 
   
def plot_sub_data(my_dat,metric, norm):    
    for s in sub_lst:
        my_dat = df_all.loc[df_all['sub']==s]
        
        # calculate the average of each part:       
        mean_p = my_dat.groupby(['part']).mean()[metric]
        m = [mean_p[0]]*10+[mean_p[1]]*10+[mean_p[2]]*10        
        m_df = pd.DataFrame({'sample_num':my_dat['sample_num'],'mean_p':m})
        
        # plot the scatter with the mean:
        fig, ax = plt.subplots(figsize=(10, 6))              
        sns.scatterplot(x="sample_num", y=metric,hue='part',data=my_dat, palette="bright")
                        
        sns.lineplot(x ='sample_num' , y='mean_p', data=m_df, color='black', linewidth=0.7, markers=True, label = "part mean")
        plt.axvline(x=(30/samp_duration)+0.5, color='r', linestyle='--') # add lines to define tasks borders
        plt.axvline(x=2*(30/samp_duration)+0.5, color='r', linestyle='--')
        
        plt.xticks(range(int(90/samp_duration+1)))
        plt.xlabel(f'sample num ({samp_duration} minutes per sample)')       
        plt.legend(bbox_to_anchor=(1.01,1.1))        
        plt.title(f'{metric} - sub {s} {norm}')       
        plt.show()
    return
# =========================================================================== 

# p = [plot_sub_data(df_all, x, '') for x in selected_metrics]
# p_n = [plot_sub_data(df_norm_all, x,' - normalized') for x in selected_metrics] 


#%%


def multiple_boxplots(my_data, vars_names):
    df_melt = my_data.melt(id_vars='part',  value_vars=[i for i in vars_names], var_name='columns')                                  

    a = sns.catplot(data = df_melt, x = 'part', y = 'value', 
                       kind = 'violin', # type of plot
                       col = 'columns', inner="quart", linewidth=1,
                       # custom order of boxplots:
                       col_order = [i for i in vars_names]).set_titles('{col_name}') # remove 'column = ' part of title
    plt.show()
    return

multiple_boxplots(df_norm_all, ['garmin_stress', 'Mean_HR_bpm'])
multiple_boxplots(df_norm_all, ['stress_index', 'PNS_index', 'SNS_index'])

multiple_boxplots(df_norm_all, ['RMSSD_ms','pNN50', 'SD2_SD1_ratio'])
multiple_boxplots(df_norm_all, ['HFpow_AR_nu', 'HFpow_AR_ms2', 'LF_HF_ratio_AR'])

# multiple_boxplots(df_norm_all, ['RMSSD_ms', 'SDNN_ms', 'pNN50'])
# multiple_boxplots(df_norm_all, ['RR_tri_index', 'SD2_SD1_ratio'])
# multiple_boxplots(df_norm_all, ['LF_HF_ratio_AR','LF_HF_ratio_FFT'])
# multiple_boxplots(df_norm_all, ['LFpow_AR_ms2', 'LFpow_AR_log', 'LFpow_AR_nu'])
# multiple_boxplots(df_norm_all, ['HFpow_AR_ms2', 'HFpow_AR_log', 'HFpow_AR_nu'])


#%%

# ===========================================================================
# ===========================  Histogram  ===================================
# ===========================================================================
"""
for f in selected_metrics:
    plt.hist(df_all[f])
    plt.title(f'{f} - Histogram')
    plt.show()
    
    plt.hist(df_norm_all[f])
    plt.title(f'{f} - normalized - Histogram')
    plt.show()
"""

#%%

### Histogram of the mean values of each part

"""
for f in selected_metrics:
    df = df_mean_long.loc[df_mean_long['variable']==f]
    plt.hist(df.p1_rest)
    plt.title(f'Rest - Mean {f} - Histogram')
    plt.show()
    plt.hist(df.p2_stress)
    plt.title(f'Stress - Mean {f} - Histogram')
    plt.show()
    plt.hist(df.p3_relax)
    plt.title(f'Relax - Mean {f} - Histogram')
    plt.show()
"""

#%%
# Histogram - all parts on one fig
for f in selected_metrics:
    df = df_mean_long.loc[df_mean_long['variable']==f] 
    plt.figure(figsize=(6,18))
    
    plt.subplot(5,1,1)
    plt.hist(df.p1_rest)
    plt.title(f'Rest - Mean {f} - Histogram')
    
    plt.subplot(5,1,2)
    plt.hist(df.p2_stress)
    plt.title(f'Stress - Mean {f} - Histogram')
        
    plt.subplot(5,1,3)
    plt.hist(df.p3_relax)
    plt.title(f'Relax - Mean {f} - Histogram')
    
    plt.subplot(5,1,4)
    plt.hist(df.reactivity)
    plt.title(f'Reactivity - Mean {f} - Histogram')
    
    plt.subplot(5,1,5)
    plt.hist(df.recovery)
    plt.title(f'Recovery - Mean {f} - Histogram')
                
    plt.show()
    
#%%

#                          plot density function      
# ===========================================================================
data =  df_mean_long
# df_mean_long.columns

# a = data['p1'][data['metric'] == 'garmin_stress']
# b = data['p2'][data['metric'] == 'garmin_stress']
# c = data['p3'][data['metric'] == 'garmin_stress']
for f in selected_metrics:
    y = data[['p1_rest', 'p2_stress', 'p3_relax']][data['variable'] == f].reset_index()
    y = y.drop(columns = ['index'])
    sns.kdeplot(data = y,  shade=True)
    plt.title(f'{f} - Density')
    plt.show()
    


               
               
              


