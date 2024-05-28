# -*- coding: utf-8 -*-
"""
Created on Mon Sep 19 10:29:22 2022

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


#%%

# import edited data that created in 'Dataframin_and_visualization.py'

exp = 2

general_path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2'
data_path = f'{general_path}/Data_Analysis'
res_path = f'{data_path}/Results'
df_all = pd.read_csv(f'{res_path}/df_all.csv')
df_all.shape
print(df_all.head(3))
df_all.sample(10)

sub_lst = np.unique(df_all['sub'])

selected_metrics = ['Mean_HR_bpm','RMSSD_ms','SD2_SD1_ratio','LF_HF_ratio_AR','HFpow_AR_nu', 'garmin_stress']                 


subjects_full = pd.read_csv(f'{res_path}/subjects_full.csv')

# df_mean_long = pd.read_csv(f'{res_path}/df_mean_long.csv')
# df_mean_per_var = pd.read_csv(f'{res_path}/df_mean_per_var.csv', index_col=0)
# df_mean_per_var.index.name = 'variable'

# features_focus = ['garmin_stress','Mean_HR_bpm', 'RMSSD_ms', 
#                   'HFpow_AR_ms2','HFpow_AR_nu','SD2_SD1_ratio','LF_HF_ratio_AR']                  
                   
#%%



#   ****** main experiment data   *****


df = pd.read_csv(f'{res_path}/df_mean_wide_edit.csv')

 


#%%

# ###########################################################################
# ===============        statistical analysis:          =====================
# ###########################################################################



#%%


############     New code to calculate the t-test and create results tables

  

#%%

from scipy.stats import ttest_rel
import pandas as pd
from scipy import stats


# # Variables of interest
# variables = ['Mean_HR_bpm','RMSSD_ms','SD2_SD1_ratio','LF_HF_ratio_AR', 'HFpow_AR_nu', 'garmin_stress']

variables = selected_metrics + ['reported_stress']

df = df[['subject', 'condition'] + selected_metrics + ['reported_stress']]

conditions = ['Baseline', 'Stress', 'Recovery']


variable_mapping = {
    'garmin_stress': 'Garmin Stress',
    'Mean_HR_bpm': 'Mean HR (bpm)',
    'SD2_SD1_ratio': 'SD2/SD1 Ratio',
    'LF_HF_ratio_AR': 'LF/HF Ratio',
    'RMSSD_ms': 'RMSSD (ms)',
    'HFpow_AR_nu': 'HF Power (nu)'
}

#%%
######################################

def descriptive_stat_table(df): 
    import scipy.stats as stats
    
    
    # Initialize an empty DataFrame with MultiIndex for columns for mean and SD
    # mean_sd_columns = pd.MultiIndex.from_product([['Baseline', 'Stress', 'Recovery'], ['Mean', 'SD', 'CV']], names=['Condition', 'Statistic'])
    mean_sd_columns = pd.MultiIndex.from_product([['Baseline', 'Stress', 'Recovery'], ['Mean', 'SD']], names=['Condition', 'Statistic'])

    mean_sd_table = pd.DataFrame(index=variables, columns=mean_sd_columns)
    
    # Compute mean and SD for each variable in each condition
    for var in variables:
        for condition in conditions:
            data_condition = df[df['condition'] == condition][var]
            mean_sd_table.loc[var, (condition, 'Mean')] = round(float(data_condition.mean()),2)
            mean_sd_table.loc[var, (condition, 'SD')] = round(float(data_condition.std()),2)
            
            # # calculate CV (coefficient of variance)    
            # cv = data_condition.std() / data_condition.mean() * 100
            # # Format CV with "%" sign
            # cv_formatted = f"{cv:.2f}%"  # Rounds CV to 2 decimal places and adds "%" sign
            # mean_sd_table.loc[var, (condition, 'CV')] = cv_formatted
    
    # Convert numbers to strings and add parentheses
    mean_sd_table['Baseline', 'SD'] = '(' + mean_sd_table['Baseline', 'SD'].astype(str) + ')'
    mean_sd_table['Stress', 'SD'] = '(' + mean_sd_table['Stress', 'SD'].astype(str) + ')'
    mean_sd_table['Recovery', 'SD'] = '(' + mean_sd_table['Recovery', 'SD'].astype(str) + ')'
    
    # mean_sd_table['Recovery', 'SD'] = mean_sd_table['Recovery', 'SD'].apply(lambda x: '({})'.format(abs(x)))
    
    print(mean_sd_table)
    # mean_sd_table = mean_sd_table.astype(float).round(decimals=2)
    # mean_sd_table.rename(columns=variable_mapping, inplace=True, errors='raise')
    return(mean_sd_table)

#%%  
mean_sd_table = descriptive_stat_table(df)
  
# Export
mean_sd_table.to_csv(f'{res_path}/df_mean_sd_conditions_{exp}.csv', index=True, float_format='%.0f')       


 #%%  

########    paired t-test
########################################

def paired_test(df):
    
    # Initialize an empty DataFrame with MultiIndex for columns for t-test results
    t_test_columns = pd.MultiIndex.from_product([['Stress vs Baseline', 'Recovery vs Baseline', 'Recovery vs Stress'],
                                                 ['t', 'p-value', 'sig']],
                                                 names=['Comparison', 'Statistic'])
    t_test_table = pd.DataFrame(index=variables, columns=t_test_columns)
    
    # Perform paired t-tests
    for var in variables:
        for i in range(len(conditions)):
            for j in range(i+1, len(conditions)):
                cond1 = conditions[i]
                cond2 = conditions[j]
                t_stat, p_val = stats.ttest_rel(
                    df[df['condition'] == cond1][var],
                    df[df['condition'] == cond2][var]
                )
                
                sig = '*' if p_val < 0.0125 else ''
                comparison = f"{conditions[j].capitalize()} vs {conditions[i].capitalize()}"
                t_test_table.loc[var, (comparison, 't')] = t_stat.astype(float).round(2)
                t_test_table.loc[var, (comparison, 'p-value')] = p_val.astype(float)
                t_test_table.loc[var, (comparison, 'sig')] = sig
    
    # # Select only the float columns
    # float_columns = t_test_table.select_dtypes(include=['float'])
    
    # # Round the float columns to reduce decimal places
    # t_test_table[float_columns.columns] = float_columns.round(decimals=2)
    
    
    
    
    # Original variable names and their desired names mapping
    
    # Rename the columns in the result table using the mapping
    t_test_table.rename(columns=variable_mapping, inplace=True)
    return t_test_table

#%%

t_test_table = paired_test(df)
t_test_table.to_csv(f'{res_path}/df_paired_ttest_conditions_{exp}.csv', index=True)       


#%%
# Concatenate mean_sd_table and t_test_table horizontally
result_table = pd.concat([mean_sd_table, t_test_table], axis=1)


#%%

# *****     pilot data   *****   copied to a seperate file in the pilot folder

"""
exp = 1
data_path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/experiment_validation'
res_path = f'{data_path}/Results'
# df_all = pd.read_csv(f'{res_path}/df_all.csv')


df_mean_wide = pd.read_csv(f'{res_path}/df_mean_wide.csv')

df = df_mean_wide.rename(columns={'sub': 'subject', "part": "condition"})
# replace the parts names to the condition names
df['condition'] = df['condition'].replace({'p1': 'Baseline', 'p2': 'Stress', 'p3': 'Recovery'})
                   
df = df[['subject', 'condition'] + selected_metrics]    


nan_rows = df[df['garmin_stress'].isnull()] 
nan_subjects = nan_rows['subject'].tolist()
print(nan_subjects)

# df = df.dropna() 

# remove invalid subjects (had missing stress points in a whole task)
mask = df['subject'].isin(nan_subjects)
df = df[~mask]

pilot_valid_sub = df['subject'].unique()

df.to_csv(f'{res_path}/df_mean_wide_edited_pilot.csv', index=True)       


#%%  
mean_sd_table = descriptive_stat_table(df)
  
# Export
mean_sd_table.to_csv(f'{res_path}/df_mean_sd_conditions_{exp}.csv', index=True, float_format='%.0f')       

#%%

t_test_table = paired_test(df)
t_test_table.to_csv(f'{res_path}/df_paired_ttest_conditions_{exp}.csv', index=True)       


#%%
# Concatenate mean_sd_table and t_test_table horizontally
result_table = pd.concat([mean_sd_table, t_test_table], axis=1)


"""

#%%
#############################################################################
#############################################################################
#############################################################################




#%%
##############               Old version:       ##############
    
## paired t-test to the stress VS relax VS baseline

df = df_mean_long.dropna()

import scipy.stats as stats

features = ['garmin_stress','Mean_HR_bpm', 'RMSSD_ms', 
                  'HFpow_AR_ms2','HFpow_AR_nu','SD2_SD1_ratio','LF_HF_ratio_AR',
                  'stress_index', 'PNS_index','SNS_index']  

dic = {'parameter':[], 'mean_baseline':[], 'mean_stress':[], 'mean_recovery':[], 
       'Tstat_paired_stress-VS-base':[], 'Pval_paired_stress-VS-base':[],
       'Tstat_paired_recovery-VS-base':[], 'Pval_paired_recovery-VS-base':[],
       'Tstat_paired_stress-VS-recovery':[], 'Pval_paired_stress-VS-recovery':[]}

for i, feature in enumerate(features):
    dic['parameter'].append(feature)
    
    
    # extracting data of a specific parameter to be tested
    s1 = df['p1_baseline'][df['variable']==feature]
    s2 = df['p2_stress'][df['variable']==feature]
    s3 = df['p3_recovery'][df['variable']==feature]
    
    mean1 = s1.mean(); mean2 = s2.mean() ; mean3 = s3.mean() 
    
    # mean_diff
    # adding the results to our dictionary
    dic['mean_baseline'].append(mean1)
    dic['mean_stress'].append(mean2)
    dic['mean_recovery'].append(mean3)
    
    # Performing the paired sample t-test
    tStat, pValue =  stats.ttest_rel(s1, s2) 
    # adding the results to our dictionary
    dic['Tstat_paired_stress-VS-base'].append(tStat)
    dic['Pval_paired_stress-VS-base'].append(pValue)
    
    # Performing the paired sample t-test
    tStat, pValue =  stats.ttest_rel(s1, s3) 
    # adding the results to our dictionary
    dic['Tstat_paired_recovery-VS-base'].append(tStat)
    dic['Pval_paired_recovery-VS-base'].append(pValue)
    
    # Performing the paired sample t-test
    tStat, pValue =  stats.ttest_rel(s2, s3) 
    # adding the results to our dictionary
    dic['Tstat_paired_stress-VS-recovery'].append(tStat)
    dic['Pval_paired_stress-VS-recovery'].append(pValue)
    
    


# df_paired_ttest = pd.DataFrame(dic)[['parameter', 'mean_baseline', 'mean_stress', 
#                                      'mean_recovery', 
#        'Tstat_paired_stress-VS-base', 'Pval_paired_stress-V-Sbase',
#        'Tstat_paired_recovery-VS-base', 'Pval_paired_recovery-VS-base',
#        'Tstat_paired_stress-VS-recovery', 'Pval_paired_stress-VS-recovery']]

df_paired_ttest = pd.DataFrame(dic)[['parameter', 'mean_baseline', 'mean_stress', 
                                      'mean_recovery', 
        'Pval_paired_stress-VS-base',
         'Pval_paired_recovery-VS-base',
         'Pval_paired_stress-VS-recovery']]

display(df_paired_ttest)
# df_paired_ttest = df_paired_ttest.astype(float)
df_paired_ttest.iloc[:, df_paired_ttest.columns != 'parameter'] = df_paired_ttest.iloc[:, df_paired_ttest.columns != 'parameter'].applymap(lambda x: float(x))
    
df_paired_ttest.to_csv(f'{res_path}/df_paired_ttest.csv', index=True)       



#%%

#############      power analysis



from statsmodels.stats.power import tt_solve_power

# mean_diff = df_paired_ttest['mean_stress']-df_paired_ttest['mean_recovery']

df_ttest_pwr = pd.DataFrame()
# columns = ["parameter", "sample_size"]

for i, feature in enumerate(features):
    dic['parameter'].append(feature)
    
    
    # extracting data of a specific parameter to be tested
    s1 = df['p1_baseline'][df['variable']==feature]
    s2 = df['p2_stress'][df['variable']==feature]
    s3 = df['p3_recovery'][df['variable']==feature]
    
    mean1 = s1.mean(); mean2 = s2.mean() ; mean3 = s3.mean() 
    
    mean_diff = (s2-s3).mean()
    sd_diff = (s2-s3).std()
        
    std_effect_size = mean_diff / sd_diff

    n = tt_solve_power(effect_size=std_effect_size, alpha=0.05/4, power=0.8, alternative='two-sided')
    
    print('{}: Sample size: {:.3f}'.format(feature,n))
    
    list = [[feature,n]]
    df_ttest_pwr = df_ttest_pwr.append(list)

df_ttest_pwr.columns = ["parameter", "sample_size"]
display(df_ttest_pwr)

df_ttest_pwr.to_csv(f'{res_path}/df_ttest_pwr.csv', index=True)       

#%%

#######   calculate relative values of stress and recovery to baseline value

##   add transformed values by reducing and dividing by mean baseline
df_mean_long['stress_rel'] = (df_mean_long['p2_stress']-df_mean_long['p1_baseline'])/df_mean_long['p1_baseline']
df_mean_long['recovery_rel'] = (df_mean_long['p3_recovery']-df_mean_long['p1_baseline'])/df_mean_long['p1_baseline']

df = df_mean_long.dropna()

#%%


## paired t-test to the relative differences

import scipy.stats as stats

features = ['garmin_stress','Mean_HR_bpm', 'RMSSD_ms', 
                  'HFpow_AR_ms2','HFpow_AR_nu','SD2_SD1_ratio','LF_HF_ratio_AR',
                  'stress_index', 'PNS_index','SNS_index']  

dic = {'parameter':[],'mean_stress_rel':[], 'mean_recovery_rel':[], 
       'Tstat_paired':[], 'Pval_paired':[]}

for i, feature in enumerate(features):
    dic['parameter'].append(feature)
    
    # # extracting data of a specific parameter to be tested
    s1 = df['stress_rel'][df['variable']==feature]
    s2 = df['recovery_rel'][df['variable']==feature]
    

    
    mean1 = s1.mean(); mean2 = s2.mean() 
    
    # Performing the paired sample t-test
    tStat, pValue =  stats.ttest_rel(s1, s2)
    
    # adding the results to our dictionary
    dic['mean_stress_rel'].append(mean1)
    dic['mean_recovery_rel'].append(mean2)
    dic['Tstat_paired'].append(tStat)
    dic['Pval_paired'].append(pValue)
    
    


df_paired_ttest_rel = pd.DataFrame(dic)[['parameter', 
                                         'mean_stress_rel', 'mean_recovery_rel',
                                         'Tstat_paired', 'Pval_paired']]

display(df_paired_ttest_rel)    
df_paired_ttest_rel.to_csv(f'{res_path}/df_paired_ttest_rel.csv', index=True)       





#%%


# ======================   Wilcoxon Signed-Rank Test   ======================
# compare mean stress to mean relax. (each subject has 3 means >> baseline (rest), stress, recovery >> X 60 subjects)
# https://www.statology.org/wilcoxon-signed-rank-test-python/

print('Wilcoxon Signed-Rank Test:')


dic = {'parameter':[],'mean_D1':[], 'stat_D1':[], 'Pval_D1':[], 
       'mean_D2':[], 'stat_D2':[], 'Pval_D2':[]}

for i, feature in enumerate(features_focus):
    group1 = df_mean_long['p1_baseline'][df_mean_long['variable']==feature]
    group2 = df_mean_long['p2_stress'][df_mean_long['variable']==feature]
    group3 = df_mean_long['p3_recovery'][df_mean_long['variable']==feature]
    
    dic['parameter'].append(feature)  
    
    # Reactivity
    Stat, pValue = stats.wilcoxon(group1, group2)
    print(f'Reactivity - parameter: {feature}, P-Value:{pValue} Statistic:{Stat}') #print the P-Value and the T-Statistic
     
    dic['mean_D1'].append((group2-group1).mean())
    dic['stat_D1'].append(Stat)
    dic['Pval_D1'].append(pValue)  
    
    # Recovery
    Stat, pValue = stats.wilcoxon(group2, group3)
    print(f'Recovery - parameter: {feature}, P-Value:{pValue} Statistic:{Stat}') #print the P-Value and the T-Statistic

    dic['mean_D2'].append((group2-group3).mean())
    dic['stat_D2'].append(Stat)
    dic['Pval_D2'].append(pValue)

df_Wilcoxon_test = pd.DataFrame(dic)
df_Wilcoxon_test = df_Wilcoxon_test.drop(['stat_D1', 'stat_D2'], axis =1)
df_Wilcoxon_test.to_csv(f'{res_path}/df_Wilcoxon_test.csv', index=True)       
display(df_Wilcoxon_test)




#%%

# =================   1 sample t-test (compare diff to zero)   =================

# data_react = df_mean_long['d_reactivity'][df_mean_long['feature']=='RMSSD_ms']
# data_recover = df_mean_long['d_recovery'][df_mean_long['feature']=='RMSSD_ms'] 

# # perform one sample t-test
# tStat, pValue =  scipy.stats.ttest_1samp(a=data_react, popmean=0, axis=0)
# print("d_reactivity:   P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) #print the P-Value and the T-Statistic

# tStat, pValue =  scipy.stats.ttest_1samp(a=data_recover, popmean=0, axis=0)
# print("d_recovery:   P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) #print the P-Value and the T-Statistic


dic = {'parameter':[],'mean_D1':[], 'Tstat_D1':[], 'Pval_D1':[], 
       'mean_D2':[], 'Tstat_D2':[], 'Pval_D2':[]}

for i, feature in enumerate(features_focus):
    dic['parameter'].append(feature)
    
    data_react = df_mean_long['D1'][df_mean_long['variable']==feature]    
    tStat, pValue =  scipy.stats.ttest_1samp(a=data_react, popmean=0, axis=0)
    print("reactivity:   P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) #print the P-Value and the T-Statistic

    dic['mean_D1'].append(data_react.mean())
    dic['Tstat_D1'].append(tStat)
    dic['Pval_D1'].append(pValue)

    data_recover = df_mean_long['D2'][df_mean_long['variable']==feature]
    tStat, pValue =  scipy.stats.ttest_1samp(a=data_recover, popmean=0, axis=0)
    print("recovery:   P-Value:{0} T-Statistic:{1}".format(pValue,tStat)) #print the P-Value and the T-Statistic

    dic['mean_D2'].append(data_recover.mean())
    dic['Tstat_D2'].append(tStat)
    dic['Pval_D2'].append(pValue)

df_1samp_ttest_deltas = pd.DataFrame(dic)[['parameter', 'mean_D1', 'Pval_D1',
                              'mean_D2', 'Pval_D2']]
    
df_1samp_ttest_deltas.to_csv(f'{res_path}/df_1samp_ttest_deltas.csv', index=True)       

#%%


# ======= create a table with mean values and wilcoxon test results =========

data_means = df_mean_per_var.copy()
# data_means.reset_index(inplace=True)
# data_means.columns = data_means.columns.map('|'.join).str.strip('|')

df_sum = pd.DataFrame()

for f in features_focus:   
    df1 = pd.DataFrame(data_means.loc[data_means.reset_index()['variable'] == f,:]).reset_index(drop=True)
    df2 = pd.DataFrame(df_Wilcoxon_test.loc[df_Wilcoxon_test['parameter'] == f,('Pval_reactivity', 'Pval_recovery')]).reset_index(drop=True)
    data_var = pd.concat([df1, df2], axis=1, levels=2)
    df_sum = pd.concat([df_sum, data_var])

df_sum.reset_index(inplace=True, drop=True)
df_sum = df_sum.rename(columns={'Pval_reactivity':'Pval_Wilcoxon_reactivity', 'Pval_recovery': 'Pval_Wilcoxon_recovery'})
df_sum.to_csv(f'{res_path}/df_sum.csv', index=False)       

display(df_sum)










#%%



### look at the interactions of the hrv with subject info (gender, pss, quiz score..)
###########################################################################  

subjects = pd.read_csv(f'{res_path}/subjects.csv')
subjects = subjects[['sub','age','gender','bmi_Kg/m^2','final_quiz_score',
                     'PSS-14_score','reported_stress_1', 'reported_stress_2', 'reported_stress_3','missing_stress_points']]

data_hrv = df_mean_long.loc[df_mean_long['variable']=='RMSSD_ms'].reset_index(drop=True)
data_sub = subjects.reset_index(drop=True)

#%%

#### create categorical variable for sdnn baseline level(threshold=50 ms)

df_mean_long['variable'].unique()
data_sdnn = df_mean_long.loc[df_mean_long['variable']=='SDNN_ms'].reset_index(drop=True)

for sub in data_sdnn['sub']:   
    if (int(data_sdnn.loc[data_sdnn['sub']==sub, 'p3_recovery'])<50): 
        data_hrv.loc[data_sub['sub']==sub, 'baseline_sdnn']='low' 
    else:
        data_hrv.loc[data_sub['sub']==sub, 'baseline_sdnn']='high' 
  
#%%

######### merge hrv data with subject data to one dataframe

mydata = pd.concat([data_hrv, data_sub.drop(columns = {'sub'})],axis=1)
mydata.columns

import seaborn
seaborn.lmplot(x="PSS-14_score",y="final_quiz_score",hue="gender", data=mydata)

mean_gender = mydata.groupby("gender").mean()

seaborn.lmplot(x="p3_recovery",y="recovery",hue="gender", data=mydata)


#%%

seaborn.lmplot(x="p3_recovery",y="recovery",hue="baseline_sdnn", data=mydata)

#%%

def plotit(x,y,dat):
    # Fit linear regression via least squares with numpy.polyfit
    # deg=1 means linear fit (i.e. polynomial of degree 1)
    # obtain a (slope) and b(intercept) of linear regression line
    a, b = np.polyfit(x, y, 1)

    plt.plot(x, b + a * x, color="k", lw=1.5)
    
    # plt.scatter(x = rmssd['rmssd_baseline'], y = rmssd['rmssd_stress'],  alpha=0.5)
    reg_form =f'y='+'%.2f' % a +'x+'+'%.2f' % b
    sns.scatterplot(x, y, data=dat, hue='baseline_sdnn', label=reg_form)


# x = mydata['p3_relax'] ; y = mydata['recovery']
# plotit(x,y, mydata)

######### plot seperatly groups of low and high sdnn at baseline
## plot 1:
x = mydata[mydata['baseline_sdnn']=='low'].p3_relax
y = mydata['recovery'][mydata['baseline_sdnn']=='low']

plt.subplot(1, 2, 1)
plotit(x,y, mydata)
## plot 2:
x = mydata[mydata['baseline_sdnn']=='high'].p3_relax
y = mydata['recovery'][mydata['baseline_sdnn']=='high']

plt.subplot(1, 2, 2)
plotit(x,y, mydata)

plt.show()


