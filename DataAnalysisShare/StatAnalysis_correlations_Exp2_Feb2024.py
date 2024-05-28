# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 18:40:55 2022

@author: hadar

Calculation of pearson correlation in two ways: 
    1 - Within subjects and average of r across subjects.
    2 - Between subjects - pearson r of mean values of each condition  
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

# import edited data that created in 'Dataframin_and_visualization_Sep2022
general_path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2'
data_path = f'{general_path}/Data_Analysis'
res_path = f'{data_path}/Results'
df_all = pd.read_csv(f'{res_path}/df_all.csv')
df_all.shape
print(df_all.head(3))
df_all.sample(10)

df_mean_long = pd.read_csv(f'{res_path}/df_mean_long.csv')
df_means = pd.read_csv(f'{res_path}/df_means.csv')

selected_metrics = ['Mean_HR_bpm','RMSSD_ms','SD2_SD1_ratio','LF_HF_ratio_AR','HFpow_AR_nu', 'garmin_stress']                 


# sub_error = [319,332,333]
sub_lst = np.unique(df_all['sub'])

# Mapping metric names
variable_mapping = {
    'garmin_stress': 'Garmin Stress Score',
    'Mean_HR_bpm': 'Mean HR',
    'SD2_SD1_ratio': 'SD2/SD1',
    'LF_HF_ratio_AR': 'LF/HF',
    'RMSSD_ms': 'RMSSD',
    'HFpow_AR_nu': 'HF power[nu]',
    'reported_stress': 'Reported Stress'
}

#%%

################  pearsons correlation:    - within subject correlation
# =========================================================================== 

# calculate perason cor between Garmin sterss to the parameters for each subject seperatly

data = df_all.dropna() 

flst = selected_metrics.copy()
flst.remove('garmin_stress')


sub_lst = [int(i) for i in sub_lst]
first_sub = sub_lst[0]

for s in sub_lst:
    df_cor_sig = pd.DataFrame(columns=['parameter', 'pearson_r', 'p-value','sig'])
    
    data_sub = data.loc[data['sub'] == s]
    a = data_sub['garmin_stress']
    n = len(a)
    if (n<2):
        continue
    
    for f in flst:
        b = data_sub[f] 
        r, pval = pearsonr(a, b)
        # print(r,", " ,pval)
        sig = '*' if pval < 0.0125 else ''
        df_cor_sig = df_cor_sig.append({'parameter':f, 'pearson_r':r, 'p-value':pval, 'sig':sig}, ignore_index=True)
        
        # check the t statistic:
        # t = r*math.sqrt(n-2)/math.sqrt(1-r**2) 
        # display(t)
    
    display(s, df_cor_sig) 
    
    if s == first_sub:
        cor_all_subs = pd.DataFrame([df_cor_sig.pearson_r])
        cor_all_subs.columns = flst
        cor_all_subs.insert(0, 'sub', s)
    else:
        new_sub = pd.DataFrame([df_cor_sig.pearson_r])
        new_sub.columns = flst
        new_sub.insert(0, 'sub', s)
        cor_all_subs = pd.concat([cor_all_subs,new_sub])


# cor_all_subs.insert(0, 'sub', sub_lst)
cor_all_subs.set_index('sub', inplace=True)

#%%
# calculate the average correlation between Garmin stress to each parameter across subjects
# (average of the results above)
cor_mean_sub = cor_all_subs.mean()
# cor_mean_sub = cor_mean_sub.drop('sub')
display(cor_mean_sub)
cor_mean_sub.name = 'mean_pearson_r'

cor_mean_pearson = cor_mean_sub.to_frame()

#%%

# test for significance of the averaged correlation: 
    
   # (apply fisher transformation on the r values and then 1-sample t-test)


t_list = [] ; p_list= []
cor_all_subs_1 = cor_all_subs.dropna()
for f in flst: 
    # perform one sample t-test
    a = cor_all_subs_1[f]
    mean_cor = a.mean()
    # transformation for the correlation before the t-test (Jeanette)
    # Get the indexes
    # indexes = a.index
    z_fisher = [0.5*(np.log(1+a[i])-np.log(1-a[i])) for i in a.index]
    plt.hist(z_fisher)
    # plt.title(f'Pearson correlation coefficient distribution - after Fisher transform. {f}')
    # plt.show()

    tStat, pValue =  scipy.stats.ttest_1samp(z_fisher, popmean=0, axis=0)
    print(f'\nperson r - Garmin_s"s and {f}:')
    print('Mean Pearson r: {:0.2f}\n  P-Value: {:0.3f}\n  T-Statistic: {:0.2f}'.format(mean_cor,pValue,tStat)) #print the P-Value and the T-Statistic
    cor_mean_pearson.loc[f,'p-value'] = pValue
    
    t_list.append(tStat) ; p_list.append(pValue)


# Define a custom function to add stars based on p-values
def add_stars(p_value):
    if p_value < 0.0125:
        return '*'
    else:
        return ''

# Apply the custom function to the 'P-value' column to create the 'Stars' column
cor_mean_pearson['sig'] = cor_mean_pearson['p-value'].apply(add_stars)

print(cor_mean_pearson)    

#%%

#### create a descriptive statistics summary table of the pearsons correlations

t = pd.Series(t_list, index = cor_all_subs.columns[0::], name = 'T_Statistic')
p = pd.Series(p_list, index = cor_all_subs.columns[0::], name = 'P_Value')

cor_all_subs_describe = cor_all_subs.describe()

cor_sum_pearson = cor_all_subs_describe.append(t,ignore_index=False)
cor_sum_pearson = cor_sum_pearson.append(p,ignore_index=False)

# cor_mean_pearson = cor_mean_pearson.rename_axis('parameter').reset_index()
print(cor_mean_pearson)
print(cor_sum_pearson)

#%%

#######   export 

cor_mean_pearson.to_csv(f'{res_path}/df_cor_mean_pearson.csv', index=False)       
cor_all_subs.to_csv(f'{data_path}/Results/cor_all_subs.csv', index=True)       

cor_sum_pearson.to_csv(f'{data_path}/Results/cor_sum_pearson.csv', index=True)       



#%%


# ****   plot  - within subject correlation  ****

#   boxplot with a table for the within subject pearson correlation   ************

# Get the correlation DataFrame 
df_corr = cor_all_subs

# Calculate mean and p-value for each metric (these are just placeholders, replace them with your actual calculations)
mean_values = df_corr.mean()
std_values = df_corr.std()
p_values = p

# Plot boxplot for each metric
plt.figure(figsize=(12, 8))
ax = df_corr.boxplot()
plt.title("Pearson's correlation coefficients for HR and HRV Metrics with Garmin Stress Score - Within-Subjects Analysis")
plt.ylabel("Pearson's correlation coefficient")
# plt.xticks(rotation=45)
# plt.xticks([])
# Setting x-ticks in the upper part
plt.xticks(position='top')

plt.grid(True)


# Add table with mean and p-value
# Transpose cell_text so that each row corresponds to a metric and each column corresponds to 'Mean' and 'P-value'
cell_text = [[f'{mean_values[i]:.2f}', f'{std_values[i]:.2f}', f'{p_values[i]:.3f}'] for i in range(len(mean_values))]  # Format mean, std, and p-value
cell_text = list(zip(*cell_text))

df_corr.rename(columns=variable_mapping, inplace=True)

table = plt.table(cellText=cell_text, rowLabels=['Mean', 'SD', 'P-value'], colLabels=df_corr.columns,
                  loc='center', cellLoc='center', bbox=[0, -0.35, 1, 0.3])  # Adjust bbox as needed
# Mark horizontal line at y=0.0 in red
ax.axhline(0.0, color='red')

# Add stars above boxplots for p-values below 0.0125
for i, p_value in enumerate(p_values):
    if p_value < 0.0125:
        plt.scatter(i+1, 1.1, marker='*', color='black', s=90, alpha = 0.5)


plt.show()


#%%


########################################################################### 
#######      Pearsons correlations of means in each condition     #########
###########################################################################  
###########################################################################


#%%

#************************    new code  ************************
#**************************************************************

#  (for Pearsons correlations of means in each condition)

data_means = pd.read_csv(f'{res_path}/df_mean_wide_edit.csv')
df = data_means
df = df.rename(columns=variable_mapping)

# Calculate Pearson correlation coefficient and p-value for each condition across subjects
conditions = df['condition'].unique()
# metrics = df.columns[2:]
metrics = selected_metrics.copy()
metrics.remove('garmin_stress')
metrics = metrics+['reported_stress']
metrics = [variable_mapping[name] for name in metrics]


results = pd.DataFrame(index=metrics, columns=pd.MultiIndex.from_product([conditions, ['Pearson_r', 'p-value', 'sig']], names=['Condition', 'Pearson Correlaltion Coefficient']))

for condition in conditions:
    condition_data = df[df['condition'] == condition]
    for metric in metrics:
        r, p_value = pearsonr(condition_data[metric], condition_data['Garmin Stress Score']) 
        results.loc[metric, (condition, 'Pearson_r')] = "{:.2f}".format(r)
        results.loc[metric, (condition, 'p-value')] = p_value
        if p_value < 0.0125:
           results.loc[metric, (condition, 'sig')] = '*'
        else:
           results.loc[metric, (condition, 'sig')] = ''



# Convert column 'A' to numeric type (integer or float)
results['Baseline','Pearson_r'] = pd.to_numeric(results['Baseline','Pearson_r'])
results['Stress','Pearson_r'] = pd.to_numeric(results['Stress','Pearson_r'])
results['Recovery','Pearson_r'] = pd.to_numeric(results['Recovery','Pearson_r'])


# results.index = [variable_mapping[name] for name in results.index]

# Display results
print(results)

results.to_csv(f'{res_path}/cor_pearson_conditions.csv', index=True)


#%%


# Plotting
fig, ax = plt.subplots(figsize=(12, 8))

bar_width = 0.2
index = np.arange(len(metrics))

colors = ['green', 'red', 'blue']
# Define three levels of grey colors
# colors = ['#CCCCCC', '#444444','#888888']

for i, condition in enumerate(conditions):
    bar_positions = index + i * bar_width
    for j, metric in enumerate(metrics):
        r = results.loc[metric, (condition, 'Pearson_r')]
        p = results.loc[metric, (condition, 'p-value')]
        
        ax.bar(bar_positions[j], r, bar_width, color=colors[i], label=condition)
        
        # add the r value and a star for significance
        if p < 0.0125:
            if r > 0:
                ax.text(bar_positions[j], r, f'{r}\n*',fontsize=12, ha='center', va='bottom')
            else: 
                ax.text(bar_positions[j], r-0.14, f'{r}\n*',fontsize=12,  ha='center', va='bottom')
        else:
            if r > 0:
                ax.text(bar_positions[j], r, f'{r}',fontsize=12,  ha='center', va='bottom')
            else: 
                ax.text(bar_positions[j], r-0.08, f'{r}',fontsize=12,  ha='center', va='bottom')


ax.set_xticks(index + 1 * bar_width)
ax.set_xticklabels(metrics)
ax.tick_params(axis='x', labelsize=12)


# ax.legend(['Baseline', 'Stress', 'Recovery'])

# Custom legend
# handles, labels = ax.get_legend_handles_labels()
# unique_labels = list(set(labels))
# ax.legend(handles[:len(conditions)], unique_labels, loc='upper right')

legend_handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(conditions))]
ax.legend(legend_handles, conditions, loc='upper right', fontsize=12)


plt.xlabel('')
plt.ylabel('Pearson Correlation Coefficient')

# ax.set_ylim(-1, 1)  # Set y-axis limits
# ax.set_yticks(np.linspace(-1, 1, num=9))  # Set y-axis ticks from -1 to 1
plt.ylim(-1,1)

# Add grid
ax.grid(True)

plt.title('Pearson Correlation Coefficient between HR and HRV Metrics and Garmin Stress Score')
plt.tight_layout()

plt.show()




  #%%


###############   Old Code

# for each condition seperatly - calculate the correlations between mean Garmin stress to the mean value of each parameter across subjects.
# creates 3 tables : correlation during baseline, cor during stress, cor during recovery.



# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

data = df_means.dropna() 
cols = data.columns
display(cols)

# flst = ['garmin_stress','HR','RMSSD','HF_ms2','HF_nu', 'LF/HF']
# for f in flst:
#     b = data[f] 
#     r, pval = pearsonr(a, b)
#     # print(r,", " ,pval)

# data1 = data.filter(like='Baseline')
# data2 = data.filter(like='Stress')
# data3 = data.filter(like='Recovery')

# cormat1 = data1.corr()
# cormat2 = data2.corr()
# cormat3 = data3.corr()

# cormat1.to_csv(f'{res_path}/cormat1.csv', index=False)
# cormat2.to_csv(f'{res_path}/cormat2.csv', index=False)
# cormat3.to_csv(f'{res_path}/cormat3.csv', index=False)


def cal_cor(df, condition): 
    df = df.dropna() 
    dic = {'condition':[],'parameter':[], 'pearson_r':[], 'pval':[]}

    df_con = df.filter(like = condition)
    cor_mat = df_con.corr()
    
    flst = df_con.columns.tolist()
    flst = [item for item in flst if 'garmin_stress' not in item]
    flst
    gar_col = f'garmin_stress_{condition}'
    
    a = df_con[gar_col]
    for f in flst:
        b = df_con[f] 
        r, pval = pearsonr(a, b)
        # print(r,", " ,pval)   
        dic['condition'].append(condition); dic['parameter'].append(f);  
        dic['pearson_r'].append(r);  dic['pval'].append(pval)
    
    df_correlation = pd.DataFrame(dic)
    
    return(df_correlation)

cor_1 = cal_cor(df_means, '1-Baseline')
cor_2 = cal_cor(df_means, '2-Stress')
cor_3 = cal_cor(df_means, '3-Recovery')

cor_1.to_csv(f'{res_path}/cor_condition1.csv', index=False)
cor_2.to_csv(f'{res_path}/cor_condition2.csv', index=False)
cor_3.to_csv(f'{res_path}/cor_condition3.csv', index=False)


pdList = [cor_1, cor_2, cor_3]  # List of your dataframes
cor_sum = pd.concat(pdList)
cor_sum = cor_sum.sort_values(by ='parameter' )
# cor_sum = cor_sum.sort_values(by ='condition' )

cor_sum.iloc[:, ~cor_sum.columns.isin(['parameter', 'condition'])] = cor_sum.iloc[:, ~cor_sum.columns.isin(['parameter', 'condition'])].applymap(lambda x: float(x))
cor_pearson_between_subs = cor_sum
cor_pearson_between_subs.to_csv(f'{res_path}/cor_pearson_between_subs.csv', index=True)

display(cor_pearson_between_subs)

