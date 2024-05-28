# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:20:04 2024

@author: hadar
"""

# import glob
# import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# import scipy.stats as stat
from scipy.stats.stats import pearsonr
# from statistics import stdev
# import matplotlib.patches as mpatches
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import MinMaxScaler
# import math 
# import scipy

#%%

# import edited data that created in 'Dataframin_and_visualization.py'


general_path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2'
data_path = f'{general_path}/Data_Analysis'
res_path = f'{data_path}/Results'
df_means_all = pd.read_csv(f'{res_path}/df_means_all.csv')
df_mean_wide = pd.read_csv(f'{res_path}/df_mean_wide_edit.csv')
subjects_full = pd.read_csv(f'{res_path}/subjects_full.csv')




#%%

# ============================================================================
########### Correlations of PSS-14_score with HR, HRV, and Garmin score
# ============================================================================


def cal_cor_with_characteristic(df,char):
    
    # Calculate correlations for Baseline columns
    correlations_baseline = {}
    for column in df.columns:
        if '_Baseline' in column:
            correlation, p_value = pearsonr(df[char], df[column])
            correlations_baseline[column] = {'correlation': correlation, 'p_value': p_value}
    
    # Calculate correlations for Delta columns
    correlations_delta = {}
    for column in df.columns:
        if '_Delta' in column:
            correlation, p_value = pearsonr(df[char], df[column])
            correlations_delta[column] = {'correlation': correlation, 'p_value': p_value}
    
    # Create DataFrame from correlations dictionaries
    correlation_df_baseline = pd.DataFrame.from_dict(correlations_baseline, orient='index')
    correlation_df_delta = pd.DataFrame.from_dict(correlations_delta, orient='index')
    
    
    print(f'Baseline Correlations with {char}:')
    print(correlation_df_baseline)
    print(f'\nDelta Correlations with {char} :')
    print(correlation_df_delta)
    
    # # Export DataFrames to CSV
    # correlation_df_baseline.to_csv('correlation_results_baseline.csv')
    # correlation_df_delta.to_csv('correlation_results_delta.csv')
    
    
    return



#%%

df = df_means_all
df = df.dropna()

#%%

cal_cor_with_characteristic(df,'PSS-14_score')

#%%

cal_cor_with_characteristic(df,'math_quiz_score')


#%%

cal_cor_with_characteristic(df,'sleep_number_hours')


#%%

cal_cor_with_characteristic(df,'bmi_Kg/m^2')


#%%

cal_cor_with_characteristic(df,'exercise_freq')


#%%


#%%
df_means_all.columns
df_means_all.shape

#%%

#########     plot values across conditions seperated by the categorical characteristics
################################################################################################
 
def create_cond_df(df, param, characteristic):
  temp_df = pd.DataFrame()
  temp_df[characteristic] = df[characteristic]
  for con in ['Baseline', 'Stress', 'Recovery']:
    temp_df[con] = df[param + '_' + con]
  #temp_df = temp_df.groupby(condition).mean().reset_index()
  temp_df = temp_df.melt(id_vars = characteristic, var_name = 'condition', value_name = param)
  return temp_df


#%%


def plot_means_for_individual_char(df, param, characteristic):
    # creating the dataframe of the specific characteristic
    temp_df = create_cond_df(df, param, characteristic)  
    
    
    # Plotting
    plt.figure(figsize=(6, 4))
    sns.lineplot(x='condition', y=param, hue=characteristic, data=temp_df, marker='o', markersize=8)
    plt.title(f'Mean {param} in Each Condition (Separated by {characteristic})')
    plt.xlabel('Condition')
    plt.ylabel(param)
    plt.legend(title=characteristic, loc='upper left')
    plt.show()
    return
    
    
#%%

df = df_means_all

# choosing the characteristic to present and calling the plot function for each metric
# selected_metrics = ['HR', 'RMSSD', 'SD2/SD1', 'LF/HF', 'HF_nu', 'garmin_stress', 'reported_stress']

char = 'sex'

plot_means_for_individual_char(df, 'HR', char)
plot_means_for_individual_char(df, 'RMSSD', char)
plot_means_for_individual_char(df, 'LF/HF', char)
plot_means_for_individual_char(df, 'HF_nu', char)
plot_means_for_individual_char(df, 'SD2/SD1', char)
plot_means_for_individual_char(df, 'garmin_stress', char)
plot_means_for_individual_char(df, 'reported_stress', char)


#%%

char = 'exercise'

plot_means_for_individual_char(df, 'HR', char)
plot_means_for_individual_char(df, 'RMSSD', char)
plot_means_for_individual_char(df, 'LF/HF', char)
plot_means_for_individual_char(df, 'HF_nu', char)
plot_means_for_individual_char(df, 'SD2/SD1', char)
plot_means_for_individual_char(df, 'garmin_stress', char)
plot_means_for_individual_char(df, 'reported_stress', char)


#%%

char = 'exercise_freq'

plot_means_for_individual_char(df, 'HR', char)
plot_means_for_individual_char(df, 'RMSSD', char)
plot_means_for_individual_char(df, 'LF/HF', char)
plot_means_for_individual_char(df, 'HF_nu', char)
plot_means_for_individual_char(df, 'SD2/SD1', char)
plot_means_for_individual_char(df, 'garmin_stress', char)
plot_means_for_individual_char(df, 'reported_stress', char)


#%%

char = 'oral_contraceptive'

plot_means_for_individual_char(df, 'HR', char)
plot_means_for_individual_char(df, 'RMSSD', char)
plot_means_for_individual_char(df, 'LF/HF', char)
plot_means_for_individual_char(df, 'HF_nu', char)
plot_means_for_individual_char(df, 'SD2/SD1', char)
plot_means_for_individual_char(df, 'garmin_stress', char)
plot_means_for_individual_char(df, 'reported_stress', char)

#%%

########## Linear Mixed-Effects Model (LMM): from chatGPT
# prepare dataframe:
subjects_full = subjects_full.drop(columns=['Unnamed: 0'])

#%%

data = pd.merge(df_mean_wide, subjects_full, on='subject', how='left')
data = data.drop(columns=['reported_stress_1', 'reported_stress_2','reported_stress_3','height[cm]', 'weight[Kg]','Dominant_hand','missing_stress_points','smoking','sleep_start_time', 'awake_time'])
data.rename(columns = {'Mean_HR_bpm':'HR', 'RMSSD_ms':'RMSSD', 'SD2_SD1_ratio':'SD2_SD1', 'HFpow_AR_nu':'HF_nu','LF_HF_ratio_AR':'LF_HF', 'PSS-14_score': 'PSS_14_score','bmi_Kg/m^2':'bmi', 'sleep_number_hours': 'sleep_dur'}, inplace = True)

data.columns
# ['subject', 'condition', 'HR', 'RMSSD', 'SD2_SD1', 'LF_HF', 'HF_nu', 'garmin_stress',
#    'reported_stress', 'age', 'sex', 'math_quiz_score',
#   'bmi', 'oral_contraceptive', 'exercise', 'PSS_14_score',
#    'exercise_freq', 'exercise_intensity', 'sleep_number_hours']

#%%

import statsmodels.api as sm
import statsmodels.formula.api as smf

# Assuming your data is stored in a pandas DataFrame called 'data'

dependant_variable = ['HR', 'RMSSD', 'SD2_SD1', 'LF_HF', 'HF_nu', 'garmin_stress', 'reported_stress']

def run_MLM(dv, data):
    # main_factor = 'condition'
    main_factor1 =  'sex'
    main_factor2 = 'exercise'
    
    
    # Define the model formula
    formula = f'{dv} ~ {main_factor1} * condition + {main_factor2} * condition + age + bmi + oral_contraceptive + sleep_dur + (1 | subject)'
    ### excluded because of error:  + math_quiz_score + PSS_14_score + exercise_intensity
    ### excluded in order to simplify the model: exercise_freq + 
    
    # Fit the linear mixed-effects model
    model = smf.mixedlm(formula, data, groups=data['subject']).fit()
    
    # Get the summary of the model
    print(model.summary())
    
    ## Save the summary of the model to a CSV file
    with open(f'{res_path}/mixed_effects_model_results_{dv}_by_condition_and_{main_factor1}_{main_factor2}.csv', 'w') as f:
        f.write(model.summary().as_text())
        
    return


#%%
for dv in dependant_variable:
    run_MLM(dv, data)
    
    
run_MLM('PSS_14_score', data.dropna())
#%%

# testing the mixed model for garmin stress based on HRV metrics 
    
dependant_variable = 'garmin_stress'
main_factor = 'RMSSD'


# Define the model formula
formula = f'{dependant_variable} ~ {main_factor} * condition + age + sex + exercise + bmi + oral_contraceptive + exercise_freq + sleep_number_hours + (1 | subject)'
### excluded because of error:  + math_quiz_score + PSS_14_score + exercise_intensity

# Fit the linear mixed-effects model
model = smf.mixedlm(formula, data, groups=data['subject']).fit()

# Get the summary of the model
print(model.summary())    
    
#%%
 