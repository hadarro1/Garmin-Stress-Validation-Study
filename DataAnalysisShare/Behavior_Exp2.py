# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:56:31 2022

@author: AlonItzko
"""
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns


def is_stressed(sub, thres = 0.1):
    global mean_diff
    return int(float(mean_diff[mean_diff['sub'] == sub] \
                     [mean_diff['feature'] == 'RMSSD_ms']['d_recovery_p3_p2']) > thres)
        # check if the sub had a lower rmssd during the stressfull task 
        # compare to the recovery
    
general_path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2'
data_path = f'{general_path}/Data_Analysis'
res_path = f'{data_path}/Results'

path_quiz = general_path+'StressTask_quiz-Exp2/quiz_results_folder/' # path to the quiz resuts




### XXXXXXXXXXXXXXX    I updated the code up to here!    XXXXXXXXXXXXXX ###



#%%
norm_df = pd.read_csv(f'{path_dfs}df_norm_all.csv')
mean_diff = pd.read_csv(f'{path_dfs}mean_df.csv')

#%%


subs1 = glob.glob(f'{path_quiz}/*sum.csv')
subs1 = [int(s[-11:-8]) for s in subs1]

subs2 = mean_diff['sub'].unique()

intersection = set(subs1).intersection(subs2)
sub_list = intersection
sub_list
len(sub_list)
################

#%%

full_dict = {'sub': [], 'gender': [], 'age': [], 'hour': [],
             'mean_RT': [], 'std_RT': [], 'delta_RT': [], 'correct_answers': [],
             'best_score': [], 'total_score': [], 'std_score': [], 'd_score_sgoli': [],
             'reactivity_rmssd': [], 'recovery_rmssd': [],'recovery_SNS': [], 'recovery_HR': [], 'recovery_SD2_SD1': [], 'reactivity_garmin': [], 'recovery_garmin': [], 
             'stressed': [], 'PSS_score':[]}

for sub in sub_list:
    full_dict['sub'].append(sub)
    g = 1 if 'F' in str(subjects[subjects['sub'] == str(sub)]['gender']) else 0
    full_dict['gender'].append(g)
    full_dict['age'].append(int(subjects['age'].loc[subjects['sub'] == str(sub)]))
    full_dict['stressed'].append(is_stressed(sub))
    full_dict['reactivity_rmssd'].append(float(mean_diff[mean_diff['sub'] == sub][mean_diff['feature'] == 'RMSSD_ms']['d_reactivity_p2_p1']))
    full_dict['recovery_rmssd'].append(float(mean_diff[mean_diff['sub'] == sub][mean_diff['feature'] == 'RMSSD_ms']['d_recovery_p3_p2']))
    full_dict['reactivity_garmin'].append(float(mean_diff[mean_diff['sub'] == sub][mean_diff['feature'] == 'garmin_stress']['d_reactivity_p2_p1']))
    full_dict['recovery_garmin'].append(float(mean_diff[mean_diff['sub'] == sub][mean_diff['feature'] == 'garmin_stress']['d_recovery_p3_p2']))
    
    full_dict['recovery_SNS'].append(float(mean_diff[mean_diff['sub'] == sub][mean_diff['feature'] == 'SNS_index']['d_recovery_p3_p2']))
    full_dict['recovery_HR'].append(float(mean_diff[mean_diff['sub'] == sub][mean_diff['feature'] == 'Mean_HR_bpm']['d_recovery_p3_p2']))
    full_dict['recovery_SD2_SD1'].append(float(mean_diff[mean_diff['sub'] == sub][mean_diff['feature'] == 'SD2_SD1_ratio']['d_recovery_p3_p2']))

    full_dict['PSS_score'].append(int(subjects['PSS-14_score'].loc[subjects['sub'] == str(sub)]))
                                                 
    sub_sum = pd.read_csv(f'{path_quiz}/{sub}_sum.csv')
    full_dict['hour'].append(int(str(sub_sum['timestamp_start'])[16:18]))
    full_dict['total_score'].append(int(sub_sum['total_score']))
    
    score_a = int(sub_sum['score_trial_A']); score_b = int(sub_sum['score_trial_B']); score_c = int(sub_sum['score_trial_C'])
    full_dict['best_score'].append(max(score_a, score_b, score_c))
    full_dict['d_score_sgoli'].append((score_c/7.5)-(score_a/9))
    full_dict['std_score'].append(np.std([score_a, score_b, score_c]))
    full_dict['correct_answers'].append(int(sub_sum['total_correct_answers']) / int(sub_sum['total_number_of_questions']))
    
    sub_rt = pd.read_csv(f'{path_quiz}/{sub}_details.csv')
    full_dict['mean_RT'].append(np.mean(sub_rt['reaction_time']))
    full_dict['std_RT'].append(np.std(sub_rt['reaction_time']))
    
    rt_a = sub_rt[sub_rt['trial'] == 'Trial A']['reaction_time']
    rt_c = sub_rt[sub_rt['trial'] == 'Trial C']['reaction_time']
    full_dict['delta_RT'].append(np.mean(rt_c) - np.mean(rt_a))
    
full_df = pd.DataFrame(full_dict)
correlation_matrix = full_df.corr()

# full_df.to_csv(f'{path_dfs}behavioral_data.csv')

print(f'Stressed Percent = {np.mean(full_df["stressed"]):.3f}')

#%%


sns.scatterplot(x='reactivity_rmssd', y='best_score', hue='stressed', data=full_df, palette="bright")   
sns.scatterplot(x='recovery_rmssd', y='best_score', data=full_df, palette="bright")   
sns.scatterplot(x='PSS_score', y='best_score', data=full_df, palette="bright")   
