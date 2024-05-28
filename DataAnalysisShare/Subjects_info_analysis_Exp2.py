# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 12:52:55 2024

@author: hadar


Analyze subject's info with HR,HRV,and Garmin's means

Takes data from subjects_full and from df_means to combine 
a full dataframe with a row for each subject.

"""

# import glob
import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import stats
# import seaborn as sns


################

sub_error = [int(i) for i in['403','404', '406', '409','412','413', '414', '415', '418', '429', '431', '432', '433', '435', '439', '445', '446', '454', '457','458','459', '460', '463', '464','472','474','475','480','488','489']]

################


general_path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2'
data_path = f'{general_path}/Data_Analysis'
res_path = f'{data_path}/Results'

path_quiz = general_path+'StressTask_quiz-Exp2/quiz_results_folder/' # path to the quiz resuts


sub_info = pd.read_csv(f'{data_path}/Data/sub_info2.csv')
sub_info = sub_info.drop([0])

######
sub_lst = sub_info['Participant_number'].unique()
sub_lst = [int(s) for s in sub_lst]

###### 
# Manually create a list based on range:
sub_lst = [*range(401, 491, 1)] 
print(sub_lst) 
######

#%%

# remove invalid subjects from the subs list:
sub_lst = [i for i in sub_lst if i not in sub_error]  # remove subs that are invalid (list of invalid is at the begining of the code)
print(f'Valid Subjects: {sub_lst}') 


#%%
# orgenize subs info into a dataframe, edit, and calculate statistics
sub_info.columns
subjects = sub_info[['Participant_number', 'watch_serial_number', 
                     'height', 'weight', 'age', 'birthday', 'sex', 'Dominant_hand',
      'final_quiz_score', 'is_best', 'missing_stress_points', 'PSS-14_score', 
       'reported_stress_1', 'reported_stress_2', 'reported_stress_3', 
       'synched_start_time']]

subjects.dtypes

####### convert columns data type to numeric
subjects = subjects.astype({'Participant_number': 'int'})

mask = subjects['Participant_number'].isin(sub_lst)
subjects = subjects[mask]    

subjects = subjects.astype({'weight':'float', 'height':'float', 'age':'int', 'final_quiz_score':'int', 'PSS-14_score':'int'}, errors='ignore')

columns_to_convert = ['final_quiz_score', 'missing_stress_points', 'PSS-14_score', 'reported_stress_1', 'reported_stress_2','reported_stress_3',]
subjects[columns_to_convert] = subjects[columns_to_convert].apply(pd.to_numeric, errors='coerce')
subjects.dtypes


subjects['bmi'] = (subjects['weight'])/((subjects['height']**2))
subjects['height'] = subjects['height']*100
subjects = subjects.rename(columns={'Participant_number': 'subject', "height": "height[cm]", "weight": "weight[Kg]", "bmi": "bmi_Kg/m^2", 'final_quiz_score':'math_quiz_score'})

subjects = subjects.drop(['is_best'], axis=1)

sub_stat = subjects.describe()  # get summary atatistic of the table
sub_stat = sub_stat.T.round(decimals=2)
# pd.options.display.float_format = '{:.5f}'.format

subjects.to_csv(f'{res_path}/subjects.csv')
# sub_stat.to_csv(f'{res_path}/sub_stat.csv')



#%%

# ############  add data from personal info questionair   ############


sub_more = pd.read_csv(f'{data_path}/Data/personal_questionair_exp2.csv')
sub_more = sub_more.sort_values(by='subject')

duplicates_mask = sub_more['subject'].duplicated()


mask= sub_more['subject'].isin(sub_lst)
sub_more = sub_more[mask] 

# Merge specific columns from df2 to df1 based on 'subject'
subjects_full = pd.merge(subjects, sub_more, on='subject', how='left', right_index=False)

subjects_full.columns
print(subjects_full)


#%%

# Filter out columns using the drop() function
subjects_full = subjects_full.drop(['watch_serial_number', 'birthday', 'synched_start_time'], axis=1)

# Convert column of 0 and 1 to boolean
subjects_full[['smoking', 'oral_contraceptive', 'exercise']] = subjects_full[['smoking', 'oral_contraceptive', 'exercise']].astype(bool)

#%%

# change exercise columns to numeric values:
# Define a mapping dictionary
exercise_mapping = {'low': 1, 'medium': 3, 'high': 5, pd.NA : 0}

# Map the exercise frequency column to numeric values
subjects_full['exercise'] = subjects_full['exercise_frequency'].map(exercise_mapping)
subjects_full['exercise'].fillna(0, inplace=True)
subjects_full = subjects_full.rename(columns={'exercise': 'exercise_freq'})
                                    
subjects_full = subjects_full.drop(['exercise_frequency'], axis=1)


#%%

subjects_full.to_csv(f'{res_path}/subjects_full.csv')

#%%




###   statistics of subjects for the "Participant Characteristics" paragraph in the method ction
#############################################################################

subjects_full.columns

df = subjects_full.drop(['sleep_start_time', 'awake_time'], axis=1)
df.rename(columns = {'sub':'subject'}, inplace = True)



# Calculate mean and standard deviation for numeric columns
numeric_stats = df.describe().loc[['mean', 'std']]

# Calculate count and percentage for non-numeric columns
categorical_stats = {}
categorical_columns = ['sex', 'Dominant_hand', 'smoking', 'oral_contraceptive', 'exercise_freq', 'exercise_intensity']
for col in categorical_columns:
    counts = df[col].value_counts()
    categorical_stats[col + '_count'] = counts
    categorical_stats[col + '_percentage'] = counts / len(df) * 100

# Combine numeric and non-numeric statistics into one DataFrame
stats_df = pd.concat([numeric_stats, pd.DataFrame(categorical_stats)])


stat_participants_categorical = pd.DataFrame(categorical_stats).transpose() 
stat_participants_numeric = numeric_stats.transpose() 

cat = df.describe(include='object')
#%%
stat_participants_categorical.to_csv(f'{data_path}/Results/stat_participants_categorical.csv', index=True)       
stat_participants_numeric.to_csv(f'{data_path}/Results/stat_participants_numeric.csv', index=True)       

describe = df.describe()
