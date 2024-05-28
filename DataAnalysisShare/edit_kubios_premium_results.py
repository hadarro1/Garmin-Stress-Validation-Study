# -*- coding: utf-8 -*-
"""
Created on Thu Apr 28 14:31:54 2022

@author: hadar
"""


import glob
import pandas as pd
import datetime
from datetime import timedelta  



# path = input('Insert path: ')
path = 'C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2/Data_Analysis/Data/kubios_results/'

# samp_duration = int(input('type the sample length (3 or 6 minutes) ?' ))

samp_duration = 3

if (samp_duration == 3):    
    filename = path + 'KubiosHRVresults_3min_WOrow1.csv'
    samples = 15 # for the second experiment (3 tasks of 15 minutes >> 45 minutes total)

elif (samp_duration == 6):  #irrelevant
    filename = path + 'KubiosHRVresults_6min.csv'
    samples = 15 


data = pd.read_csv(filename)
'''
data = open(filename, 'r').readlines()

# path = 'C:/Users/hadar/OneDrive/Documents/MASTER/2-RESEARCH/Vikkie_HRV/'
# filename = path + 'KubiosHRVresults.csv'
# data = open(filename, 'r').readlines()

import csv
file = open(filename)
csvreader = csv.reader(file)
header = next(csvreader)
print(header)
rows = []
for row in csvreader:
    rows.append(row)
print(rows)
file.close()

#%%

data = pd.DataFrame()
for r in range(len(rows)):
    data = data.append([rows[r]], ignore_index=True)
# for r in range(5):
#     data = data.append([rows[r]])    

#%%
colnames = data.iloc[0,:]
data.columns = colnames  
data.drop(index=0, inplace=True) 
# data.columns = header  
data.head()
data.columns
#%%
'''

sublist = [int(data['FileName'][int(sub)].split('_')[0].split('-')[1]) for sub in range(len(data['FileName']))] # I changed the loop range, since the df starts in 0-index
print(sublist)

#%%

my_dict = {}
for i in range(samples):      # run on all samples of all subs
    
    # for c in data.columns:
    #     if (f'S{i+1}_' in [c]):  # take all the column S1 (for the first iteration)
    #         samp_cols = [c]
    samp_cols = [c for c in data.columns if f'S{i+1}_' in c]
    data_samp = data[samp_cols]
    data_samp.insert(0,'FileName', data.FileName)
    
    # set the recording and the sample onset:
    
    t_rec = [datetime.datetime.strptime(data.FileName[int(sub)].split('_')[1].split('.')[0], "%Y-%m-%d %H-%M-%S") for sub in range(len(data.FileName))]
    sample_time = []
    for sub in range(len(data.FileName)):    
        onset = data_samp[f'S{i+1}_Onset-Offset'][1].split('-')[0]
        onset_hr = pd.to_timedelta(int(onset.split(':')[0]), unit = "hour")
        onset_min = pd.to_timedelta(int(onset.split(':')[1]), unit = "min")
        onset_sec = pd.to_timedelta(int(onset.split(':')[2]), unit = "sec")
        sample_time.append(t_rec[sub-1] + onset_hr + onset_min + onset_sec)
    
        # add the columns with the time info
    data_samp.insert(0,'sample_time', sample_time)
    data_samp.insert(0,'t_recording', t_rec) 
    

    # add the coulumns with the sub and sample number info
    data_samp.insert(0,'sample_num', f'{i+1}')
    data_samp.insert(0,'sub', [data.FileName[int(sub)].split('_')[0].split('-')[1] for sub in range(len(data.FileName))])

    
    # change the columns names that it will be uniform for all samples to join them later
    col_names = data_samp.columns # get the original columns names
    new_colnames = [s.replace(f'S{i+1}_', "") for s in col_names]
    data_samp.columns = new_colnames
    
    my_dict[f'{i+1}'] = data_samp

#%%

df = pd.DataFrame()
for sub in sublist:
    for k in my_dict.keys():
        x = my_dict.get(k)
        sub_dat = x.loc[x['sub'] == f'{sub}']
        df = df.append(sub_dat)
    


df.columns
#%%

df_kub_prem = df[['sub','sample_num','sample_time',
                 'Stress index','PNS index','SNS index',
                 'LF_HF_ratio_AR', 'LF_HF_ratio_FFT', 
                 'LFpow_AR (ms2)', 'LFpow_AR (log)', 'LFpow_AR (n.u.)', 
                 'HFpow_AR (ms2)', 'HFpow_AR (log)', 'HFpow_AR (n.u.)',
                 'Mean RR (ms)', 'RMSSD (ms)', 'SDNN (ms)', 'pNNxx (%)','HRV triangular index',
                 'Mean HR (bpm)','SD HR (bpm)','Min HR (bpm)','Max HR (bpm)',
                 'SD2_SD1_ratio']]

# rename columns
names = ['subID','sample_num', 'sample_time',
       'stress_index', 'PNS_index', 'SNS_index', 
       'LF_HF_ratio_AR', 'LF_HF_ratio_FFT',
       'LFpow_AR_ms2', 'LFpow_AR_log', 'LFpow_AR_nu', 
       'HFpow_AR_ms2', 'HFpow_AR_log', 'HFpow_AR_nu',
       'Mean_RR_ms', 'RMSSD_ms', 'SDNN_ms', 'pNN50', 'RR_tri_index',
       'Mean_HR_bpm', 'SD_HR_bpm', 'Min_HR_bpm','Max_HR_bpm',
       'SD2_SD1_ratio']
df_kub_prem.columns = names

df_kub_prem.to_csv(f'{path}KubiosHRVresults_edited_{samp_duration}.csv', index=False)       



# [['subID','sample_num', 'sample_time','part', 'garmin_stress',
#        'stress_index', 'PNS_index', 'SNS_index', 
#        'RMSSD_ms', 'LF_HF_ratio_AR', 'LF_HF_ratio_FFT', 
#        'Mean_RR_ms', 'SDNN_ms', 'pNN50', 'RR_tri_index',
#        'Mean_HR_bpm', 'SD_HR_bpm', 'Min_HR_bpm','Max_HR_bpm']]

