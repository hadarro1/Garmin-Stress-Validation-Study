# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:24:45 2024

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
dur = 15  # duration of each task/condition [minutes]

sub_error = [int(i) for i in['403','404', '406', '409','412','413', '414', '415', '418', '429', '431', '432', '433', '435', '439', '445', '446', '454', '457','458','459', '460', '463', '464','472','474','475','480','488','489']]


#%%



subID = glob.glob(data_path+'/eliteHRV_export/*')
subID = [s.split('\\')[-1][-3:] for s in subID if 'elite' in s.split('\\')[-1]]

parts_ord = ["p1", "p2", "p3"]


all_hrv_features = ['sub', 'part', 'sample_num', 'sample_time',
           'garmin_stress', 'Mean_HR_bpm', 'RMSSD_ms', 'ln_RMSSD', 'pNN50', 'SDNN_ms',
           'RR_tri_index', 'SD2_SD1_ratio',
           'LFpow_AR_ms2', 'LFpow_AR_log', 'LFpow_AR_nu', 
           'HFpow_AR_ms2', 'HFpow_AR_log', 'HFpow_AR_nu',          
           'LF_HF_ratio_AR', 'LF_HF_ratio_FFT',
           'stress_index', 'PNS_index', 'SNS_index']       
            
selected_metrics = ['garmin_stress','Mean_HR_bpm','SD2_SD1_ratio','LF_HF_ratio_AR', 'RMSSD_ms','HFpow_AR_nu']                  
           
 
# selected_metrics = ['garmin_stress','Mean_HR_bpm','SD2_SD1_ratio','LF_HF_ratio_AR',                  
#                    'RMSSD_ms','pNN50', 'HFpow_AR_ms2','HFpow_AR_nu']
                            


#%%

# ===========================================================================
# =====================    prepare the dataframes:     ======================
# ===========================================================================

################     import Kubios data:
    
kub_dat_path = f'{data_path}/Data/kubios_results/KubiosHRVresults_edited_{samp_duration}.csv'
f = open(kub_dat_path, 'r')
df_kub = pd.read_csv(f)
print(df_kub.head(3))

sub_kub = df_kub['subID'].unique()
sub_kub = [int(s) for s in sub_kub]

#%%

#################    import Garmin data:
    
gar_path = f'{data_path}/Data/garmin_data/garmin_processed_data/'
    
garmin_stress = 'garmin_stress_3' if (samp_duration == 3) else 'garmin_stress_6' 
sub_gar = glob.glob(f'{gar_path}*_{garmin_stress}.csv')
sub_gar = [int(s.split('\\')[-1][0:3]) for s in sub_gar]

df_gar_miss = pd.DataFrame(columns=['sub', 'missing_stress_points'])

df_garmin = pd.DataFrame()
# loop over subjects:
for sub in sub_gar:
    gar_dat_path = f'{gar_path}{sub}_{garmin_stress}.csv' 
    gf = open(gar_dat_path, 'r')
    df_garmin_sub = pd.read_csv(gf)
    print(df_garmin_sub.head(3))
    df_garmin_sub['subID'] = sub
    df_garmin = df_garmin.append(df_garmin_sub)
    
    df_gar_miss.loc[len(df_gar_miss.index)] = [sub, df_garmin_sub['mean_stress'].isnull().sum()]

df_garmin.rename(columns={ "mean_stress": 'garmin_stress'}, inplace = True) 

    
    
# df_gar_miss = df_garmin.groupby(['subID', 'part'], as_index=False)

#%%  
# ===========================================================================
# ===============    Merg Kubios (HRV) data with Garmin data    =============
# ===========================================================================

""" create subject's list of only subject that has both data (kubios and garmin)
and combine into one whole dataframe
"""

sub_lst = set(sub_kub).intersection(sub_gar)  # get the subs that have both data sets
sub_lst = [i for i in sub_lst if i not in sub_error]  # remove subs that are invalid (list of invalid is at the begining of the code)

df_all = pd.DataFrame()
for sub in sub_lst:
    df_kub_sub = df_kub[df_kub['subID'] == sub].reset_index()
    df_gar_sub = df_garmin[['garmin_stress', 'part']][df_garmin['subID'] == sub]
    data = pd.concat([df_gar_sub, df_kub_sub], axis=1)
    df_gar_sub = df_garmin[['subID', 'garmin_stress', 'part']][df_garmin['subID'] == sub]
    data.rename(columns={ data.columns[3]: "sub" }, inplace = True) 
    # data.drop(data.columns[5], axis=1, inplace=True)
    df_all = pd.concat([df_all, data])

df_all = df_all.reset_index()
df_all['ln_RMSSD'] = np.log(df_all['RMSSD_ms'])
df_all = df_all[all_hrv_features]


#%%

########## remove part 1 = the first 10 samples (30 minutes):
# ===========================================================================
   
display(df_all.shape) # (960, 23)
# df_all = df_all[df_all['part'] != 'p1']
# df_all = df_all[df_all['sample_num'].between(11, 31)]
# display(df_all.shape) # (640,23)
display(df_all.columns)


#%%


# ###########################################################################
# ==========      sempling shorter periods from the tasks:          ===========
# ###########################################################################
print(inp)  # defined at the beginning


if inp == '1':
    ######     18-minute per task:    ######
    # df1 = df_all.loc[df_all['sample_num'].isin([2,3,4,5,6,7, 12,13,14,15,16,17, 22,23,24,25,26,27])]
    ######     15-minute per task:    ######
    # df1 = df_all.loc[df_all['sample_num'].isin([2,3,4,5,6, 12,13,14,15,16, 22,23,24,25,26])]
    ######     12-minute per task:    ######
    df1 = df_all.loc[df_all['sample_num'].isin([2,3,4,5, 12,13,14,15, 22,23,24,25])]    
    

    # df2 = df_all[df_all['part'] == 3]
    # df_all_samp = pd.concat([df1,df2], axis=0)
    df_all = df1.sort_values(['sub', 'sample_num'])
    mydata = df_all.copy()
    mydata['part'] = mydata['part'].replace(['p1','p2','p3'],[1,2,3])
    
    mydata = df_all.groupby(['sub', 'part'], as_index=False).mean()                            
    
    
    def multiple_boxplots(my_data, vars_names):
        df_melt = my_data.melt(id_vars='part',  value_vars=[i for i in vars_names], var_name='columns')                                  
    
        a = sns.catplot(data = df_melt, x = 'part', y = 'value', 
                           kind = 'violin', # type of plot
                           col = 'columns', inner="quart", linewidth=1,
                           # custom order of boxplots:
                           col_order = [i for i in vars_names]).set_titles('{col_name}') # remove 'column = ' part of title
        plt.show()
        return
    
    multiple_boxplots(mydata, ['RMSSD_ms', 'HFpow_AR_nu'])
    
    print('Analyzing sampled data')

else:
    print('Analyzing full data')





#%%

df_all.to_csv(f'{data_path}/Results/df_all.csv', index=False)       

describe_all = df_all.describe().iloc[:, 2::]
describe_all.to_csv(f'{data_path}/Results/describe_all.csv', index=False)       

#%%
X = df_all[selected_metrics]
from matplotlib.colors import ListedColormap
cmap = ListedColormap([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
_ = pd.plotting.scatter_matrix(X,                               
                               figsize=(15, 15),
                               marker='o',
                               hist_kwds={
                                   'bins': 20,
                                   'alpha': 0.8
                               },
                               s=60,
                               alpha=0.4)

#%%

# ===========================================================================
###############         normalization within subject:         ###############
# ===========================================================================
""" create a normalized version within subject 
(each sub is normalized only to itself)
"""   
df_norm_all = pd.DataFrame()

for sub in sub_lst: 
    df_sub = df_all[df_all['sub']==sub]
    data = df_sub.drop(columns = ['sub', 'part','sample_num', 'sample_time'])
    
    scaler = StandardScaler()
    norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    # scaler = MinMaxScaler()
    # norm = pd.DataFrame(scaler.fit_transform(data), columns=data.columns, index=data.index)
    #### to merg back with the other datatype:
    df_ordinal = df_sub[['sub', 'part','sample_num', 'sample_time']]
    df_norm_sub = pd.concat([df_ordinal, norm], axis=1, join='inner')
    df_norm_all = pd.concat([df_norm_all, df_norm_sub])


df_norm_all.to_csv(f'{data_path}/Results/df_norm_all.csv', index=False)       
           
