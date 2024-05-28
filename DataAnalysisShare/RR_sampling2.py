# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 13:48:07 2021

@author: hadar

LONG EXPERIMENT VERSION (30-30-0-30): 
this code samples the elite export files of RR taken with POLAR to the 
6 minutes time intervals.

wehn there is only one long HRV reading for the whole experiment
"""
#%%
# runfile('C:/Users/hadar/OneDrive/Documents/MASTER/2-RESEARCH/experiment_validation/Analysis_codes/fun_data_organize.py', wdir='C:/Users/hadar/OneDrive/Documents/MASTER/2-RESEARCH/experiment_validation/Analysis_codes')

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import re
import os
import datetime
from datetime import timedelta  
import glob
import shutil
import math
# from fun_data_organize import * # my module


# samp_dur = 6
samp_dur = 3
break_duration = 0  
exp_min = 45

general_path = "C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2/Data_Analysis/Codes/"
garmin_path = "C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2/Data_Analysis/Data/garmin_data/"
# elite (polar) data:
elite_path = "C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2/Data_Analysis/Data/eliteHRV_export/"


# set the sub_ID :   
# get a list of sub_id from the eliteHRV data folder
sublist = glob.glob(elite_path+'elite_*') 
sublist = [s.split('\\')[-1][-3:] for s in sublist if 'elite' in s.split('\\')[-1]]
print(sublist)

#################################
# to run a specific subject data:
sublist = [488,489,490]
#################################


# invalids = ('319','332','333')
# sublist = [s for s in sublist if s not in invalids]


# sub_ID = int(input( "Enter subject ID:\n  "))
# # sub_ID = 309
# print(sub_ID)
#%%

# ========================================================================== 
# -------- start time calculations: ----------------------------------------
# ========================================================================== 


# get start time from the sub_info table of all the particpants
sub_info = pd.read_csv('C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2/Data_Analysis/Data/sub_info2.csv')
sub_info.columns
t_table = sub_info[['Participant_number','synched_start_time']]
t_table = t_table.iloc[1:]
t_table.rename(columns={ 'Participant_number': 'sub_ID' }, inplace = True)
print(t_table)

# count how many invalid subjects:
print(t_table['synched_start_time'].isna().sum())
len(t_table)

# to get rid of invalid participants

t_table = t_table.dropna() 
print(t_table)
t_table.shape
print(f'Number of valid datasets: {len(t_table)}')

# create a dataframe of samples intervals- order and parts

df_samp_int = pd.DataFrame(columns=['duration', 'accumulate', 'exp_part'])
n = int(exp_min/samp_dur)
df_samp_int = pd.DataFrame({'duration': [samp_dur for _ in range(n)] , 'accumulate': [int(i*samp_dur) for i in range(n)]})
df_samp_int['exp_part'] = ['p1']*int(n/3)+['p2']*int(n/3)+['p3']*int(n/3)


# for i in range(int(n)):
#     i=int(i)
#     # print(i)
#     if (i in range(int(n/3))):   
#         df_samp_int = df_samp_int.append({'duration':samp_dur, 'accumulate': int(i*samp_dur), 'exp_part':'p1'}, ignore_index=True)
#     if (i in range(int(n/3),int(2*n/3))):
#         df_samp_int = df_samp_int.append({'duration':samp_dur, 'accumulate': int(i*samp_dur), 'exp_part':'p2'}, ignore_index=True)
#     if (i in range(int(2*n/3),n)):
#        df_samp_int = df_samp_int.append({'duration':samp_dur, 'accumulate': int(i*samp_dur), 'exp_part':'p3'}, ignore_index=True)

"""

# # get the start reading time from the file name:
t_start_str_list = []
start_elite_lst_from_filename = []
for f in range(len(elite_files_list)):
    file_name = os.path.basename(elite_files_list[f])
    start_time_str = file_name.split(".") [0]
    # globals()[f"start_time_str_{f+1}"] = start_time_str
    t_start_str_list.append(start_time_str)
    start_elite_lst_from_filename.append(datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H-%M-%S"))
    print(start_time_str)
t_start_p1 = start_time_str

# //////  OR //////: get the start time from garmin data:_____________________________
gar_processed_dat = pd.read_csv(garmin_path + str(sub_ID) + '_garmin_stress.csv') 
t_start_p1 = gar_processed_dat['timestamp_6_min'][0]
p3 = gar_processed_dat.timestamp_6_min[gar_processed_dat.part=='p3'].reset_index(drop=True) 
# t_start_p3 = p3[0]
# start_elite_lst_from_garmin = [datetime.datetime.strptime(t_start_p1, "%Y-%m-%d %H:%M:%S")
#                 datetime.datetime.strptime(t_start_p3, "%Y-%m-%d %H:%M:%S")]
start_elite_lst_from_garmin = [datetime.datetime.strptime(t_start_p1, "%Y-%m-%d %H:%M:%S")]


# ////// OR /////: insert the start reading time manually:_____________________________
t_start_str_list = ['2022-03-09 12-12-00']
# convert string to datetime
start_elite_lst_manual = [datetime.datetime.strptime('2022-03-09 12-12-00', "%Y-%m-%d %H-%M-%S")]

"""                  

# >>>>>>>  start time from one of the methods: >>>>>>>>>
# t_start = start_elite_lst_from_filename
# t_start = start_elite_lst_manual

# t_start = start_elite_lst_from_garmin
# t_start = ['2022-03-09 12-12-00']
# print("\ncheck - the start time is: " + str(t_start)+", " + str(t_start[1]) )
# print("\ncheck - the start time is: " + str(t_start))

#%%
# ========================================================================== 
# ---------------------------     Functions:     --------------------------- 
# ========================================================================== 



def exp_intervals(samp_dur, break_duration, path, df_samp_int):   
   general_path = path 
   
   # samp_dur = 6  # duration of each HRV sample interval:
   # break_duration = input('insert the break duration in minutes. \n')
   # if break_duration == 0:
   #     df_samp_int = pd.read_csv(general_path+'intervals-long_version_NO_BREAK.csv')
   # else: 
   #     df_samp_int = pd.read_csv(general_path+'intervals-long_version.csv') 
   
   total_dur = pd.to_timedelta(max(df_samp_int['accumulate']) + samp_dur, unit="min")
   # supposed to return: Timedelta('0 days 01:30:00')

   # devide to 3 different parts of the experiment (relax - stress - relax): 
   parts_order = df_samp_int['exp_part'].unique()
   parts_ord = [parts_order[i] for i in range(len(parts_order))]
   parts_length = []     
   intervals = []
   for p in parts_order:
       x = df_samp_int[df_samp_int['exp_part'] == p]
       globals()[f"interval_{p}"] = x
       intervals.append(x)  # create a list of lists for the parts intervals
 
       parts_length.append(sum(df_samp_int.duration[df_samp_int['exp_part'] == p]))
                                                                                
   return parts_ord, parts_length, total_dur, intervals, df_samp_int


def create_timestamp(RR, reading_time):   
    # convert data into time difference for creating timestamps - using a Timedelta
    delta_t = pd.to_timedelta(RR, unit="ms")
    sum = reading_time
    timestamp = []
    for dt in delta_t:
        timestamp.append(sum)
        sum += dt
    timestamp.append(sum)

    return timestamp

### read the data from the text files and create data frames: 
# (for last version: there is only one RR file to process. (no break))    
def get_RR_data(): # get the start reading time from the file name:    
    t_start_str_list = []
    start_elite_lst = []
    for f in range(len(elite_files_list)):
        file_name = os.path.basename(elite_files_list[f]) # get filename
        reading_time_str = file_name.split(".") [0] # get reading start time from the filename
        # globals()[f"start_time_str_{f+1}"] = start_time_str
        t_start_str_list.append(reading_time_str)
        start_elite_lst.append(datetime.datetime.strptime(reading_time_str, "%Y-%m-%d %H-%M-%S"))
        print(start_elite_lst)
    
    RR_list = []
    RR_df_list = []
    data_edges = pd.DataFrame() 
    count = 0
    for file in elite_files_list:        
        # ***** put all RR data from 'elite' files in a list, convert str to int: *****
        RR_file = open(file, 'r')
        RR_dat = RR_file.readlines()
        RR_dat = [int(RR_dat[i]) for i in range(len(RR_dat))]      
        # globals()[f"RR{count+1}"] = RR_dat  # create a list for each data file
        # RR_list.append(RR_dat)  # create a list of of the data lists
        
        # ***** create timestamp to each RR data:  ******** 
        timestamp = create_timestamp(RR_dat, start_elite_lst[count])
            
        RR_df = pd.DataFrame(data = {'RR': RR_dat, 'timestamp': timestamp[1:len(timestamp)]})
        globals()[f"my_data{count+1}"] = RR_df
        RR_df_list.append(RR_df) 
        
        ### check the first and last timestampand- compare with the operation table:
        # print(f"\nfile {count+1}: first timestamp in the reading is: " + str(timestamp[0]))
        # print(f"\nfile {count+1}: last timestamp in the reading is: " + str(timestamp[-1]))
        data_edges = data_edges.append({'file': f"file{count+1}", 'first': timestamp[0], 'last': timestamp[-1]}, ignore_index = True) 
        
        count +=1
    print(data_edges)
    return RR_df_list, data_edges


def trim_RR_rec(t_start, total_dur):
    my_data = RR_df_list[0]
    
    t1 = t_start
    t2 =  t1 + pd.to_timedelta(total_dur, unit="min") + pd.to_timedelta(1, unit="min") 
    RR_long = [my_data.RR[i] for i in range(len(my_data)) if (t1 <= my_data.timestamp[i] <= t2)]
    timestamp_long = [my_data.timestamp[i] for i in range(len(my_data)) if (t1 <= my_data.timestamp[i] <= t2)]
    
    df_long = pd.DataFrame(data = {'RR': RR_long, 'timestamps': timestamp_long})       
    df_long = df_long.reset_index(drop=True) 
    print(df_long.head)
    
    return df_long, RR_long


def export_long_rec():
    path = elite_path + f'elite_{sub_ID}/long_rec/'
    
    if os.path.exists(path):
        print('The folder is already exist - deleting it')        
        shutil.rmtree(path) #deletes a directory and all its contents.
    
    os.makedirs(path)
    
    filepath = path + f'sub-{sub_ID}_' + str(df_long.timestamps[0].strftime("%Y-%m-%d %H-%M-%S")) 
    # save to txt (RR only):
    with open(filepath +".txt", 'w') as f:
        for item in RR_long:
            f.write("%s\n" % item) 


def RR_sample(t_start, parts_ord, parts_length, total_dur, intervals, df_samp_int):          
    # 3 different parts of the experiment (relax - stress - relax):
    # and by the 2 data files of RR:     
    RR_list = []
    time_list = []
    starts = []
    ends = []
    # remove the time data of the break:
    df_samp_int = df_samp_int[df_samp_int.exp_part!='break']  
    df_samp_int = df_samp_int.reset_index(drop=True) 
    
    #  # (first file for parts 1+2, second file for part 3):
    # seq_list = pd.concat([df_samp_int.duration[df_samp_int.exp_part=='p1'], df_samp_int.duration[df_samp_int.exp_part=='p2']])
    # seq_list = [seq_list, df_samp_int.duration[df_samp_int.exp_part=='p3']]
    
    my_data = RR_df_list[0]
       
    seq = df_samp_int
    t1 = t_start
    for x in seq.index:
        t2 =  t1 + pd.to_timedelta(seq.duration[x], unit="min")      
        RR_sample = [my_data.RR[i] for i in range(len(my_data)) if (t1 <= my_data.timestamp[i] <= t2)]
        timestamp_sample = [my_data.timestamp[i] for i in range(len(my_data)) if (t1 <= my_data.timestamp[i] <= t2)]
        
        RR_list.append(RR_sample)
        time_list.append(timestamp_sample)          
  
        starts.append(timestamp_sample[0])
        ends.append(timestamp_sample[-1])
        
        # update the start time for the next sample:
        t1 = t2
    
    # for f in range(len(RR_df_list)):            
    #     my_data = RR_df_list[f]    
    #     seq = seq_list[f] 
    #     t1 = t_start[f]        
    #     for x in seq.index:
    #         t2 =  t1 + pd.to_timedelta(seq[x], unit="min")    
    #         RR_sample = [my_data.RR[i] for i in range(len(my_data)) if (t1 <= my_data.timestamp[i] <= t2)]
    #         timestamp_sample = [my_data.timestamp[i] for i in range(len(my_data)) if (t1 <= my_data.timestamp[i] <= t2)]
            
    #         RR_list.append(RR_sample)
    #         time_list.append(timestamp_sample)          
  
    #         starts.append(timestamp_sample[0])
    #         ends.append(timestamp_sample[-1])
            
    #         # update the start time for the next sample:
    #         t1 = t2
                      
    all_samp_time = df_samp_int.accumulate[df_samp_int.exp_part!='break']
    all_parts = df_samp_int.exp_part[df_samp_int.exp_part!='break']
    
    my_data_samp = pd.DataFrame(data = {'RR': RR_list, 'timestamps': time_list, 'samp_time': all_samp_time , 'part': all_parts})       
    my_data_samp = my_data_samp.reset_index(drop=True) 
    print(my_data_samp.head)
    
    # ceck the samples borders:
    samp_borders = pd.DataFrame(data = {'start': starts, 'end': ends , 'part': all_parts})
    samp_borders['samp_length'] = samp_borders['end']-samp_borders['start']       

    return my_data_samp, RR_list, time_list, samp_borders


def export_RR_samp(my_data_samp, RR_list, time_list, samp_dur):
    lst_tables = []
    # open a new folder for the samples:
    folder_name = f'elite_{sub_ID}/samples_RR_{sub_ID}_{samp_dur}/'    
    if os.path.exists(elite_path + folder_name):
        print('The folder is already exist')
    else:
        os.makedirs(elite_path + folder_name) 
    
    for s in range(len(RR_list)):
        # sample = my_data_samp.loc[s]
        time_samp = time_list[s]
        RR_samp = RR_list[s]
               
        filepath = elite_path + folder_name + str(time_samp[0].strftime("%Y-%m-%d %H-%M-%S"))                       
        
        # save to txt (RR only):
        with open(filepath +".txt", 'w') as f:
            for item in RR_samp:
                f.write("%s\n" % item)    
       
        # save a table to csv:
        table = pd.DataFrame(data = {'RR': RR_samp, 'timestamps': time_samp})               
        # table.to_csv(filepath + ".csv", index=False)
        lst_tables.append(table)
    
    print('samples are exported.')
    
    return lst_tables

    
###################################################################################################
#%%






# ==========================================================================
# ---------------- main code: ----------------------------  
# ==========================================================================

# create the general time frame by the experiment setup:
parts_ord, parts_length, total_dur, intervals, df_samp_int = exp_intervals(samp_dur, break_duration, general_path, df_samp_int)

for sub in sublist:    
    sub_ID = sub
    # import data:
    e_subfolder_path = elite_path + "elite_" + str(sub_ID) + "/"     
    elite_files_list = glob.glob(e_subfolder_path +'*.txt')
    print("elite file to smaple: \n" + str(elite_files_list[0:2]))

    # # create a folder for kubios results:
    # kub_folder_path = "C:/users/hadar/OneDrive/Documents/MASTER/2-RESEARCH/experiment_validation/kubios_results/results_"+str(sub_ID) 
    # os.makedirs(kub_folder_path)

    # set sample time:    
    t1_ind = t_table[t_table.sub_ID == str(sub_ID)].index.tolist()
    
    if len(t1_ind) == 0:
        continue
    
    t_start = datetime.datetime.strptime(str(t_table.synched_start_time[t1_ind[0]]), "%d/%m/%Y %H:%M")
    
    print(f'\nstart time of this recording is: {t_start}')
          
    total_borders = pd.Series(data=[t_start, t_start + total_dur], index=['start', 'end'])

    print('parts ord:', parts_ord,'\nparts length:',parts_length,'\ntotal dur:',total_dur,'\ntotal_borders:\n', total_borders)
    # print('make sure that the sample intervals fit the operation table:  !!!!!!!')
    
    ### read the data from the text files and create data frames with timestamps: 
    RR_df_list, data_edges = get_RR_data()
    
    df_long, RR_long = trim_RR_rec(t_start, total_dur)
    export_long_rec()
    print(f'\nSubject {sub_ID}: long recording has been exported.')
                                            
    my_data_samp, RR_list, time_list, samp_borders = RR_sample(t_start, parts_ord, parts_length, total_dur, intervals, df_samp_int)
    lst_tables = export_RR_samp(my_data_samp, RR_list, time_list, samp_dur)
    print("\ngreat! finished sampling, check the folder to verify the txt files.")

    filepath = elite_path + "elite_"+str(sub_ID)+"/"
    samp_borders.to_csv(filepath + "samp_borders.csv", index=False)


# to check the parts, break and length...:
# parts_length... : [30, 30, 0, 30]
# parts_ord ... : ['p1', 'p2', 'break', 'p3']
# total_dur supposed to be: Timedelta('0 days 01:33:00')
# samp_intervals = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 63, 69, 75, 81, 87] # new long version of 1.5 hr 


############################    the end    ##################################

#%%
"""
dst_dir = elite_path + 'all_subs_long_rec/'
if os.path.exists(dst_dir):
        print('The folder is already exist - deleting it')        
        shutil.rmtree(dst_dir) #deletes a directory and all its contents.    
os.makedirs(dst_dir)

print ('Named explicitly:')
for name in glob.glob(elite_path+'*/long_rec/*'):    
    if name.endswith(".txt"): 
        shutil.copy(name, dst_dir)
        print ('\t', name)


"""










#%%

#     my_data_samp_T = my_data_samp.T  # transpose dataframe

# # remove the time data of the break:
# parts_borders.drop(['break'], axis=0, inplace=True)

# # parts_ord.pop(parts_ord.index('break'))  # remove the break from the parts list 

 
# samp_dur = pd.to_timedelta(samp_duration, unit="min")
 
# samples = []
# for d in range(len(RR_df_list)):
#     for i in range(len(elite_slice)):
#         t1 = elite_slice.start[i]
#         t2 = elite_slice.end[i]
#         # if (t1 <= RR_df_list[d].timestamp.iloc[0] <= t2):
#         # if (t1 >= RR_df_list[d].timestamp.iloc[0]) & (t1 <= RR_df_list[d].timestamp.iloc[-1]) :
#         mask = (RR_df_list[d]['timestamp'] >= t1) & (RR_df_list[d]['timestamp'] < t2)
#         samp = RR_df_list[d].loc[mask]            
#         if len(samp) != 0 :       
#             samples.append(samp)
#             print('sampling.')
#         else: 
#             print('not contain this time interval, not sampling.')
    
# check = [samples[x]['timestamp'].iloc[0] for x in (range(len(samples)))  ]
# print(len(check))

# count=0
# for p in part_ord:
#     interval = f"interval_{p}"
#     my_data = my_data1
# mask = (my_data['timestamp'] >= parts_borders.start['p1']) & (my_data['timestamp'] < parts_borders.end['p1'])
#     # print(df.loc[mask])
#     df_p1 = df.loc[mask1]    


#########################################################################


##### fun 3 : 
# def cal_slice(t_start, duration):
#     for i in range(len(sample_intervals)-1):
#         next_start = start_sampling[i] + timedelta(minutes = sample_intervals[i])
#         start_sampling.append(next_start)
        
#     samp_index = []
#     for dur in range(0, len(sample_intervals)) :
#         if sample_intervals[dur] == 6:
#             samp_index.append(dur)
#     start_sampling = [start_sampling[i] for i in range(len(start_sampling)) if (sample_intervals[i]==6)]  
     
#     end = [start_sampling[i] + timedelta(minutes=duration) for i in range(len(start_sampling))]
    
#     df_slice = pd.DataFrame(data = {'start': start_sampling, 'end' : end})
    
#     return df_slice


# def cal_slice_manual(starts, duration):
#     # need to change to fit the new long version:
#     starts = [datetime.datetime.strptime(manual_time[i], "%Y-%m-%d %H:%M:%S") for i in range(len(manual_time))]
#     # build the start time list y:
#     start_sampling = []
#     start_sampling.insert(0, starts[0])
#     start_sampling.insert(1, starts[0]+timedelta(minutes=duration))
#     start_sampling.insert(2, starts[0]+timedelta(minutes=duration*2))
#     start_sampling.insert(3, starts[0]+timedelta(minutes=duration*3))
#     start_sampling.insert(4, starts[1])
#     start_sampling.insert(5, starts[1]+timedelta(minutes=duration))
#     start_sampling.insert(6, starts[2])
#     start_sampling.insert(7, starts[2]+timedelta(minutes=duration))
#     # # duration = [str(duration[i]) for i in range(len(duration))]  
#     # duration = [datetime.datetime.strptime(duration[i], "%M") for i in range(len(duration))]  
#     # delta_t = pd.to_timedelta(duration, unit = "m")
#     ends = [start_sampling[i] + timedelta(minutes=duration) for i in range(len(start_sampling))]
#     df_slice = pd.DataFrame(data = {'start': start_sampling, 'end' : ends})

#     return (df_slice)

#########################################################################


# def sum_sec(data, start = 0):
#     sum = start
#     sec = []
#     for d in data:
#         sec.append(sum)
#         sum += d/1000
#     sec.append(sum)

#     return sec


# seconds = sum_sec(data)
# seconds = [str(seconds[i]) for i in range(len(seconds))]   
# seconds = [datetime.datetime.strptime(time_str[i], "%S") for i in range(len(seconds))]   

# duration = [str(duration[i]) for i in range(len(duration))]  
# duration = [datetime.datetime.strptime(duration[i], "%M") for i in range(len(duration))]  
# delta_t = pd.to_timedelta(duration, unit = "m")

# parts_ord.pop(parts_ord.index('break')) 


# def sampling(my_data, df_slice):
#     RR_list = []
#     time_list = []
#     l = len(df_slice.start)
#     for s in range(l):
#         t1 = df_slice.start[s]
#         t2 = df_slice.end[s]
#         RR_sample = [my_data.RR[i] for i in range(len(my_data)) if (t1 < my_data.timestamps[i] <= t2)]
#         timestamp_sample = [my_data.timestamps[i] for i in range(len(my_data)) if (t1 <= my_data.timestamps[i] <= t2)]
#         RR_list.append(RR_sample)
#         time_list.append(timestamp_sample)
        
#         df = pd.DataFrame(data = {'RR': RR_sample, 'timestamps' : timestamp_sample})       
#         folder_name = "elite_"+str(sub_ID)+"/"+"samples_RR_"+str(sub_ID)+"/"
#         filename = folder_name + str(timestamp_sample[0].strftime("%Y-%m-%d %H-%M-%S"))
        
#         timestamp_sample = str(timestamp_sample)
        
#         # open a new folder for the samples:
#         os.makedirs(subfolder_path + "samples_RR_" + str(sub_ID))
#         # save to csv:
#         # df.to_csv(filename+".csv", index=False)       
#         # save to txt (RR only):
#         with open(filename +".txt", 'w') as f:
#             for item in RR_sample:
#                 f.write("%s\n" % item)

#     return RR_list, time_list


#########################################################################
# # to fix the test data (synthesized- sub 999)
# # if there is overlaping between the data files:
# mask1 = (my_data1['timestamp'] >= parts_borders.start['p1']) & (my_data1['timestamp'] <= parts_borders.end['p2'])
# my_data1 = my_data1.loc[mask1]  
# RR_1 = [my_data1.RR[i] for i in range(len(my_data1))]
# mask2 = (my_data2['timestamp'] >= parts_borders.start['p3']) & (my_data2['timestamp'] <= parts_borders.end['p3'])
# my_data2 = my_data2.loc[mask2]
# RR_2 = [my_data2.RR[i] for i in range(len(my_data2))]

# RR_df_list = [RR_1, RR_2] 

# folder_name = "elite_"+str(sub_ID)+"/"+"samples_RR_"+str(sub_ID)+"/"
# filename = [start_time_str_1 , start_time_str_2]   
# # save to txt (RR only):
# for i in range(len(RR_df_list)):
#     with open(elite_path + folder_name + filename[i] +".txt", 'w') as f:
#         f.write("%s\n" % RR_df_list[i])
              
