# -*- coding: utf-8 -*-
"""
Created on Sat Aug  7 12:35:23 2021

@author: hadar

Short EXPERIMENT VERSION (15-15-15): 
this code subset the stress data of garmin to the specific time series of the experiment 
and can calculate the average stress for 6 minutes time interval or keep it in 3-minute intervals
"""
invalids = [int(i) for i in['403','404', '406', '409','412','413', '414', '415', '418', '429', '431', '432', '433', '435', '439', '445', '446', '454', '457','458','459', '463', '464','472','474','475','480']]

#%%
# runfile('fun_data_organize.py', wdir='C:/Users/hadar/OneDrive/Documents/MASTER/2-RESEARCH/GarminValidation_Exp2/Data Analysis/Codes')

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import re
import os
import datetime
from datetime import timedelta  
import glob
import statistics as stat
from fun_data_organize import * # my module

general_path = "C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2/Data_Analysis/Codes/"
garmin_path = "C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2/Data_Analysis/Data/garmin_data/"
# samp_dur = 6
samp_dur = 3
break_duration = 0
exp_min = 45
# set the sub_ID :
# sub_ID = int(input( "Enter subject ID:\n  "))
# sub_ID = 303
# sub_ID = [101:111]

#%%
# get a list of sub_id from the garmin data folder
sublist = glob.glob(garmin_path+'*_garminStressDetails_*') 
sublist = [s.split('\\')[-1].split('_')[0] for s in sublist]
print(sublist)
len(sublist)

#################################
# to run a specific subject data:
sublist = [488,489,490]
#################################

# get start time from the sub_info table of all the particpants
sub_info = pd.read_csv('C:/users/hadar/OneDrive/Documents/3_MASTER/2-RESEARCH/GarminValidation_Exp2/Data_Analysis/Data/sub_info2.csv')
# sub_info.columns
t_table = sub_info[['Participant_number','synched_start_time']]
t_table = t_table.iloc[1:]
t_table.rename(columns={ 'Participant_number': 'sub_ID' }, inplace = True)
# time_file = open(general_path+'start_time_to analyze.csv', 'r')
# t_table = pd.read_csv(time_file)
# t_table = t_table[["sub_ID", "synched_start_time"]]

print(t_table)


# count how many invalid subjects:
print(t_table['synched_start_time'].isna().sum())
print(t_table['synched_start_time'].isna())

invalid_lst = t_table[t_table['synched_start_time'].isna()]['sub_ID']
invalid_lst = [str(x) for x in invalid_lst]
invalid_lst

len(t_table)
# to get rid of invalid participants

t_table = t_table.dropna() 
print(t_table)
print(f'Number of valid datasets: {len(t_table)}')

# update sublist by removing invalid subjects:
sublist = [s for s in sublist if s not in invalid_lst]
sublist
# if there are subjects that have a stress data file but need to be excluded manually from analysis:

sublist = [s for s in sublist if s not in invalids]
len(sublist)

#%%
# create a dataframe of samples intervals- order and parts

df_samp_int = pd.DataFrame(columns=['duration', 'accumulate', 'exp_part'])
n = int(exp_min/samp_dur)
df_samp_int = pd.DataFrame({'duration': [samp_dur for _ in range(n)] , 'accumulate': [int(i*samp_dur) for i in range(n)]})
df_samp_int['exp_part'] = ['p1']*int(n/3)+['p2']*int(n/3)+['p3']*int(n/3)


# t1_ind = t_table[t_table.sub_ID==str(sub_ID)].index.tolist()
# t_start = datetime.datetime.strptime(str(t_table.synched_start_time[t1_ind[0]]), "%d/%m/%Y %H:%M")

"""
to enter start time manually...
"""
# text = """"check what is the starting time of the HRV recording,
# that match Garmin data interval.
# type in this format:  '2021-01-30 01:01:00' """
# start_time_str = input(text+ ":   ")
# # start_time_str = '2021-12-28 12:33:00'
# # convert string to datetime:
# t_start = datetime.datetime.strptime(start_time_str, "%Y-%m-%d %H:%M:%S")
# print("\ncheck - the start time is: " + str(t_start))



# garmin data:
# import the stress data file:
# garmin_files = glob.glob(garmin_path + str(sub_ID) + '_garminStressDetails*.csv')
# garmin_file = garmin_files[0]
# print("garmin file to smaple: " + garmin_file)


# -------- Functions: ------------------------------------  
# ________________________________________________________
#%%

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


def prepare_data():
   ### read the data from the text file: 
   f = open(garmin_file, 'r')
   df = pd.read_csv(f)
   print(df.head())
   #choose only the 2 relevante columns:

   df = df[["ActivityDate", "StressLevelValue"]]
   # to simplify the column names:
   df = df.rename(columns={"ActivityDate": "timestamp", "StressLevelValue": "stress"})
   # stress_data_list = [int(df.stress[i]) for i in range(len(df.stress))]   
   
   if (df["timestamp"][0][-1] == 'M' or df["timestamp"][0][-1] == 'm'):
   # function to convert the time string to datetime format:
       format = '%m/%d/%Y %I:%M %p' # The format of the time series given
   else:      
       format = '%m/%d/%Y %H:%M' # The format of the time series given

   df.timestamp = [datetime.datetime.strptime(df.timestamp[i], format) for i in range(len(df.timestamp))]
   # replacing all missing data point of "-1", "-2"  with nan
   df = df.replace(-1 , np.NaN)  
   df = df.replace(-2 , np.NaN) 
       
   return df



# create the reading time series (cut the relevante data from the full data):
def cut_data():    
    mask = (df['timestamp'] >= total_borders[0]) & (df['timestamp'] < total_borders[1])
    # print(df.loc[mask])
    my_data = df.loc[mask]    
    my_data = my_data.set_index('timestamp')
   
    return my_data

 
def check_missing_points(my_data):
    print("\nsubject: "+str(sub_ID))
          
    min_3 = pd.to_timedelta(3, unit="min")
    c=0
    for i in range(len(my_data)-1):
        delta = df.timestamp[i+1]-df.timestamp[i]     
        if delta != min_3 :
            print("\nthe timeseries is not complete - fix it on index "+ str(i) +", on time point: "+ str(df.timestamp[i])+'\n')
            c+=1
    if (c==0):
        print("Data is OK, no missing timestamps.\n")
    else:
        print("\nthere are " +str(c)+ " missing timestamps.\n")
        
    # check for missing data (-1 or -2 that are NA now): 
    stress_data_points = len(my_data['stress'])
    na_num = my_data['stress'].isna().sum()
    na_percentage = "{:.0%}".format(na_num/stress_data_points)
    if na_num == 0:
        print('no missing stress points\n')
    else:    
        print('\nmy_data contain ', str(na_num), 'missing data points(NA), \nout of ', str(stress_data_points), 'stress data point.\n'
           'which is: ',na_percentage,' of the stress scores are missing.\n'
          , 'total of '+ str(stress_data_points-na_num), 'scores were recorded.\n')
    
    return na_num, na_percentage
 

def slice_parts(parts):
    lst_df_parts = []
    # parts_ord.remove("break")
    for p in parts:
        mask = (df['timestamp'] >= parts_borders.start[p]) & (df['timestamp'] < parts_borders.end[p])
        df_p = df.loc[mask]    
        df_p = df_p.set_index('timestamp')       
        lst_df_parts.append(df_p)       
    return lst_df_parts


# calculate the mean stress for each 6 minutes intervals (for 2 stress values):
def cal_6_min_means(lst_df_parts):
    for p in range(3):
        # list_parts = [df_p1, df_p2, df_p3]        
        x = lst_df_parts[p][::2]      
        timestamp_6_min = x.index.to_list()
        mean_stress = []
        count = 0
        for ind in range(len(x)):    
            pair = lst_df_parts[p].stress.iloc[count:count+2]
            # mean_stress.append(stat.mean(pair)) 
            mean_stress.append(np.nanmean(pair))
             
            count += 2
        
        p_mean = pd.DataFrame(data = {'timestamp_6_min': timestamp_6_min, 'mean_stress' : mean_stress})   
        globals()[f"p{p+1}_means"] = p_mean
    
    # add part category:
    p1_means["part"]="p1"
    p2_means["part"]="p2"
    p3_means["part"]="p3"
    
    # combine all data to one dataframe and reset the indexes:
    df_garmin_6 = pd.concat([p1_means, p2_means, p3_means])
    df_garmin_6.reset_index(drop=True)

    return df_garmin_6


#%%    
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# -------- main code: ------------------------------------  
# ________________________________________________________    

exported_subs=[]
# sub=469

for sub in sublist:
    sub_ID = sub
    print(sub_ID)
    t1_ind = t_table[t_table.sub_ID==str(sub_ID)].index.tolist()
    t_start = datetime.datetime.strptime(str(t_table.synched_start_time[t1_ind[0]]), "%d/%m/%Y %H:%M")
    
    garmin_file = glob.glob(garmin_path + str(sub_ID) + '_garminStressDetails*.csv')[0]
   
    # import garmin data and check if you need to fix the start time:
    df = prepare_data()
    # to check the format and the borders of the data:
    df_borders = pd.concat([df.head(1),df.tail(1)])
    print(df_borders)
    type(df.timestamp[0])
    df.head(5)
    df.tail(5)
    
    # # check for the start time fix:    
    # t_start_new = fix_start_time(df, t_start)
    t_start_new=t_start
    # create the general time frame by the experiment setup:
    parts_ord, parts_length, total_dur, intervals, df_samp_int = exp_intervals(samp_dur, break_duration, general_path, df_samp_int)
    total_borders = pd.Series(data=[t_start_new, t_start_new + total_dur], index=['start', 'end'])
    print(total_borders)
    # print('\ncheck the experiment part division and the interval order (breaks and readings..):'
    #       '\n does it match the exp setup?', parts_ord)
    
    # create the time frame for this data set: 
    parts_borders = set_borders(t_start_new,parts_ord, parts_length, total_dur, intervals)
    print(parts_borders)
    
    # remove the time data of the break:
    # parts_borders.drop(['break'], axis=0, inplace=True)
    # print(parts_borders)
    
    ##==========================================================================
    # manually chang parts borders because of synch problem with polar reading:
    #==========================================================================
    #     for i in range(len(parts_borders.start)):
    #     parts_borders.iloc[i, 0] =  parts_borders.iloc[i, 0]+pd.to_timedelta(3, unit="min") 
    # for i in range(len(parts_borders.end)):
    #     parts_borders.iloc[i, 1] =  parts_borders.iloc[i, 1]-pd.to_timedelta(3, unit="min") 
    # print(parts_borders)
    #==========================================================================
    
    # create the data cut I need:
    my_data = cut_data()
    print(my_data)
    my_data
    # check if there are no missing timestamps :
    na_num, na_percentage = check_missing_points(my_data)
    
    ################### slice the data to the 3 parts of the exp: ##################
    lst_df_parts = slice_parts(parts_ord)
    
    # count missing points for every part:
    dict_missing = dict()
    for p in range(len(lst_df_parts)):
        na_num_part, na_percentage_part = check_missing_points(lst_df_parts[p])
        dict_missing[f'p{p+1}']=[na_num_part,na_percentage_part]
    df_missing = pd.DataFrame(dict_missing).T
    df_missing.columns = "number_of_missing_stress_points", "percentage"
    print(df_missing)
    
    
    if samp_dur == 6:    
        # calculate the average stress for each 6 min (for every 2 stress value):
        df_garmin_6 = cal_6_min_means(lst_df_parts)
        df_garmin_stress = df_garmin_6
        
    else:
        lst_df_parts = []
        timestamp_3 = my_data.index.to_list()
        mean_stress =  my_data.stress.to_list()
        df_garmin_3 = pd.DataFrame(data = {'timestamp_3': timestamp_3, 'mean_stress' : mean_stress})
        for p in parts_borders.index:
            mask = (df_garmin_3.timestamp_3 >= parts_borders.start[p]) & (df_garmin_3.timestamp_3 < parts_borders.end[p])
            df_p = df_garmin_3.loc[mask] 
            df_p['part'] = p
            lst_df_parts.append(df_p)
        df_garmin_3 = pd.concat([lst_df_parts[0], lst_df_parts[1], lst_df_parts[2]])
        df_garmin_stress = df_garmin_3
    
    
    # df_part_mean_stress = df_garmin_stress.groupby('part').mean()
    # df_part_std = df_garmin_stress.groupby('part').std()
    df_parts = df_garmin_stress.groupby('part').agg(mean_stress=('mean_stress', 'mean'), std_stress=('mean_stress', 'std'))
    df_parts = pd.concat([df_parts,df_missing], axis=1)
    total_row = [np.mean(df_garmin_stress.mean_stress),
                           np.std(df_garmin_stress.mean_stress),
                           na_num, na_percentage]
    
    df_parts.loc['sum'] = total_row
    df_sum_total = df_parts
    print(df_sum_total)
    
    ####################### export the table to csv file:#########################
    # save to csv:
    
    df_garmin_stress.to_csv(garmin_path+'/garmin_processed_data/'+str(sub_ID)+f'_garmin_stress_{str(samp_dur)}'+".csv", index=False)       
    exported_subs.append(sub_ID)
    # df_sum_total.to_csv(garmin_path+str(sub_ID)+'_part_summary'+".csv", index=True)       


############################    the end    ##################################



# ****************************************************************************
# print(my_data.shape[0]) # to know the dimension of a dataframe

# # create the dataframe for sampling the data:   
# def sample_frame():
#     samp_delta = pd.to_timedelta(samp_intervals, unit="min")
#     t_starts = []
#     for i in range(len(samp_delta)):
#         t_starts.append(t_start+samp_delta[i])    
#     t_ends = [t_starts[i] + pd.to_timedelta(samp_dur, unit="min") for  i in range(len(t_starts))]

#     sampling = pd.DataFrame(data = {'start': t_starts, 'end' : t_ends}) 

#     return sampling

# def sample_parts():
#     p1_deltas = pd.to_timedelta(interval_p1, unit="min")
#     t_starts = []
#     for i in range(len(p1_deltas)):
#         t_starts.append(t_start+p1_deltas[i])    
#     t_ends = [t_starts[i] + pd.to_timedelta(samp_dur, unit="min") for  i in range(len(t_starts))]
#     sampling_p1 = pd.DataFrame(data = {'start': t_starts, 'end' : t_ends}) 

#     return sampling_p1


# p1_mean =df_p1.resample("6min").mean()
# p2_mean =df_p2.resample("6min").mean()
# p3_mean =df_p3.resample("6min").mean()

# index = pd.array(range(1,len(df_garmin_stress)+1), dtype=int)
# df_garmin_stress["index"]=index

# df_relax = pd.DataFrame(index=['timestamp'], columns=['stress'])
# for i in range(len(sampling.start)):
#     start = sampling.start[i]
#     end = sampling.end[i] 
#     samp_data = my_data.loc[(my_data.index >= start) & (my_data.index  < end)]
#     if i == 0:
#        df_relax = samp_data
#     else: 
#        df_relax = df_relax.append(samp_data)
    

# ****************************************************************************

    # delta = pd.to_timedelta(samp_intervals, unit="min")
#     sum = reading_time
#     timestamps = []
#     for dt in delta_t:
#         timestamps.append(sum)
#         sum += dt
#     timestamps.append(sum)


# ts1 = pd.date_range("2021-07-14 11:54:00", periods=60, freq="min") # create a time series of 1 min intervals of 30 time points
# ts3 = pd.date_range("2021-07-14 11:54:00", periods=20, freq="3min") # create a time series of 1 min intervals of 30 time points
# ts6 = ts1[::6]


# idx = pd.date_range("2021-07-14 11:54:00", periods=17, freq="3min")
# # idx = my_data.timestamp
# print(len(idx))
# ts = pd.Series(range(len(idx)), index=idx)
# stress = pd.Series(my_data.stress, index=idx)
# print(len(my_data.stress))
# my_sample_6_mean =ts.resample("6min").mean()

# ****************************************************************************

# my_series = pd.Series(data=df.timestamp, index=[2,20,2])
# data = np.array(['a', 'b', 'c', 'd', 'e'])
# s = pd.Series(data, index =[1000, 1001, 1002, 1003, 1004])
# my_data[0:17:2]
# len(my_data)

# for i in range(0, 9):
#     globals()[f"my_variable{i}"] = f"Hello from variable number {i}!"


# print(my_variable3)
# Hello from variable number 3!

# ****************************************************************************

# foldername = str(sub_ID)+"_garmin_stress_sample/"
# subgarmin_path = garmin_path + foldername      
# os.makedirs(garmin_path + foldername)


# check for updating the start time to the garmin data:
def fix_start_time(df, t_start):
    t1 =  t_start-pd.to_timedelta(6, unit="min")
    t2 =  t_start+pd.to_timedelta(6, unit="min")
    mask = (df['timestamp'] >= t1) & (df['timestamp'] <= t2)
    data_check = df.loc[mask]    
    # data_check = data_check.set_index('timestamp')
    print('\nthe data around the original start time is: \n', data_check)
    gap = pd.to_timedelta(1, unit="min")
    for t in data_check.timestamp:
        # if (pd.to_timedelta(0, unit="min") <= (t - t_start) <= gap):
        if (abs((t - t_start)) <= gap):
            if(t_start==t):
                print("no change in start time: ", t_start)
            else:
                print('the closest start time is: ', t)
            x = t
            
    return x