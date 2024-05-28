# -*- coding: utf-8 -*-
"""
Created on Fri Aug 13 14:52:46 2021

@author: hadar
"""
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
# import re
import os
import datetime
from datetime import timedelta  
import glob
# import statistics as stat
############## time calculations: (general for both data sources)  ############

def exp_intervals(samp_duration, break_duration, path):   
   general_path = path 
   
   # samp_duration = 6  # duration of each HRV sample interval:
   # break_duration = input('insert the break duration in minutes. \n')
   if break_duration == 0:
       df_samp_int = pd.read_csv(general_path+'intervals-long_version_NO_BREAK.csv')
   else: 
       df_samp_int = pd.read_csv(general_path+'intervals-long_version.csv') 
   
   total_dur = pd.to_timedelta(max(df_samp_int['accumulate']) + samp_duration, unit="min")
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


   # samp_intervals = [0,6,15,21,30,36,42]  # old version of(6,6,3,6,6,3,6,6,6)    
   # samp_intervals = [0, 6, 12, 18, 24, 30, 36, 42, 48, 54, 60, 63, 69, 75, 81, 87, 93] # new long version of 1.5 hr 

   # exp_parts = [sum(df_samp_int.duration[df_samp_int['exp_part'] == 'p1']), sum(df_samp_int.duration[df_samp_int['exp_part'] == 'p2']), sum(df_samp_int.duration[df_samp_int['exp_part'] == 'p3'])]   
   # exp_parts = [12,3,12,3,18]  # (relax,break, stress,break,relax)
   # exp_parts = [30,30,3,30]  # (relax, stress,break,end_relax)

   # interval_p1 = df_samp_int.accumulate[df_samp_int['exp_part'] == 'p1'] 
   # interval_p2 = df_samp_int.accumulate[df_samp_int['exp_part'] == 'p2'] 
   # interval_p3 = df_samp_int.accumulate[df_samp_int['exp_part'] == 'p3'] 
   
   # interval_list = df_samp_int.duration.to_list()
   # idx_break = interval_list.index(3)
   # idx_break = df_samp_int.index[df_samp_int['exp_part']=='break']

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


def set_borders(t_start, parts_ord, parts_length, total_dur, intervals):
    starts = []
    ends = []
    # devide to 3 different parts of the experiment (relax - stress - relax):
    for p in range(len(parts_ord)): 
        intervals[p] = intervals[p].reset_index(drop=True)
        # name = parts_ord[p]
        # globals()[f"start_{name}"] = t_start + pd.to_timedelta(intervals[p][0], unit="min")  
        starts.append(t_start + pd.to_timedelta(intervals[p].accumulate[0], unit="min"))

        e = len(intervals[p])-1
        # globals()[f"end_{name}"] = t_start + pd.to_timedelta(intervals[p][e]+samp_duration, unit="min") 

        ends.append(t_start + pd.to_timedelta(intervals[p].accumulate[0]+parts_length[p],unit="min"))
                                              
    
    # parts_ord.pop(parts_ord.index('break'))  # remove the break from the parts list    
    parts_borders = pd.DataFrame(data={'start': starts , 'end': ends},index=parts_ord)
                                 
    return parts_borders


    # p1_start = t_start
    # p2_start = t_start + pd.to_timedelta(interval_p2[0], unit="min")
    # p3_start = t_start + pd.to_timedelta(interval_p3[0], unit="min")
    
    # dur_p1 = pd.to_timedelta(interval_p1[-1]+6, unit="min")
    # dur_p2 = pd.to_timedelta(interval_p2[-1]+6-interval_p2[0], unit="min")
    # dur_p3 = pd.to_timedelta(interval_p3[-1]+6-interval_p3[0], unit="min")
    
    # p1_end = p1_start + dur_p1
    # p2_end = p2_start + dur_p2
    # p3_end = p3_start + dur_p3

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
  
# data_frame.reset_index(drop=True)
