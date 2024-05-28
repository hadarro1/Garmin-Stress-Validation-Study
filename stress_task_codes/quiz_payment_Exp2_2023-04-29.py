# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 09:34:37 2021

@author: AlonItzko
"""
import glob
import datetime as dt

path = 'quiz_results_folder/'
scores_files = glob.glob(path+'*sum.csv')

date2time = lambda s: dt.datetime(int(s[:4]), int(s[5:7]), int(s[8:10]), 
                      int(s[11:13]), int(s[14:16]), int(s[17:]))

dates = []
quiz_max = 0
for f in scores_files:
    score = open(f, 'r')
    score = score.readlines()[1].split(',')
    date = score[5]
    dates.append(date2time(date))
    score = score[-3:]
    score = max([int(s) for s in score])
    if score > quiz_max:
        quiz_max = score

sub_file = scores_files[dates.index(max(dates))]
sub_score = open(sub_file, 'r')
sub_score = sub_score.readlines()[1].split(',')
sub_id = sub_score[0]
sub_scores = sub_score[-3:]
sub_max = max([int(s) for s in sub_scores])

extra = 20 if sub_max >= quiz_max else 15 if sub_max > 240 else 10 if sub_max > 170 else 0
dur_meet2 = float(input("second meeting duration ( 1.25 or 1.5 or 1.75 hours):____  "))
meet1_pay = 40
pay_for_time = int(meet1_pay + dur_meet2*40)

print(f'## sub ID = {sub_id} ##\n\nmax score = {quiz_max}\nsub max = {sub_max}\n\npay for time = {pay_for_time} NIS\nBonus = {extra} NIS\n\n')
print(f'## Total Payment = {pay_for_time+extra} NIS ##')
