import random
import datetime
import time
import csv
import os
import pandas as pd
from pandas import DataFrame
import glob
import math
import numpy as np



# this function calculate time limit for each question level for a specific subject (= 90% of average RT of the specific dufficulty level)
def get_timelimits(sub_id): 
    path = 'quiz_results_folder/'
    details1 = open(f'{path}{sub_id}_details.csv', 'r')
    data = pd.read_csv(details1)
    
    mean_rt_lst=[]
    for l in range (1,6):  
        m = data['reaction_time'].loc[(data['level'] == l) & (data['feedback'] == True)].mean()
        mean_rt_lst.append(m)
        
    for i in range(len(mean_rt_lst)):   
        if math.isnan(mean_rt_lst[i]):  
            if i == 0:  # (level 1)
               mean_rt_lst[i] = mean_rt_lst[i+1]-1 # use the RT of level 2 and reduce 1 sec.
            elif i == 4:
               mean_rt_lst[i] = mean_rt_lst[i-1]+1 # use the RT of level 4 and add 1 sec. 
            else:
               mean_rt_lst[i] = (mean_rt_lst[i-1]+mean_rt_lst[i+1])/2 # use the average of RT of lower and higher level.  
               
    print('mean_rt_lst: ' , mean_rt_lst)
    
    timelimits = np.asarray(mean_rt_lst) * 0.9
    timelimits = [round(timelimits[i],0) for i in range(len(timelimits))]
    # timelimits = [int(timelimits[i]) for i in range(len(timelimits)) if math.isnan(timelimits[i]) == False]

    print('timelimits: ' , timelimits )


    return(timelimits)


def get_max_val():
    path = 'quiz_results_folder/'
    scores = glob.glob(path+'*sum.csv')
    max_score = 0
    for s in scores:
        df = pd.read_csv(s).dropna(axis=0, how='all')
        score = df[['score_trial_A','score_trial_B','score_trial_C']].max().max()
        if int(score) > max_score:
            max_score = int(score)
    print("the highest trial score in the results folder is:" + str(max_score))
    return max_score


#%%
# check if time is up and if not create a question:
# choose randomly a form of question and call the relevant function to create the question. returns the solution
def ask_question():

    l = list(range(1, 6)) # types of questions (5 difficulty levels)
    level = random.choice(l)
    if level == 1:                        
        solution, question_txt = fun_q1()   #  x + y - z = ?
    elif level == 2:                      
        solution, question_txt = fun_q2()   #  x*y - (z) = ?
    elif level == 3:  #  x*y-z*y+w = ?
        solution, question_txt = fun_q3()   #  x + y*z - w = ?
    elif level == 4:  #  x*y-z*y+w = ?
        solution, question_txt = fun_q4()   #  (x-y)*z - w = ? 
    elif level == 5:  #  x*y-z*y+w = ?
        solution, question_txt = fun_q5()   #  x*y - z*y - w = ?

    print(f'level {level}')
    print(f'solution = {solution}')
    
    return solution, question_txt, level


def check_answer(ans, correct_solution):
    if ans == correct_solution:
        feedback = True
    else:
        feedback = False
    return feedback


def check_reaction_time(t_start):
    t_end = time.time()
    reaction_time = t_end - t_start
    return reaction_time

#%%
# create a question in the form:  x + y - z = ?
def fun_q1():  # difficulty level 1
    x = random.randint(3, 10)
    y = random.randint(1, 10)
    z = random.randint(1, x-2)
    
    solution = x + y - z
    question_txt = (" {} + {} - {} = ?".format(x, y, z))
    print(question_txt)

    return solution, question_txt


# create a question in the form:  x * y - (z) = ?
def fun_q2(): # difficulty level 2
    x = random.randint(1, 10)
    y = random.randint(1, 10)
    
    z = random.randint(x*y-10, x*y)
           
    solution = x * y - z
    question_txt = (" {} * {} - ({}) = ?".format(x, y, z))
    print(question_txt)

    return solution, question_txt


# create a question in the form:  x+y*z-w = ?. prints the question and returns the solution:
def fun_q3(): # difficulty level 3
    x = random.randint(2, 30)
    y = random.randint(1, 20)
    z = random.randint(1, 9)
    tmp = x + y * z
    if tmp < 10:
        w = random.randint(1, tmp-1)
    else:
        w = random.randint(tmp-9, tmp-1)

    solution = x + y * z - w
    question_txt = (" {} + {} * {} - {} = ?".format(x, y, z, w))
    print(question_txt)

    return solution, question_txt


# create a question in the form:  (x-y)*z-w = ? . prints the question and returns the solution.
def fun_q4(): # difficulty level 4
    y = random.randint(1, 28)
    x = random.randint(y+2, 30)
    z = random.randint(2, 9)
    tmp = (x - y) * z
    if tmp < 10:
        w = random.randint(1, tmp-1)
    else:
        w = random.randint(tmp - 9, tmp-1)

    solution = (x - y) * z - w

    question_txt = (" ({} - {}) *{} - {} = ?".format(x, y, z, w))
    print(question_txt)

    return solution, question_txt


# create a question in the form: x*y-z*y-w = ?. prints the question and returns the solution:
def fun_q5():  # difficulty level 5
    x = random.randint(3, 20)
    y = random.randint(2, 10)
    z = random.randint(1, x-2)
    tmp = (x - z) * y
    if tmp < 10:
        w = random.randint(1, tmp-1)
    else:
        w = random.randint(tmp-9, tmp-1)

    solution = x * y - z * y - w

    # question_text = (x, "*", y, " - ", z, "*", y, " +", w, " = ? ")
    # print(x, "*", y, " - ", z, "*", y, " +", w, " = ? ")
    question_txt = (" {} * {} - {} * {} - {} = ?".format(x, y, z, y, w))
    print(question_txt)

    return solution, question_txt


#%%
def save_csv_details_1(data_list, subject_number, RESULTS_FOLDER):
    df_trials = DataFrame(data_list)
    df_trials.columns = ['count_quest', 'question_txt', 'correct_sol', 'level', 
                         'subject_answer', 'feedback', 'reaction_time',
                         'count_correct_ans', 'trial_score_now', 'trial']

    filename = RESULTS_FOLDER+str(subject_number)+"_details_1.csv"
    df_trials.to_csv(filename, index=False)
    
    
def save_csv_details(data_list, subject_number, RESULTS_FOLDER):
    df_trials = DataFrame(data_list)
    df_trials.columns = ['count_quest', 'question_txt', 'correct_sol', 'level', 
                         'subject_answer', 'feedback', 'reaction_time',
                         'count_correct_ans', 'trial_score_now', 'trial']

    filename = RESULTS_FOLDER+str(subject_number)+"_details.csv"
    df_trials.to_csv(filename, index=False)


# save to excel file the summary data for one user for one experiment session:
def save_csv_sum(summary_row, subject_number, RESULTS_FOLDER):
    # name and path of the csv file:
    # filename = RESULTS_FOLDER+str(subject_number)+"_"+trial+"_sum.csv"
    filename = RESULTS_FOLDER+str(subject_number)+"_sum.csv"
    # fields names (headers)
    fields = ['sub_ID', 'total_score', 'total_number_of_questions', 
              'total_correct_answers', 'timestamp_start', 'timestamp_end',
               'score_trial_A', 'score_trial_B', 'score_trial_C']
    
    # writing to csv file:
    with open(filename, 'w') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile)       
        csvwriter.writerow(fields) # writing the fields       
        csvwriter.writerows(summary_row) # writing the data rows
    csvfile.close()
    
    # df_sum = DataFrame([summary_row])
    # df_sum.columns = [fields]
    # df_sum.to_csv(filename, index=False)


def save_csv_sum2(dic_sum, subject_number, RESULTS_FOLDER):
    # name and path of the csv file:
    # filename = RESULTS_FOLDER+str(subject_number)+"_"+trial+"_sum.csv"
    filename = RESULTS_FOLDER+str(subject_number)+"_sum.csv"
    # fields names (headers)
   
    df_sum = DataFrame(dic_sum,  index=[0]).dropna(axis=0, how='all')
    df_sum.to_csv(filename, index=False)



def timer(start, limit):  # ALON's function
    '''
    Timer generator.
    get the start time and a limit.
    yield the time from start in seconds every second
    '''
    cur = start
    while time.time() - start < limit:
        if time.time() - cur >= 1:
            cur = time.time()
            yield time.time() - start
