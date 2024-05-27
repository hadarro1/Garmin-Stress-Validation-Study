# -*- coding: utf-8 -*-
"""
Created on Wed May 19 13:08:22 2021
@author: hadar

#### 
- A shorter version for the second experiment! only 15 minute total task.
- The score calculation was changed too: 
    reaction time 0-3 sec >> 10 points 
    reaction time 4-13 sec >> 9-1 points 
    reaction time longer then 13 sec >> 0 points 
####

I use tkinter package for the GUI.
using several function from my module 'fun_quiz'.
there are 3 trials of 7.5, 8.5, 9 minutes.
after welcome screen and instructions there are arithmetic questions (created randomly from 5 difficulty levels),
that the user has to answer as fast as he can 
(by typing numbers and pressing the NEXT button with the mouse).
the score depends on the reaction time (up to 5 sec its 10 points, and then every second it decreases by 1 point, it remains 1 point between 15 to 20 seconds and after 20 its 0.)

still missing: 
    - automatically move to the next question after x seconds, without an event - no reaction of user (x = a variable depends on the individual performance in a prior trial - 90% of the average rt of each difficulty level)
    - progress bar relate to the specific timelimit of the question.


"""
##############     version for PC and MAC      ###################

# ref: https://codereview.stackexchange.com/questions/106464/generating-arithmetic-quiz-questions
# colors: http://www.science.smith.edu/dftwiki/index.php/File:TkInterColorCharts.png
import time
import math
from tkinter import *
from tkinter import ttk
from functools import partial
# from playsound import playsound
from fun_quiz import *
import threading
import glob
import pygame

# modules to install: pygame, functools
# there are function in the file "fun_quiz.py")

# =====================    parameters to set:   ================================

# for total task duration of 15 minutes (including 2 30 seconds break)
max_duration_A = 240 #(4 minutes) 
max_duration_B = 240 #(4 minutes) 
max_duration_C = 240 #(4 minutes) 
initial_break_time = 30 
counter_break = initial_break_time 
rt_long = 10
max_num_quest = 9999  # total number of questions

TIME_DELAY_PER_QUESTION = 8_000 # [milisec]
#________________
LIMIT = 8
#________________

RESULTS_FOLDER = 'quiz_results_folder/'
data_list = []
trials_scores_list = [0,0,0]


# =========================  Create a GUI Window ===============================
root = Tk()
root.geometry("1500x700")  # window size
root.title("MATH QUIZ")
pygame.mixer.init()  # to initiate the pygame (for sounds)

# =================== define all variable for the quiz: ==========================
# time measurement:
start_quiz_time = datetime.datetime.now()
START = 0
t_start_trial = 0  # start countdown to max_duration of the quiz. (after subject press 'START')
total_score  = 0
best_score = get_max_val()   # find the higher score for the progress bar: (the function is in "fun_quiz_gui.py")

# ==============  create the StringVar and Int_Var and set initial values ==================
trial_var = StringVar()
trial_var.set("Trial A")
max_time_var = IntVar()
max_time_var.set(max_duration_A)

subject_number_var = IntVar()
subject_number_var.set(0)

count_var = IntVar()
count_var.set(0)
count_correct_var = IntVar()
count_correct_var.set(0)
score_var = IntVar()  # hold the accumulated score from the beginning of the current trial. 
score_var.set(0)  
recent_trial_score_var = IntVar()  # hold the score of the trial that ended befor. 
recent_trial_score_var.set(0)

question_txt_var = StringVar()
question_txt_var.set("")
solution_var = IntVar()
solution_var.set(0)
level_var = IntVar()
level_var.set(0)

reaction_message_var = StringVar()
reaction_message_var.set("")
feedback_txt_var = StringVar()
feedback_txt_var.set("")
sum_txt_1_var = StringVar()
sum_txt_2_var = StringVar()


#%%
# ========================= define the functions for GUI  ==========================


def welcome_submit():
    """
    activated by the 'button_submit' in the first screen.
    present the second screen: instructions + start button

    Returns
    -------
    SUB = the subject ID 

    """
    SUB = int(e.get())
    subject_number_var.set(SUB)
    e.destroy()
    button_submit.destroy()
    lab2.destroy()
    instructions.pack()
    button_start.pack()
    return


def start_q():
    """
    - activated by the button_start. (after user read the instructions screen)
    - clear the screen. 
    - call the function 'ask_question()' to create a random arithmetic question, 
        solution, difficulty level. and saves them as variables.
    - present the first question screen, which includes: 
        trial number, instruction text, arithmetic question, 
        entry box for user answer, 'button_next'.
    
    - set the START to calculate reaction time in another function later.
    - set the t_start_trial to control the trial duration.

    Returns
    -------
    START : float (represent time e.g., 1659002725.7501898)
        is the time the question was presented to the user. 
        we will use it for reaction time calculation
    """
    
    frame_welcome.destroy()
    frame_core.pack(pady=20)
    button_next.pack(pady=10)

    sol, quest, level = ask_question()
        
    question_txt_var.set(quest); solution_var.set(sol); level_var.set(level)

    global START, t_start_trial
    START= time.time()   # for reaction time
    t_start_trial = time.time()

    ### play a bell for a new question and stressful ticking 5 seconds after the bell
    threading.Thread(target=play_sound_bell_and_tick).start() 
    
    
    ##
    # Create timeout
    # progress bar
    ##
    
    # root.after(TIME_DELAY_PER_QUESTION, timeout)
    
    ### move to the next question automatically if no response and rt > LIMIT
    # start1 = time.time()
    # rt = time.time()-start
    # while True:
    #   print(rt)
    #        if time.time() - start1 > LIMIT:
    #            submit_input(rt_long)
 
    return START


def next_q():
    """
    same like start_q() for all the questions after the first.
    - activated by the button_next. 
    - clear the screen. 
    - call the function 'ask_question()' to create a random arithmetic question, 
        solution, difficulty level. and saves them as variables.
    - present the first question screen, which includes: 
        trial number, instruction text, arithmetic question, 
        entry box for user answer, 'button_next'.
    
    - set the START to calculate reaction time in another function later.
    - set the t_start_trial to control the trial duration.
    
    - calculate trial duration to end the trial when its the trial timelimit.

    Returns
    -------
    START : float (represent time e.g., 1659002725.7501898)
        is the time the question was presented to the user. 
        we will use it for reaction time calculation
    """
    # delete labs from former trial:
    frame_messages.pack_forget()
    entry_ans.delete(0, END)
    question_txt_var.set(" ")

    # print("\nmaximal time for this trial is:"+str(max_time_var.get()))
    
    t = time.time()
    # to check duration of total session and cal the end function if its longer than max_duration:
    # (depends on the trial duration)
    dt = t-t_start_trial
    # print("seconds from start:" + str("{:.2f}".format(dt)))
    # root.after(TIME_DELAY_PER_QUESTION, timeout)
    
    if trial_var.get() == "Trial C" and dt >= max_time_var.get():
        end_window()
    elif dt >= max_time_var.get():
        end_trial()
    else:
        lab_directions.pack(pady=10)
        button_next.pack(pady=20)
        # lab_quest.pack(pady=10)

        sol, quest, level = ask_question()
        question_txt_var.set(quest)
        solution_var.set(sol)
        level_var.set(level)

        feed1 = "your score is " + str(score_var.get())
        # feed2 = percent(score_var.get())
        # feedback = feed1 + '\nprogress to first place:\t' + feed2
        feed2 = percent2(score_var.get())
        feedback = feed1 + '\nyou are far from the first place by ' + str(feed2) + ' points.'
        
        feedback_txt_var.set(feedback)
        frame_messages.pack(pady=10)

        # frame_core.pack(pady=20)
        # entry_ans.pack(pady=10)

        global START  
        START = time.time()

    threading.Thread(target=play_sound_pyg_tick).start()
    
    # marker = False
    # while True:
    #     #time.sleep(1)
    #     rt = time.time()-start1
    #     if rt > 5:
    #         marker = True
    #         break
    # print(rt, marker)

    return START

def timeout(): 
    """
    move to next question if reaction time is longer than LIMIT by calling next_q() function

    """
      
    next_q()


#%%
def submit_input(rt_long):
    """
    - calculate reaction time, stop sound, compare user input with solution 
    - calculate score depends on reaction time.
    - and saves variables of the specific question and answer for the details table.   
        trials_data_list.append([...])
        { count_trial,   quest, sol,   user_input,   feedback, 
        reaction_time,   correct_count_trial,   score_trial  }
        
    - calls the next_q() after 1 sec.

    Parameters
    ----------
    rt_long : int  
        number of seconds that if reaction time is greater than- there is a massage warn the user that his rt is long.

    Returns
    -------
    None.

    """
    end = time.time()
    reaction_t = end - START
    stop_pygame()
    
    print(trial_var.get() + " , quest number " + str(count_var.get()))

    user_input = entry_ans.get()
    entry_ans.delete(0, END)
    
    # print("the reaction time is " + str("{:.2f}".format(reaction_t)) + " seconds")

    frame_messages.pack(pady=20)

    # take all variables to check the sub answer:
    sol, quest, level = solution_var.get(), question_txt_var.get(), level_var.get()
    tmp_count, tmp_correct_count = count_var.get(), count_correct_var.get()
    tmp_score = score_var.get()

    if user_input == "" or user_input == " ":
        feedback = False
        threading.Thread(target=play_sound_pyg_incorrect).start()
        subtract = 10
        score_var.set(tmp_score - subtract)
        reaction_message_var.set("you didn't respond. " + "-" + str(subtract))
        lab_reaction_message.pack()

    else:
        user_input = int(user_input)
        # call function to check answer:
        feedback = check_answer(user_input, sol)

        if feedback:
            threading.Thread(target=play_sound_pyg_correct).start()
            count_correct_var.set(tmp_correct_count+1)
            if reaction_t <= 3:
                add = 10
                score_var.set(tmp_score + add)
            elif 3 < reaction_t <= 13:
                math.ceil(reaction_t)
                add = 13-int(math.ceil(reaction_t))
                score_var.set(tmp_score + add)
            # elif 15 < reaction_t <= 20:  ### cancel the score for reaction time longer than 15 sec
            #     add = 1
            #     score_var.set(tmp_score + add)
            else:
                add = 0
                score_var.set(tmp_score + add)

            feedback_txt_var.set("correct!   " + "+" + str(add))
            lab_feedback.config(bg="pale green")
            lab_feedback.pack()

        else:  # feedback=False
            threading.Thread(target=play_sound_pyg_incorrect).start()
            time.sleep(0.5)
            subtract = 10
            score_var.set(tmp_score - subtract)
            feedback_txt_var.set("incorrect!   " + "-" + str(subtract))
            lab_feedback.config(bg="tomato")
            lab_feedback.pack()
            
        print(feedback_txt_var.get())
        print('score is: '+ str(score_var.get()))


        if reaction_t > rt_long:
            reaction_message_var.set("your reaction time is long!\n"
                                     + str(round(reaction_t)) + " seconds")
        else:
            reaction_message_var.set("")
        lab_reaction_message.pack()

    # update the trial variables and save them in the data list:
    count_var.set(tmp_count+1)
    count_quest, count_correct_ans,  = count_var.get(), count_correct_var.get() 
    trial_score_now = score_var.get()
    trial = trial_var.get()
    data_list.append([count_quest, quest, sol, level, user_input, feedback, reaction_t, 
                      count_correct_ans, trial_score_now, trial])

    button_next.pack_forget()

    global counter_break
    counter_break = initial_break_time
    root.after(1000, next_q)

    return

#%%


def end_trial():
    global counter_break
    counter_break = initial_break_time

    threading.Thread(target=play_sound_coin).start()
    print("end trial - break (function end_trial())")
    # export_data()
    frame_core.pack_forget()
    # frame_messages.pack_forget()
    
    # update the score for the trial that just ended:
    tmp = score_var.get()
    recent_trial_score_var.set(tmp) 
    print("the score of the trial is: " + str(recent_trial_score_var.get()))
    
    # save the score of the trial in the list:
    if trial_var.get() == "Trial B":        
        trials_scores_list[1] = recent_trial_score_var.get() 
    
    if trial_var.get() == "Trial A":
        trials_scores_list[0] = score_var.get()
    
    frame_break.pack(pady=10)

    # reset txt variables:
    question_txt_var.set("")
    solution_var.set(0)
    reaction_message_var.set("")
    feedback_txt_var.set("")

    # lab_countdown.pack()
    threading.Thread(target=count_break_label(lab_countdown, frame_break)).start()
    # count_break_label(lab_countdown)

    root.after(initial_break_time*1000, start_new_trial)
    return



#%%
def count_break_label(lab_countdown, frame_break):
    """
    a timer for the break between trials

    Parameters
    ----------
    lab_countdown : tkinter label
        numbers of seconds.
    frame_break : tkinter frame
        DESCRIPTION.

    Returns
    -------
    counter_break : int
        length of break [seconds].

    """
    def count():
        global counter_break
        if counter_break == 0:
            return counter_break
        counter_break -= 1
        lab_countdown.config(text=str(counter_break))
        frame_break.after(1000, count)

    if counter_break == 0:
        return
    count()


#%%
def start_new_trial():
    print("\n\n new trial (function start_new_trial(), counter_break = " + str(counter_break))
    
    if trial_var.get() == "Trial B":
        # update the trial name:
        trial_var.set("Trial C")  
        # update the duration for next break:
        max_time_var.set(max_duration_A + max_duration_B + initial_break_time
                         + max_duration_C + initial_break_time)   
    if trial_var.get() == "Trial A":
        # update the trial name:
        trial_var.set("Trial B")
        # update the duration for next break:
        max_time_var.set(max_duration_A+max_duration_B+initial_break_time)
        
    # update the best score due the last trial:
    global best_score
    if recent_trial_score_var.get() > best_score:        
        best_score = recent_trial_score_var.get()
        print("best score updated = ", best_score)
    
    score_var.set(0)
    
    # trial_score_var.set(0)
    frame_break.pack_forget()   
    # frame_timer.pack(pady=10)
    frame_core.pack(pady=20)
    button_next.pack(pady=10)

    sol, quest, level = ask_question()
    question_txt_var.set(quest)
    solution_var.set(sol)

    global start
    START= time.time()  # for reaction time

    threading.Thread(target=play_sound_bell_and_tick).start()
    # threading.Thread(target=play_sound_pyg_tick).start()

    return start


#%%
def end_window():
    global counter_break
    counter_break = initial_break_time
    
    threading.Thread(target=play_sound_coin).start()
    print("end of last trial - break (function end_window)")
    
    frame_core.destroy()
    
    total_count = count_var.get()
    total_correct_answers = count_correct_var.get()
         
    final_trial_score = score_var.get()  # update the score for the trial that just ended:
    recent_trial_score_var.set(final_trial_score)   
    print("the score of the last trial is: " + str(final_trial_score))
       
    trials_scores_list[2] = final_trial_score  # save the trial score in the list
    global total_score
    total_score = sum(trials_scores_list)  # the total score of the 3 trials together
    
    txt_1 = ('\n\nTime is up!'  
             '\nYou have completed the last trial of the quiz.'
             '\n\nPlease stay in your sitting position for a few minutes.'
             '\n\nyour score in the last trial is: ' + str(final_trial_score)+
             '\nyour best trial score is: '+str(max(trials_scores_list))+
             '\nand your total score is: ' + str(total_score))
             
    txt_2 = ("\n\nPress 'EXIT QUIZ' to save the data.")
    sum_txt_1_var.set(txt_1)
    sum_txt_2_var.set(txt_2)
    frame_exit.pack(pady=20)
    
    return total_score



# =========================    end of loop    =================================
# =============================================================================
# =============================================================================
# =============================================================================



#%%

# =============   create Tkinter widgets: frames, labels and text   ===========

frame_welcome = LabelFrame(root, text="", padx=5, pady=5)
lab1 = Label(frame_welcome, text="\n Welcome to the fast math Quiz.\n", font=('times', 32, 'bold'))
lab2 = Label(frame_welcome, text="Please enter your participant number here and press 'submit'", font=('Gisha', 20))
# create an entry box (for user input):
e = Entry(frame_welcome, width=20, borderwidth=5, font=20)
button_submit = Button(frame_welcome, text="Submit", font=30, fg="blue",
                       command=welcome_submit)
button_submit.bind("<Return>", welcome_submit)
instructions = Text(frame_welcome, height=16, width=90)
instructions.config(font=('Gisha', 18))
instructions_text = " Thank you for participating!  Now you will be completing an arithmetic quiz." \
                    "\n\n The score for a correct answer will be calculated based on your response time."\
                    "\n\n A 3 seconds response time or less will earn you 10 points." \
                    "\n After 3 seconds, the score will be decreased from 10 to 1, by a point per second." \
                    "\n A delayed response of more than 13 seconds will not earn any points." \
                    "\n\n An incorrect answer will result in a 10 point reduction." \
                    "\n\n Try to type a correct answer as fast as possible in order to collect as many points as you can. " \
                    "\n Your monetary bonus is calculated based of your score." \
                    "\n\n Please press 'START' to begin the quiz."

instructions.insert(END, instructions_text)
# 'Comic Sans MS'
button_start = Button(frame_welcome, text="START", font=60, fg="black", bg="SpringGreen2", padx=40, pady=20,
                      command=start_q)
button_start.bind("<Return>", start_q)
# ________________________________________________________________________________________________________________

frame_core = LabelFrame(root, padx=5, pady=5)
lab_trial = Label(frame_core, textvariable=trial_var, font=('Gisha', 28), fg="blue")
lab_trial.pack(pady=10)
lab_directions = Label(frame_core, text="Type your answer here and press 'NEXT'\n\n ", font=('Gisha', 24))
lab_directions.pack(pady=10)
lab_quest = Label(frame_core, textvariable=question_txt_var, font=('times', 30, 'bold'))
lab_quest.pack()
entry_ans = Entry(frame_core, width=20, borderwidth=8, font=100)
entry_ans.pack(pady=10)
button_next = Button(frame_core, text="NEXT", font=60, bg="grey", padx=50, pady=30, command=partial(submit_input, rt_long))
# button_next.bind("<Return>", partial(submit_input, rt_long))
button_next.pack(pady=10)
# _________________________________________________________________________________________________________________

frame_messages = LabelFrame(root, padx=5, pady=5)
lab_reaction_message = Label(frame_messages, textvariable=reaction_message_var, fg="red", font=('Gisha', 30))
lab_feedback = Label(frame_messages, textvariable=feedback_txt_var, font=('Gisha', 24))
# _________________________________________________________________________________________________________________

frame_break = LabelFrame(root, padx=5, pady=10)
lab_countdown = Label(frame_break, text="", bg="light blue", font=('times', 40, 'bold'))
lab_countdown.pack()
lab_break = Label(frame_break, text='\n\nTime is up!  You have completed the trial'
                                    '\nPlease stay to sit for a one minute break,\nthe next trial will soon begin.'
                                    '\n\nyour trial score is:  ', font=('Gisha', 25))
lab_break.pack()
lab_score_trial = Label(frame_break, textvariable=recent_trial_score_var, font=('Gisha', 25))
lab_score_trial.pack()
# _________________________________________________________________________________________________________________

frame_exit = LabelFrame(root, padx=5, pady=5)
lab_sum_final_trial = Label(frame_exit, textvariable=sum_txt_1_var, font=('Gisha', 25))
lab_sum_final_trial.pack()
lab_sum_all = Label(frame_exit, textvariable=sum_txt_2_var, font=('Gisha', 25))
lab_sum_all.pack()
button_end_exit = Button(frame_exit, text='EXIT QUIZ', font=70, padx=70, pady=30, fg="red", command=root.destroy)
button_end_exit.pack()




#%%
# ==========================    more functions:   =============================

# calculate how far the user from the first place

# returns a string that describes relatively
def percent(current_trial_score):
    global best_score
    max_val = best_score if best_score >= 100 else 100
    n = int(10 * current_trial_score/max_val)
    if current_trial_score == best_score:
        return '#'*10
    elif n < 10:
        return '#'*n + '_'*(10-n)
    elif n == 0:
        return '_'*10
    else:
        return 'First Place !'
    

# returns the number of points missing for winning the first place
def percent2(current_trial_score):
    global best_score
    max_val = best_score if best_score >= 100 else 100
    n = int(10 * current_trial_score/max_val)
    delta = best_score - current_trial_score
    
    return delta

    
# sound functions:
def play_sound_pyg_correct():
    pygame.mixer.music.load("audio/correct_sound.ogg")
    pygame.mixer.music.play()
def play_sound_pyg_incorrect():
    pygame.mixer.music.load("audio/incorrect_sound.ogg")
    pygame.mixer.music.play()
def play_sound_pyg_tick():
    # time.sleep(5)
    pygame.mixer.music.load("audio/Ticking_noise-edit-Short3secSilence.ogg") 
    # pygame.mixer.music.load("audio/Ticking_noise-edit.ogg")
    pygame.mixer.music.play()
def play_sound_coin():
    pygame.mixer.music.load("audio/coin.ogg")
    pygame.mixer.music.play()

def play_sound_bell_and_tick():
    pygame.mixer.music.load("audio/bell_and_tick.ogg")
    pygame.mixer.music.play()
def stop_pygame():
    pygame.mixer.music.stop()



def export_data2(total_score):
    sub_id = subject_number_var.get()
    # trial = trial_var.get()
    end_quiz_time = datetime.datetime.now()
    total_count = count_var.get()
    total_correct_answers = count_correct_var.get()
    
    fields = ['sub_ID', 'total_score', 'total_number_of_questions', 
              'total_correct_answers', 'timestamp_start', 'timestamp_end',
               'score_trial_A', 'score_trial_B', 'score_trial_C']
    dic_sum = {'sub_ID':sub_id, 
                   'total_score':total_score, 'total_number_of_questions':total_count, 
                   'total_correct_answers':total_correct_answers, 
                   'timestamp_start':start_quiz_time.strftime("%Y-%m-%d %H:%M:%S"),     
                   'timestamp_end':end_quiz_time.strftime("%Y-%m-%d %H:%M:%S"),
                   'score_trial_A':trials_scores_list[0] , 
                   'score_trial_B':trials_scores_list[1] , 
                   'score_trial_C':trials_scores_list[2] }
    
    
    # save the summary data row into a separate csv file:
    # save_csv_sum(summary_row, sub_id, RESULTS_FOLDER)
    save_csv_sum2(dic_sum, sub_id, RESULTS_FOLDER)

    # write the data from each trial in a dataframe and save as csv file:
    save_csv_details(data_list, sub_id, RESULTS_FOLDER)
  
    
  
#%%


# =============================================================================
# =============================================================================
# =============================================================================

# ======================    activate the gui window:   ========================

frame_welcome.pack(pady=10)
lab1.pack()
lab2.pack(pady=10)
e.pack()
button_submit.pack()

root.mainloop()

# ===================  after exit gui - save all data in csv files:  ==============
stop_pygame()  # to stop the tikking sound

export_data2(total_score)

# =============================================================================
# =============================================================================
# =============================================================================









# # to change the text of exist button.

# button_submit_ans['text'] = 'START'
# button_stop_timer = Button(root, text='Stop', width=25, command=XXXX)
# button_stop_timer.pack()
# button_exit.config(width=10, height=5)

# build a timer with threading: #https://www.geeksforgeeks.org/create-countdown-timer-using-python-tkinter/
# https://stackoverflow.com/questions/34029223/basic-tkinter-countdown-timer

# for sec in range(5, 0, -1):
#     lab_countdown.config(text=str(sec))
#     lab_countdown.after(1000)


# def fun_timer()
    # seconds = break_time
    # while seconds:
    # seconds -= 1
    # # seconds_var.set(seconds)
    # root.after(1000, countdown)


# def countdown():
#
#     # change text in label
#     time.sleep(5)
#     lab_countdown['text'] = '00:05'
#     time.sleep(5)
#     lab_countdown['text'] = '00:10'
#     time.sleep(5)
#     lab_countdown['text'] = '00:15'
#     time.sleep(5)
#     lab_countdown['text'] = '00:20'


# def countdown(seconds):
#     seconds = seconds_var.get()
#     while seconds:
#         seconds -= 1
#         seconds_var.set(seconds)
#         root.after(1000, countdown)