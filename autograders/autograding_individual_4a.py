import autograding_individual_4_1a as p1
import autograding_individual_4_2a as p2
import autograding_individual_4_3a as p3
import autograding_individual_4_4a as p4
import pandas as pd
import numpy as np
import filecmp
import os

# To get debug messages
DEBUG = True

# Colors for console printing,
W = '\033[0m'   # white (normal)
R = '\033[31m'  # red
O = '\033[33m'  # orange
Y = '\033[93m'  # yellow
G = '\033[32m'  # green


"""
    ~~~~~~~~~~~~~~~~~ Project 4   ver 4a ~~~~~~~~~~~~~~~~~
    This file will run ALL of Project 4!
    Place this within the file tree given below.
        1) You may run all the problems at once with with this file
            1a) This file will call all of the problem-autograders.
            1b) You can run those autograders on their own if you wish.
        2) You can specify whether you are grad student or not below.
        
      !!!) DO NOT RUN on your LOGIN NODE!! This is a no no.
      !!!) You can run on the gpel machines-terminals.
        
    RUN LIKE: python or python3 autograding_individual_4a.py
    
    Contact the TA if you have any issues.     
    
    >> Make sure you have the structure below!:
    Project_3/         [..] 
        test_data/
            DNA_Files/
            Problem_2/
            Problem_3/
            Problem_4/
        your_solutions/    [.]   
            autograding_individual_4a.py    <--- Place here.
            autograding_individual_3_2a.py  <--- This too!
            autograding_individual_3_4a.py  <--- This too!
            autograding_individual_3_4a.py  <--- This too! (If grad)
            autograding_sbatch.sbatch
            Problem_2/
            Problem_3/
            Problem_4/
            
    NOTE: please ignore the TA) TODO: tags, they will be used for changing the autograder between projects.
"""


# !! --> CHANGE THIS if you are NOT a grad student! (They get more problems.) <-- !!
ind_is_grad = True

# change this to your name if you want (does not affect grading)
ind_student_name = "Student"

# file locations
#   Note: this python file is located in './'.
ind_test_dir = "../test_data"   # this contains the test data
ind_this_dir = "."              # this is ShawJes_HW/

# ------------------------------------------
# this will autograde one project submission
def autograde(in_this_dir, in_test_dir, in_student_name, in_is_grad):
    # for mass grading purposes, ignore if individually grading
    #   getting the abs path resolves some issues...
    this_dir     = os.path.abspath(in_this_dir)
    test_dir     = os.path.abspath(in_test_dir)
    student_name = in_student_name
    is_grad      = in_is_grad

    # store result dataframes here
    all_res = []

    # Problem 1
    try:
        print(O + "\nGrading problem 1..." + W)
        res_p1 = p1.autograde_p1(this_dir, test_dir, student_name, DEBUG)
        all_res.append([res_p1[0], res_p1[1]])
    except Exception as err:
        print('\n' + R + "Unexpected error while grading problem 1!")
        print(Y + str(err) + '\n' + W)
        all_res.append(None)

    # Problem 2
    try:
        print(O + "\n\nGrading problem 2..." + W)
        res_p2 = p2.autograde_p2(this_dir, test_dir, student_name, DEBUG)
        all_res.append([res_p2[0], res_p2[1]])
    except Exception as err:
        print('\n' + R + "Unexpected error while grading problem 2!")
        print(Y + str(err) + '\n' + W)
        all_res.append(None)

    # Problem 3
    try:
        print(O + "\n\nGrading problem 3..." + W)
        res_p3 = p3.autograde_p3(this_dir, test_dir, student_name, DEBUG)
        all_res.append(res_p3)
    except Exception as err:
        print('\n' + R + "Unexpected error while grading problem 3!")
        print(Y + str(err) + '\n' + W)
        all_res.append(None)

    # Problem 4
    if is_grad:
        try:
            print(O + "\n\nGrading problem 4..." + W)
            res_p4 = p4.autograde_p4(this_dir, test_dir, student_name, DEBUG)
            all_res.append([res_p4[0], res_p4[1]])
        except Exception as err:
            print('\n' + R + "Unexpected error while grading problem 4!")
            print(Y + str(err) + '\n' + W)
            all_res.append(None)

    return [all_res]


# --------------------------------------------------------
# This will run autograde project 3 IF you run this file only!
# This is so it is possible to grade multiple submissions with another python file.
if __name__ == "__main__":
    print(G + "Autograding for Project 3:" + W + '\n')
    all_res = autograde(ind_this_dir, ind_test_dir, ind_student_name, ind_is_grad)

    # student grades
    grades = pd.DataFrame(
        np.nan,
        index=[ind_student_name],
        columns=[]
    )

    # student timings
    times = pd.DataFrame(
        np.nan,
        index=[ind_student_name],
        columns=[]
    )

    # merge results and print to file
    count = 1
    for res in all_res:
        for i in range(len(res)):
            if res[i] is not None:
                grades = pd.concat([grades, res[i][0]], axis=1)
                times  = pd.concat([times,  res[i][1]], axis=1)
                res[i][0].to_csv("P" + str(count) + "_grades.csv")
                res[i][1].to_csv("P" + str(count) + "_times.csv")
            count = count + 1

    # find all correct
    total = str(len(grades.columns))
    correct = str(int(grades.sum(axis=1)[0]))

    # show grades
    print(Y + "\n Final grades:" + W)
    grades.to_csv("Project_4_grades.csv")
    print(grades)

    # show times
    print(Y + "\n Final timings:" + W)
    times.to_csv("Project_4_times.csv")
    print(times)

    # show correct
    print(R + "\n --> " + correct + "/" + total + " problems correct\n" + W)
