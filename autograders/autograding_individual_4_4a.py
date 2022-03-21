import pandas as pd
import numpy as np
import filecmp
import os

# To get debug messages
DEBUG = True

# Colors for console printing,
W = '\033[0m'  # white (normal)
R = '\033[31m'  # red
O = '\033[33m'  # orange
Y = '\033[93m'  # yellow
G = '\033[32m'  # green

"""
    ~~~~~~~~~~~~~~~~~ Project 4_4   ver 4_4a ~~~~~~~~~~~~~~~~~
    This file will run problem 4 of Project 4 ONLY!
    Place this within the file tree given below.
        1) You may run this autograder on its own.
        2) You may run all the problems at once with autograding_individual_4a.py
        3) You may specify what thread counts you want to run below.

    RUN LIKE: python or python3 autograding_individual_4_1a.py

    Contact the TA if you have any issues.    

    >> Make sure you have the structure below!:
    Project_4/         [..] 
        test_data/
            Problem 1 and 2/
            Problem 3/
            Problem 4/
        your_solutions/    [.]   
            autograding_individual_4_4a.py   <--- Place here.
            Problem_4/
            ...

    NOTE: please ignore the TA) TODO: tags, they will be used for changing the autograder between projects.
"""

# change this to your name if you want (does not affect grading)
ind_student_name = "Student"

# file locations
#   Note: this python file is located in './'.
ind_test_dir = "../test_data"  # this contains the test data
ind_this_dir = "."  # this is your_solutions/

# indicates if the folder has been made (don't change)
is_missing = False
is_made = False

# TA) TODO: add column names for each test performed
test_names = [
    "p4"
]


# ------------------------------------------
# this will autograde one project submission
def autograde_p4(in_this_dir, in_test_dir, in_student_name, debug):
    # for mass grading purposes, ignore if individually grading
    #   getting the abs path resolves some issues...
    this_dir = os.path.abspath(in_this_dir)
    test_dir = os.path.abspath(in_test_dir)
    student_name = in_student_name

    # Print the test dir and project dir
    if debug:
        print(G + "--> Test dir: " + test_dir + W)
        print(G + "--> Project dir: " + this_dir + W)

    # get num cols for threads
    columns = []
    for test in test_names:
        columns.append(test)

    # student grades
    grade = pd.DataFrame(
        np.nan,
        index=[student_name],
        columns=columns
    )

    # student timing
    time = pd.DataFrame(
        np.nan,
        index=[student_name],
        columns=columns
    )

    # TA) TODO: add the correct test files for every problem
    #  each problem will have its own set of test files
    #   list the test file names for each problem here
    #   NOTE: the test files have special directories, so we have to specify them here

    # t_dir = problem directory
    # t_in  = input files
    # t_out = expected output
    # t_get = program result
    # t_tim = program time

    # Problem 3 ####################
    # expected input
    t_dir = test_dir + "/Problem 4/"
    t_p4_in = [
        t_dir + "mat_input.csv"
    ]
    # expected output
    t_p4_out = [
        t_dir + "mat_p4_output.csv"
    ]

    # actual program output
    t_dir = this_dir + "/Problem_4/"
    t_p4_get = [
        t_dir + "results.csv",
    ]
    t_p4_tim = [
        t_dir + "time.csv",
    ]

    # TA) TODO: for each problem, generate commands to make and run the test cases
    #   generate the commands to run the tests here
    c_p4 = []

    # TA) TODO: alternate program names, if needed
    p4_names = [
        "convolution_maxpooling_CUDA"
    ]
    p4_dims = [
        2048,
        2048
    ]

    # TA) TODO: generate the problems' command-variables
    # Problem 3
    for file in range(len(test_names)):  # For input
        c_p4.append([
            p4_names[0],     # program name
            p4_dims[0],      # input dim
            p4_dims[1],      # input dim
            t_p4_in[0],      # input file
            t_p4_get[file],  # resulting output file
            t_p4_tim[file]   # resulting time file
        ])

    #  we have everything we need to test a problem now
    #   grade each individual problem here!
    # TA) TODO: specify each problem's test parameters
    # Problem 3
    test_params = []
    for file in range(len(test_names)):  # For input
        test_params.append([
            this_dir + "/Problem_4/",
            t_p4_out[file],
            t_p4_get[file],
            c_p4[file],
            False
        ])

    # testing results
    test_results = [None] * len(columns)
    time_results = [None] * len(columns)

    # test every problem in a loop
    grade_index = 0
    for file in range(len(test_names)):
        params = test_params[file]
        result = grade_problem_4(
            params[0],  # Problem dir
            params[1],  # Expected outputs of test i
            params[2],  # Output file names
            params[3],  # Command for getting test i results
            params[4],  # Whether to let the differences have an error range
            debug  # Whether to print debug statements
        )

        # set results
        test_results[file] = result[0]
        time_results[file] = result[1]

        # add each result to the dataframes
        grade.loc[student_name, columns[file]] = test_results[file][0]
        time.loc[student_name, columns[file]] = str(time_results[file][0])

    return [grade, time]


# -----------------------------------------------------------------
# this will take a problem, run the student code, then compare with
#   expected output. it will return the points earned
def grade_problem_4(student_dir, t_output, t_res, incommand, exact, debug):
    global is_missing
    global is_made

    # how many tests to run
    n = 1

    # array of scores to return
    scores = []
    for i in range(n):
        scores.append(0)

    # array of times to return
    times = []
    for i in range(n):
        times.append(-1.0)

    # make programs
    if not is_made:
        # alert what directory doesn't have a makefile (dont copy it)
        if not os.path.isfile(student_dir + "Makefile"):
            print('\n' + R + "ERROR: Missing Makefile when trying to test in " + student_dir + "!" + W)
            print(R + "       Skipping testing for " + incommand[0] + "..." + W)
            is_made = True
            is_missing = True
            return scores

        # run makefile
        command = "(cd " + student_dir + " && make)"
        os.system(command)
        is_made = True

    # simply return zeros if failed to make
    elif is_missing:
        return scores

    # try to make and run the code
    try:
        # generate the command and run the program
        #   ex. ./Problem_1/parallel_mult_mat_mat in.csv 10 10 ...
        command = "(cd " + student_dir + " && ./" + str(incommand[0]).replace(" ", "\\ ")
        for j in range(1, len(incommand) - 1):
            command = command + ' ' + str(incommand[j]).replace(" ", "\\ ")
        command = command + ' ' + str(incommand[-1]).replace(" ", "\\ ") + ")"

        if debug:
            print('\n' + G + "--> Running test for " + incommand[0] + "'s output: " + t_res + W)
        os.system(command)

        # get data from csv result and expected outputs
        result = np.genfromtxt(t_res, delimiter=",", dtype=float)
        expected = np.genfromtxt(t_output, delimiter=",", dtype=float)

        # compare file dims
        if expected.shape != result.shape:
            print(R + "Output file " + t_output + "'s dimensions are different from expected result's!" + W)
            matches = False

        # compare the files by simply looking at the text
        elif exact:
            matches = filecmp.cmp(
                t_output,
                t_res,
                shallow=False
            )

        # compare by considering value-errors
        else:
            diff = np.sum(np.absolute(expected - result))
            diff = diff / np.ravel(expected).shape[0]
            if diff < 0.1:
                matches = True
            else:
                matches = False

        # give final score
        if matches:
            scores[0] = 1
        else:
            if debug:
                print(R + "The expected output: " + t_output + " does not match the result!" + W)

        # get times
        t = np.genfromtxt(incommand[-1], delimiter=',')
        times[0] = t

        if debug:
            print(Y + "    Test result " + str(i) + " = " + str(scores[0]) + W)
            print(Y + "    Time result " + str(i) + " = " + str(times[0]) + "s" + W)

    # catch the weird stuff
    except Exception as err:
        print('\n' + R + "Unexpected error!")
        print(Y + str(err) + W)

    return [scores, times]


# -----------------------------------------
# generate file names from a given template
# FORMAT LIKE THIS: "test_{index}.txt"
def gen_filenames(template, n):
    filenames = []
    for i in range(1, n + 1):
        filename = template.format(index=i)
        filenames.append(filename)
    return filenames


# --------------------------------------------------------
# This will run this problem IF you run this file only!
# This is so it is possible to grade multiple submissions with another python file.
if __name__ == "__main__":
    print(G + "Autograding for Project 4 Problem 4:" + W + '\n')
    res = autograde_p4(ind_this_dir, ind_test_dir, ind_student_name, DEBUG)

    total = str(len(res[0].columns))
    correct = str(int(res[0].sum(axis=1)[0]))

    print(Y + "\n Final grades:" + W)
    res[0].to_csv("P4_4_grades.csv")
    print(res[0])

    print(Y + "\n Final timings:" + W)
    res[1].to_csv("P4_4_times.csv")
    print(res[1])

    print(R + "\n --> " + correct + "/" + total + " problems correct\n" + W)
