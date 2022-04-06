from termcolor import colored

from naive_algorithm import naive
from dlt_algorithm import dlt
from dlt_modified_algorithm import dlt_modified

import numpy as np


def print_bar():
    print(colored("---------------------------------------------------------", 'green'))


def print_blue(text):
    print(colored(text, 'blue'))


def print_red(text):
    print(colored(text, 'red'))


def print_blue_and_yellow(text1, text2):
    print(colored(text1, 'yellow'), colored(text2, 'blue'))


print_red("Testing \"trapeze and rectangle\" example:")

a = (-3, -1, 1)
b = (3, -1, 1)
c = (1, 1, 1)
d = (-1, 1, 1)
e = (1, 2, 3)
f = (-8, -2, 1)

a_target = (-2, -1, 1)
b_target = (2, -1, 1)
c_target = (2, 1, 1)
d_target = (-2, 1, 1)

print_bar()

# Testing naive algorithm with first 4 points
start = [a, b, c, d]
target = [a_target, b_target, c_target, d_target]

naive_matrix = naive(start, target)

print_blue("Raw naive matrix:")
print(naive_matrix)

naive_matrix = np.round(naive_matrix, 5)

print_blue("Naive matrix rounded to 5 decimals:")

# Matrix should be equal to:
# [[ 2. -0.  0.]
#  [-0.  2. -1.]
#  [ 0. -1.  2.]]
print(naive_matrix)
print_bar()

print_blue("Values of e and f:")
print("e = ", e)
print("f = ", f)

print_blue("Calculating e' and f' using naive matrix:")

# Values of e' and f' should be:
# e' = (2, 1, 4)
# f' = (-16, -5, 4)
e_target = naive_matrix @ np.array([e[0], e[1], e[2]])
f_target = naive_matrix @ np.array([f[0], f[1], f[2]])


print("e' = ", e_target)
print("f' = ", f_target)

print_bar()

# Testing dlt algorithm with all points
start_dlt = [a, b, c, d, e, f]
target_dlt = [a_target, b_target, c_target, d_target, e_target, f_target]

dlt_matrix = np.array(dlt(start_dlt, target_dlt))

print_blue("Raw dlt matrix:")
print(dlt_matrix)

dlt_matrix = np.round(dlt_matrix, 5)

print_blue("Rounded dlt matrix:")
print(dlt_matrix)

for i in range(3):
    for j in range(3):
        if naive_matrix[i, j] != 0 and dlt_matrix[i, j] != 0:
            dlt_matrix = dlt_matrix / dlt_matrix[i, j] * naive_matrix[i, j]

print_blue("Showing that after division and multiplying, \ndlt matrix is the same as the naive one:")
print(dlt_matrix)

print_bar()

# Testing modified dlt
dlt_modified_matrix = dlt_modified(start_dlt, target_dlt)

print_blue("Raw modified dlt matrix:")
print(dlt_modified_matrix)

dlt_modified_matrix = np.round(dlt_modified_matrix, 5)

print_blue("Rounded modified dlt matrix:")
print(dlt_modified_matrix)

for i in range(3):
    for j in range(3):
        if naive_matrix[i, j] != 0 and dlt_modified_matrix[i, j] != 0:
            dlt_modified_matrix = dlt_modified_matrix / dlt_modified_matrix[i, j] * naive_matrix[i, j]

print_blue("Showing that after division and multiplying, modified dlt\nmatrix is "
           "the same as the naive and regular dlt matrix:")
print(dlt_modified_matrix)

print_bar()

print_red("Defending the project:")
print_bar()

print_blue_and_yellow("1.", "Comparing the matrices of DLT with 4 points and with 5,\n"
                            "showing they slightly differ:")

start_4 = [a, b, c, d]
target_4 = [a_target, b_target, c_target, d_target]
dlt_4_matrix = dlt(start_4, target_4)

start_5 = [a, b, c, d, e]
target_5 = [a_target, b_target, c_target, d_target, e_target]
dlt_5_matrix = dlt(start_5, target_5)

dlt_4_matrix = np.round(dlt_4_matrix, 5)
dlt_5_matrix = np.round(dlt_5_matrix, 5)

print_blue("4 points matrix:")
print(dlt_4_matrix)
print_blue("5 points matrix:")
print(dlt_5_matrix)

print_blue_and_yellow("2.", "Seeing that matrices are the same after switching\n"
                            "positions of two points:")

start = [a, b, c, d]
target = [a_target, b_target, c_target, d_target]
dlt_matrix = dlt(start, target)

start_switched = [b, a, c, d]
target_switched = [b_target, a_target, c_target, d_target]
dlt_matrix_switched = dlt(start, target)

dlt_matrix = np.round(dlt_matrix, 5)
dlt_matrix_switched = np.round(dlt_matrix_switched, 5)

print_blue("Matrix before switching:")
print(dlt_matrix)
print_blue("Matrix after switching:")
print(dlt_matrix_switched)

print_blue_and_yellow("3.", "Seeing that DLT is not invariant on coordinates change:")

T_before = np.array([[0, 1, 2], [-1, 0, 3], [0, 0, 1]])
T_after = np.array([[1, -1, 5], [1, 1, -2], [0, 0, 1]])

a_old_t = T_before @ np.array([-3, -1, 1])
b_old_t = T_before @ np.array([3, -1, 1])
c_old_t = T_before @ np.array([1, 1, 1])
d_old_t = T_before @ np.array([-1, 1, 1])

a_new_t = T_after @ np.array([-2, -1, 1])
b_new_t = T_after @ np.array([2, -1, 1])
c_new_t = T_after @ np.array([2, 1, 1])
d_new_t = T_after @ np.array([-2, 1, 1])

start_t = [b_old_t, a_old_t, c_old_t, d_old_t]
target_t = [b_new_t, a_new_t, c_new_t, d_new_t]

matrix_t = dlt(start_t, target_t)
matrix_t = np.round(matrix_t, 5)


print_blue("Transformation matrix for old coordinates:")
print(T_before)

print_blue("Transformation matrix for new coordinates:")
print(T_after)

print_blue("Dlt matrix before transformation:")
print(dlt_4_matrix)

print_blue("Dlt matrix after transformation:")
print(matrix_t)

print_blue_and_yellow("4.", "Seeing that modified DLT is not invariant on coordinates change:")

dlt_modified_matrix = dlt_modified(start, target)
dlt_modified_matrix_t = dlt_modified(start_t, target_t)

dlt_modified_matrix_t = np.round(dlt_modified_matrix_t, 5)
dlt_modified_matrix = np.round(dlt_modified_matrix, 5)

print_blue("Dlt modified matrix before transformation:")
print(dlt_4_matrix)

print_blue("Dlt modified matrix after transformation:")
print(matrix_t)

print_bar()




