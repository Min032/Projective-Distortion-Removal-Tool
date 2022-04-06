import dlt_algorithm
import u_tools as ut

import numpy as np


def get_normalization_matrix(points):
    points_afine = np.array([(p[0]/p[2], p[1]/p[2], 1) for p in points])
    c = points_afine.mean(axis=0)

    G = np.array([[1, 0, -c[0]],
                  [0, 1, -c[1]],
                  [0, 0, 1]])

    points_afine = [(G @ np.array(point)) for point in points_afine]
    dist = [np.sqrt(p[0]*p[0]+p[1]*p[1]) for p in points_afine]

    dist_average = sum(dist) / len(points_afine)
    k = np.sqrt(2) / dist_average

    S = np.array([[k, 0, 0],
                  [0, k, 0],
                  [0, 0, 1]])

    return S @ G


def dlt_modified(before, after):
    T_before = get_normalization_matrix(before)
    before = np.array([(T_before @ np.array(p)) for p in before])

    T_after = get_normalization_matrix(after)
    after = np.array([(T_after @ np.array(p)) for p in after])

    P = dlt_algorithm.dlt(before, after)

    return np.negative(np.linalg.inv(T_after) @ P @ T_before)


def dlt_modified_and_homogenize(before, after):
    before = ut.homogenize_dlt(before)
    after = ut.homogenize_dlt(after)

    return dlt_modified(before, after)

