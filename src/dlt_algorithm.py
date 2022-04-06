import numpy as np
import u_tools as ut


def get_2_x_9_matrix(points):
    A, B = points
    matrix = np.array([
        [0, 0, 0, -B[2] * A[0], -B[2] * A[1], -B[2] * A[2], B[1] * A[0], B[1] * A[1], B[1] * A[2]],
        [B[2] * A[0], B[2] * A[1], B[2] * A[2], 0, 0, 0, -B[0] * A[0], -B[0] * A[1], -B[0] * A[2]]
    ])

    return matrix


def dlt(before, after):
    matrix = np.array([]).reshape(0, 9)

    for i in range(0, len(before)):
        M = get_2_x_9_matrix((before[i], after[i]))
        matrix = np.concatenate((matrix, M))

    u, s, vh = np.linalg.svd(matrix)
    matrix = np.transpose(vh)

    P = matrix[:, -1]
    P = P.reshape(3, 3)

    return P


def dlt_and_homogenize(before, after):
    before = ut.homogenize_dlt(before)
    after = ut.homogenize_dlt(after)

    return dlt(before, after)

