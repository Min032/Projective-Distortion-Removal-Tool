import numpy as np
import u_tools as ut


def get_matrix(points):
    aA, aB, aC, aD = points

    # Equation: AX = B
    B = np.array([aD[0], aD[1], aD[2]])
    A = np.array([[aA[0], aB[0], aC[0]],
                  [aA[1], aB[1], aC[1]],
                  [aA[2], aB[2], aC[2]]])

    X = np.linalg.solve(A, B)

    if np.linalg.matrix_rank(A) <= 0:
        return np.array([])
    else:
        return np.array([[aA[0]*X[0], aB[0]*X[1], aC[0]*X[2]],
                         [aA[1]*X[0], aB[1]*X[1], aC[1]*X[2]],
                         [aA[2]*X[0], aB[2]*X[1], aC[2]*X[2]]])


def naive(before, after):
    finish = get_matrix(after)
    start = np.linalg.inv(get_matrix(before))
    return finish @ start


def naive_and_homogenize(before, after):
    before = ut.homogenize_naive(before)
    after = ut.homogenize_naive(after)

    return naive(before, after)
