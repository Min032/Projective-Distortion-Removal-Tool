def homogenize_naive(points):
    A, B, C, D = points[0], points[1], points[2], points[3]
    return (A[0], A[1], 1), (B[0], B[1], 1), (C[0], C[1], 1), (D[0], D[1], 1)


def homogenize_dlt(points):
    homogenized_points = []
    for p in points:
        homogenized_points.append((p[0], p[1], 1))
    return homogenized_points
