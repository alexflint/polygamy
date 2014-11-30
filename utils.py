import StringIO
from fractions import Fraction
import numpy as np

from polynomial import Polynomial


def skew(m):
    """Return the matrix form of the cross product for the 3-vector m."""
    m0, m1, m2 = m
    return np.array([[0.,  -m2,  m1],
                     [m2,   0., -m0],
                     [-m1,  m0,  0.]])


def skew_jacobian():
    """Return the jacobian of flatten(skew(x)) with respect to x."""
    return np.array([[0,  0,  0],
                     [0,  0, -1],
                     [0,  1,  0],
                     [0,  0,  1],
                     [0,  0,  0],
                     [-1, 0,  0],
                     [0, -1,  0],
                     [1,  0,  0],
                     [0,  0,  0]])


def array_str(arr):
    strings = []
    maxlen = 0
    for row in arr:
        rowstrings = []
        for elem in row:
            if isinstance(elem, Polynomial):
                s = elem.format(compact=True)
            else:
                s = unicode(elem)
            rowstrings.append(s)
            maxlen = max(maxlen, len(s))
        strings.append(rowstrings)

    ss = StringIO.StringIO()
    ss.write('[')
    for i, rowstrings in enumerate(strings):
        if i > 0:
            ss.write('\n ')
        ss.write('[')
        for j, s in enumerate(rowstrings):
            if j > 0:
                ss.write('  ')
            ss.write(s)
            ss.write(' '*(maxlen-len(s)))
        ss.write(']')
    ss.write(']')
    return ss.getvalue()


def arraymap(f, arr):
    arr = np.asarray(arr)
    return np.array([f(x) for x in arr.flat]).reshape(arr.shape)


def astype(x, ctype):
    return arraymap(lambda xi: xi.astype(ctype) if isinstance(xi, Polynomial) else ctype(xi), x)


def asfraction(x, denom_limit=None):
    return arraymap(lambda xi: Fraction(xi).limit_denominator(denom_limit), x)


def evaluate_array(arr, *args, **kwargs):
    return arraymap(lambda p: p(*args, **kwargs), arr)


def flatten(x):
    flat = []
    for xi in x:
        if isinstance(xi, Polynomial):
            flat.append(xi)
        else:
            try:
                flat.extend(xi)
            except TypeError:
                flat.append(xi)
    return flat


def cayley_mat(s):
    return np.eye(3) * (1. - np.dot(s, s)) + 2.*skew(s) + 2.*np.outer(s, s)


def cayley_denom(s):
    return 1. + np.dot(s, s)


def cayley(s):
    return cayley_mat(s) / cayley_denom(s)


def normalized(x):
    x = np.asarray(x)
    return x / np.linalg.norm(x)


def essential_matrix(R1, p1, R2, p2):
    Rrel = np.dot(R2, R1.T)
    prel = np.dot(R1, p2-p1)
    return essential_matrix_from_relative_pose(Rrel, prel)


def essential_matrix_from_relative_pose(Rrel, prel):
    return np.dot(Rrel, skew(prel))


def list_depth(x):
    """Determine how many levels of nested lists are inside x."""
    while True:
        try:
            if len(x) == 0:
                return 1
            else:
                return max(map(list_depth, x)) + 1
        except TypeError:
            return 0
