import numpy as np


def bezier(params, t):
    """Evaluate a bezier curve at time t (between 0 and 1)"""
    return np.dot(bezier_coefs(t, len(params)-1), params)


def bezier_coefs(t, order):
    """Evaluate a bezier curve at time t (between 0 and 1)"""
    if order == 0:
        return np.array([1.])
    else:
        c = bezier_coefs(t, order-1)
        return np.hstack((c, 0)) * (1.-t) + np.hstack((0, c)) * t


def bezier_deriv(params, t):
    if len(params) == 1:
        return 0.
    else:
        a = bezier(params[:-1], t)
        b = bezier(params[1:], t)
        aderiv = bezier_deriv(params[:-1], t)
        bderiv = bezier_deriv(params[1:], t)
        return aderiv*(1.-t) + bderiv*t - a + b


def bezier_second_deriv(params, t):
    if len(params) == 1:
        return 0.
    else:
        aderiv = bezier_deriv(params[:-1], t)
        bderiv = bezier_deriv(params[1:], t)
        aderiv2 = bezier_second_deriv(params[:-1], t)
        bderiv2 = bezier_second_deriv(params[1:], t)
        return aderiv2*(1.-t) + bderiv2*t - aderiv*2. + bderiv*2.


def zero_offset_bezier(params, t):
    if t == 0:
        # should return a numpy array for t=0, not a Polynomial
        return np.zeros_like(params[0])
    else:
        return bezier(np.vstack((np.zeros(len(params[0])), params)), t)


def zero_offset_bezier_second_deriv(params, t):
    return bezier_second_deriv(np.vstack((np.zeros(len(params[0])), params)), t)
