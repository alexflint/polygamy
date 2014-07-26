import numpy as np


def evaluate_bezier(params, t):
    """Evaluate a bezier curve at time t (between 0 and 1)"""
    if len(params) == 1:
        return params[0]
    else:
        return evaluate_bezier(params[:-1], t) * (1.-t) + evaluate_bezier(params[1:], t) * t


def evaluate_bezier_deriv(params, t):
    if len(params) == 1:
        return 0.
    else:
        a = evaluate_bezier(params[:-1], t)
        b = evaluate_bezier(params[1:], t)
        aderiv = evaluate_bezier_deriv(params[:-1], t)
        bderiv = evaluate_bezier_deriv(params[1:], t)
        return aderiv*(1.-t) + bderiv*t - a + b


def evaluate_bezier_second_deriv(params, t):
    if len(params) == 1:
        return 0.
    else:
        aderiv = evaluate_bezier_deriv(params[:-1], t)
        bderiv = evaluate_bezier_deriv(params[1:], t)
        aderiv2 = evaluate_bezier_second_deriv(params[:-1], t)
        bderiv2 = evaluate_bezier_second_deriv(params[1:], t)
        return aderiv2*(1.-t) + bderiv2*t - aderiv*2. + bderiv*2.


def evaluate_zero_offset_bezier(params, t):
    if t == 0:
        # should return a numpy array for t=0, not a Polynomial
        return np.zeros_like(params[0])
    else:
        return evaluate_bezier(np.vstack((np.zeros(len(params[0])), params)), t)


def evaluate_zero_offset_bezier_second_deriv(params, t):
    return evaluate_bezier_second_deriv(np.vstack((np.zeros(len(params[0])), params)), t)
