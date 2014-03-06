__author__ = 'Alex Flint'

def evaluate_bezier(params, t):
    if len(params) == 1:
        return params[0]
    else:
        return evaluate_bezier(params[:-1], t) * (1-t) + evaluate_bezier(params[1:], t) * t
