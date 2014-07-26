from polynomial import Polynomial, parse
from compilation import function_from_expression

import numpy as np


def write_polynomials(polynomials, path):
    if isinstance(polynomials, Polynomial):
        polynomials = [polynomials]
    with open(path, 'w') as fd:
        for polynomial in polynomials:
            fd.write(polynomial.format(use_superscripts=False) + ';\n')


def write_solution(values, path):
    with open(path, 'w') as fd:
        for i, xi in enumerate(values):
            fd.write('x%d %.12f\n' % (i, xi))


def load_polynomials(path):
    return parse(*[line.strip('; \n').replace('^', '**') for line in open(path)])


def load_functions(path, varnames):
    return [function_from_expression(line.strip('; \n').replace('^', '**'), varnames)
            for line in open(path)]


def load_solution(path):
    varnames = []
    solution = []
    with open(path) as fd:
        for line in fd:
            var, value = line.strip().split()
            varnames.append(var)
            solution.append(float(value))
    return varnames, np.array(solution)
