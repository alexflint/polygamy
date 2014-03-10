__author__ = 'alexflint'

from fractions import Fraction
import numpy as np

from polynomial import *
from spline import *
import action
import multivariate


def is_squarefree(fs, zeros):
    J = polynomial_jacobian(fs)
    for zero in zeros:
        Jcur = J(*zero)
        print Jcur


def inner(a, b):
    return sum(ai*bi for ai, bi in zip(a, b))

def inner(a, b):
    return [[ai*bj for bj in b] for ai in a]

def dot(A, x):
    return [sum(Aij * xj for Aij, xj in zip(Ai, x)) for Ai in A]

def main():
    np.random.seed(123)
    ordering = GrevlexOrdering()

    # Construct symbolic problem
    time = Polynomial.coordinate(0, 5)
    params = [ Polynomial.coordinate(i+1, 5, Fraction) for i in range(4) ]

    p = evaluate_bezier(params, time)
    v = p.partial_derivative(0)
    a = v.partial_derivative(0)

    # Sample ground truth
    true_times = [ Fraction(1,8), Fraction(4,8), Fraction(6,8), Fraction(7,8), Fraction(5,8) ]
    true_params = [2, 1, 3, -5]
    nv = len(true_params)

    true_ys = [ p(t, *true_params) for t in true_times ]
    noisy_ys = [ y + Fraction(np.random.randint(-3, 3), 10) for y in true_ys ]

    print 'True ys:',true_ys
    print 'Noisy ys:',noisy_ys

    # Construct the polynomial system for the minimal problem version
    minimal_system = []
    for t, y in zip(true_times, true_ys):
        g = p.evaluate_partial(0, t).drop(0)
        minimal_system.append(p.evaluate_partial(0, t).drop(0) - y)

    print '\nMinimal system:'
    for f in minimal_system:
        print '  ', f

    # Construct the least squares spline cost
    spline_cost = Polynomial(nv, Fraction)
    for t, y in zip(true_times, noisy_ys):
        residual = p.evaluate_partial(0, t).drop(0) - y
        spline_cost += residual**2

    # Construct the least squares rotation estimation cost
    s = [ Polynomial.coordinate(i, 3, Fraction) for i in range(3) ]
    Q = constdiag(1 - inner(s,s), 3)


    cost = cayley_cost
    optimization_system = [cost.partial_derivative(i) for i in range(nv)]
    print '\nOptimization cost:', cost
    print '\nOptimization system:'
    for f in optimization_system:
        print '  ', f



    F = optimization_system

    # Evaluate at (hopefully unique) zero
    fun = polynomial_vector(F)
    print '\nEvaluated at zero:'
    print fun(*true_params)
    print 'Jacobian at zero:'
    print polynomial_jacobian(F)(*true_params)

    # Compute grobner basis
    print 'Computing grobner basis...'
    G = gbasis(F, ordering, limit=100)
    print '\nGrobner basis:'
    for g in G:
        print '  ',g

    # Compute action matrix
    soln = []
    for i in range(nv):
        coord = Polynomial.coordinate(i, nv, Fraction)
        M = action.action_matrix_from_grobner_basis(coord, G, ordering)
        eigvals,eigvecs = np.linalg.eig(M.astype(float))
        print 'Coordinate %d:' % i
        print eigvals

        if len(eigvals) == 1:
            soln.append(eigvals[0])

    report = multivariate.polish_root(optimization_system, soln)
    print '\nPolished solution:'
    print report.x

    print 'p: ', p
    print 'v: ', v
    print 'a: ', a


if __name__ == '__main__':
    main()
