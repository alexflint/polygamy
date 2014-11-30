__author__ = 'alexflint'

__author__ = 'alexflint'

import StringIO
from fractions import Fraction
import numpy as np

from polynomial import *
from bezier import *
import action
import multivariate


def skew(m):
    m = np.asarray(m)
    return np.array([[  0,    -m[2],  m[1] ],
                     [  m[2],  0,    -m[0] ],
                     [ -m[1],  m[0],  0.   ]])


def is_squarefree(fs, zeros):
    J = polynomial_jacobian(fs)
    for zero in zeros:
        Jcur = J(*zero)
        print Jcur


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
    for i,rowstrings in enumerate(strings):
        if i > 0:
            ss.write('\n ')
        ss.write('[')
        for j,s in enumerate(rowstrings):
            if j > 0:
                ss.write('  ')
            ss.write(s)
            ss.write(' '*(maxlen-len(s)))
        ss.write(']')
    ss.write(']')
    return ss.getvalue()

def arraymap(f, x):
    if isinstance(x, Polynomial):
         return f(x)
    else:
        try:
            return np.array(map(lambda xi: arraymap(f, xi), x))
        except:
            return f(x)

def astype(x, ctype):
    return arraymap(lambda xi: xi.astype(ctype) if isinstance(xi, Polynomial) else ctype(xi), x)

def asfraction(x, denom_limit=None):
    return arraymap(lambda xi: Fraction(xi).limit_denominator(denom_limit), x)


def flatten(x):
    L = []
    for xi in x:
        if isinstance(xi, Polynomial):
            L.append(xi)
        else:
            try:
                L.extend(xi)
            except:
                L.append(xi)
    return L


def main():
    np.random.seed(123)
    np.set_printoptions(precision=5, suppress=True, linewidth=300)
    ordering = GrevlexOrdering()

    def cayley_mat(s):
        return np.eye(3) * (1 - np.dot(s,s)) + 2*skew(s) + 2*np.outer(s,s)

    def cayley_denom(s):
        return 1 + np.dot(s,s)

    def cayley(s):
        return cayley_mat(s) / cayley_denom(s)

    # Construct symbolic problem
    nv = 3
    params = [Polynomial.coordinate(i, nv, Fraction) for i in range(nv)]

    true_params = [1, 3, 4]

    residuals = flatten(np.outer(params, params) - np.outer(true_params, true_params))

    auxiliary_system = []

    print '\nResiduals:'
    for r in residuals:
        print '  ', r

    cost = sum(r**2 for r in residuals)
    optimization_system = [cost.partial_derivative(i) for i in range(nv)] + auxiliary_system
    print '\nCost being optimized:', cost
    print '\nOptimization system:'
    for f in optimization_system:
        print '  ', f

    # Evaluate at (hopefully unique) zero
    fun = polynomial_vector(optimization_system)
    print '\nEvaluated at ground truth:'
    print fun(*true_params)

    # Compute grobner basis
    print '\nComputing grobner basis...'
    G = gbasis(optimization_system, ordering, limit=100)
    print '  Grobner basis has size %d' % len(G)

    print 'Checking grobner basis...'
    assert is_grobner_basis(G, ordering)

    print 'Grobner basis:'
    for g in G:
        print '  ', g

    print 'Grobner basis leading terms:'
    for g in G:
        print '  ', as_polynomial(g.leading_term(ordering). monomial)

    # Compute action matrix
    print '\nSolving...'
    soln = []
    for i in range(0,nv):
        coord = Polynomial.coordinate(i, nv, Fraction)
        M = action.action_matrix_from_grobner_basis(coord, G, ordering)

        eigvals, eigvecs = np.linalg.eig(M.astype(float))
        print 'Coordinate %d:' % i
        print eigvals

        if len(eigvals) == 1:
            soln.append(eigvals[0])

    #report = multivariate.polish_root(optimization_system, soln)
    #print '\nPolished solution:'
    #print report.x


if __name__ == '__main__':
    main()
