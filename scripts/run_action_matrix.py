__author__ = 'alexflint'

import itertools
import operator

import numpy as np

import ring
from polynomial import *
from action import *


def dot(xs, ys, num_vars):
    return sum(as_polynomial(x,num_vars) * as_polynomial(y,num_vars) for x, y in zip(xs,ys))


def coefficients(polynomial, monomials):
    return np.array([polynomial[m] for m in monomials])


def from_coefficients(coefficients, monomials):
    return dot(coefficients, monomials, len(monomials[0]))


def check_action_matrix(Mp, p, c, monomials, grobner_basis, ordering):
    """Return true if Mp is the action matrix for p as evaluated at
    the coefficient vector c and given basis monomials."""
    q1 = remainder(p * from_coefficients(c, monomials), grobner_basis, ordering)
    q2 = from_coefficients(np.dot(Mp, c), monomials)
    return q1 == q2


def squarefree_ideal(F):
    n = F[0].num_vars
    F_ext = [ f.copy() for f in F ]
    for i in range(n):
        var_order = [i] + [j for j in range(n) if j != i]
        G = gbasis(F, VariableReordering(var_order, LexOrdering()))
        monic_generator = None
        for g in G:
            ks = filter(lambda k: any(term.monomial[k] for term in g), range(n))
            if len(ks) == 1 and ks[0] == i:
                monic_generator = g.normalized(GrevlexOrdering())
                break
        assert monic_generator is not None
        print 'monic generator: ', monic_generator
        sqf = squarefree(monic_generator, var=i, ordering=GrevlexOrdering()).normalized(GrevlexOrdering())
        print '  square free: ', sqf
        if sqf != monic_generator:
            print 'appending'
            F_ext.append(sqf)
    return F_ext


def squarefree(f, var=0, ordering=GrevlexOrdering()):
    f = f.astype(fractions.Fraction)
    fp = f.partial_derivative(var)
    def mod(a, b):
        q, r = a.divide_by(b, ordering)
        return r
    q, r = f.divide_by(ring.gcd(f, fp, mod), ordering)
    assert r == 0
    return q


def main():
    np.set_printoptions(precision=5, suppress=True, linewidth=200)

    zeros = np.array([[-2, -1],
                      [3,  2],
                      [4,  5],
                      [6,  7]], dtype=fractions.Fraction)

    num_vars = zeros.shape[1]
    ordering = GrevlexOrdering()
    p = Polynomial.coordinate(0, num_vars, ctype=fractions.Fraction)

    # Setup a mock ideal
    F = ideal_from_variety(zeros, fractions.Fraction)
    print '\nIdeal:'
    print '\n'.join(map(str, F))

    # Evaluate on the variety to confirm zeros
    print '\nEvaluated at zeros:'
    fun = polynomial_vector(F)
    for zero in zeros:
        print 'I(%s) = %s' % (','.join(map(str, zero)), fun(*zero))

    # Evaluate jacobian on the variety
    J = polynomial_jacobian(F)
    for zero in zeros:
        print '\nJacobian at %s:' % zero
        print J(*zero)

    # Set up a grobner basis
    print '\nGrobner basis:'
    G = gbasis(F, ordering)
    print '\n'.join(map(str, G))

    # Construct a linear basis for the quotient algebra
    basis_monomials = quotient_algebra_basis(G, ordering)
    print '\nQuotient algebra monomials:'
    print [ Term(1, monomial) for monomial in basis_monomials ]

    # Construct the action matrix for x
    M = action_matrix_from_grobner_basis(p, G, ordering).astype(float)
    print 'Multiplication matrix for %s:' % p
    print M

    # Check the action matrix
    for i in range(10):
        c = np.random.randint(0, 10, size=len(basis_monomials))
        check_action_matrix(M, p, c, basis_monomials, G, ordering)

    eigvals,eigvecs = np.linalg.eig(M)
    print 'Eigenvalues:'
    print eigvals
    print 'Eigenvectors:'
    print eigvecs / eigvecs[0]


if __name__ == '__main__':
    main()
