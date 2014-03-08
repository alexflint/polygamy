__author__ = 'alexflint'

import itertools
import operator

import numpy as np

import ring
from polynomial import *


class QuotientAlgebraError(Exception):
    pass


def monomials_not_divisible_by(monomials):
    """Find all monomials not divisible by any monomial in M.
    Raises QuotientAlgebraError if there are an infinite
    number of such monomials."""
    if len(monomials) == 0:
        raise QuotientAlgebraError('list of leading monomials was empty')

    # Find the univariate leading terms
    rect = [None] * len(monomials[0])
    for monomial in monomials:
        active_vars = [ (i,a) for i,a in enumerate(monomial) if a>0 ]
        if len(active_vars) == 1:
            i,a = active_vars[0]
            if rect[i] is None or rect[i] > a:
                rect[i] = a

    # Is the quotient algebra finite dimensional?
    if any(ri is None for ri in rect):
        raise QuotientAlgebraError('quotient algebra does not have a finite basis')

    # Find monomials not divisble by the basis
    output = []
    for candidate in itertools.product(*map(range, rect)):
        if not any(can_divide_monomial(candidate, m) for m in monomials):
            output.append(candidate)
    return output


def quotient_algebra_basis(fs, ordering):
    """Find the set of monomials not divisible by any leading term in fs."""
    # Get the list of leading terms
    leading_monomials = [ f.leading_term(ordering).monomial for f in fs ];
    return monomials_not_divisible_by(leading_monomials)


def dot(xs, ys, num_vars):
    return sum(as_polynomial(x,num_vars) * as_polynomial(y,num_vars) for x,y in zip(xs,ys))


def coefficients(polynomial, monomials):
    return np.array([polynomial[m] for m in monomials])


def from_coefficients(coefficients, monomials):
    return dot(coefficients, monomials, len(monomials[0]))


def check_action_matrix(Mp, p, c, monomials, grobner_basis, ordering):
    """Return true if Mp is the action matrix for p as evaluated at
    the coefficient vector c and given basis monomials."""
    q1 = remainder(p * from_coefficients(c, monomials), grobner_basis, ordering)
    q2 = from_coefficients(np.dot(Mp, c), monomials)
    success = q1 == q2
    print 'q1:',q1
    print 'q2:',q2
    print 'equal?', success
    return success


def squarefree(f, var=0, ordering=GrevlexOrdering()):
    f = f.astype(fractions.Fraction)
    fp = f.partial_derivative(var)
    def mod(a, b):
        q, r = a.divide_by(b, ordering)
        return r
    q, r = f.divide_by(ring.gcd(f, fp, mod), ordering)
    assert r == 0
    return q


class VariableReordering(MonomialOrdering):
    def __init__(self, var_ordering, monomial_ordering):
        self._vars = var_ordering
        self._inner = monomial_ordering
    def __call__(self, a, b):
        aa = tuple(a[i] for i in self._vars)
        bb = tuple(b[i] for i in self._vars)
        return self._inner(aa, bb)


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
        print 'monic generator: ',monic_generator
        sqf = squarefree(monic_generator, var=i, ordering=GrevlexOrdering()).normalized(GrevlexOrdering())
        print '  square free: ',sqf
        if sqf != monic_generator:
            print 'appending'
            F_ext.append(sqf)
    return F_ext


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

    # Set up a grobner basis
    print '\nGrobner basis:'
    G = gbasis(F, ordering)
    print '\n'.join(map(str, G))

    # Construct a linear basis for the quotient algebra
    basis_monomials = quotient_algebra_basis(G, ordering)
    print '\nQuotient algebra monomials:'
    print [ Term(1,monomial) for monomial in basis_monomials ]

    # Construct the action matrix for x
    rem = lambda f: remainder(f, G, ordering)
    remainders = [ rem(p*monomial) for monomial in basis_monomials ]
    M,B = matrix_form(remainders, basis_monomials)
    M = M.T.astype(float)


    print 'Remainders:'
    for m,r in zip(basis_monomials,remainders):
        print '  rem(%s) = %s' % (p*m, r)

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
