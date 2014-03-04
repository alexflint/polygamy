__author__ = 'alexflint'

import itertools
import operator

import numpy as np

from polynomial import *

def product(*xs):
    return reduce(operator.mul, xs)

def ideal_intersection(*Fs):
    return [ f for F in Fs for f in F ]

def ideal_union(*Fs):
    return [ product(*F) for F in itertools.product(*Fs) ]

def coordinate_poly(k, num_vars, ctype=None):
    return Polynomial.create([Term(1, tuple(i==k for i in range(num_vars)))], num_vars, ctype)

def ideal_from_zero(zero, ctype=None):
    '''Construct an ideal that vanishes at the given zero.'''
    num_vars = len(zero)
    return [ coordinate_poly(i, len(zero), ctype) - zi for i,zi in enumerate(zero) ]

def ideal_from_variety(zeros, ctype=None):
    '''Construct an ideal from a finite variety.'''
    return ideal_union(*[ideal_from_zero(zero, ctype) for zero in zeros])

def quotient_monomials(fs, ordering):
    """Find the set of monomials not divisible by any leading term in fs."""
    # Get the list of leading terms
    leading_terms = [ f.leading_term(ordering) for f in fs ]

    # Find the univariate leading terms
    rect = [None] * fs[0].num_vars
    for lt in leading_terms:
        active_vars = [ (i,a) for i,a in enumerate(lt.monomial) if a>0 ]
        if len(active_vars) == 1:
            i,a = active_vars[0]
            if rect[i] is None or rect[i] < a:
                rect[i] = a

    # Is the quotient algebra finite dimensional?
    if any(ri is None for ri in rect):
        return None

    # Find monomials not divisble by the basis
    output = []
    for candidate in list(itertools.product(*map(range, rect))):
        if not any(lt.divides(candidate) for lt in leading_terms):
            output.append(candidate)
    return output



if __name__ == '__main__':
    zeros = np.array([[-2, -1],
                      [3,  2],
                      [4,  5]])

    num_vars = zeros.shape[1]
    ordering = GrevlexOrdering()

    # Setup a mock ideal
    F = ideal_from_variety(zeros, float)
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
    basis_monomials = quotient_monomials(G, ordering)
    print '\nQuotient algebra monomials:'
    print [ as_term(monomial,num_vars) for monomial in basis_monomials ]

    # Construct the action matrix for x
    p = coordinate_poly(1, num_vars) + coordinate_poly(0, num_vars)
    remainders = [ remainder(p*monomial, G, ordering) for monomial in basis_monomials ]
    print 'Remainders:'
    for monomial,remainder in zip(basis_monomials,remainders):
        print '  rem(%s) = %s' % (p*monomial, remainder)

    M,X = matrix_form(remainders, basis_monomials)
    print 'Multiplication matrix for %s:' % p
    print M

    import numpy as np
    np.set_printoptions(precision=5, suppress=True, linewidth=200)

    eigvals,eigvecs = np.linalg.eig(M)
    print 'Eigenvalues:'
    print eigvals
    print 'Eigenvectors:'
    print eigvecs
    print eigvecs / eigvecs[0]
