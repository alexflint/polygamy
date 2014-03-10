__author__ = 'alexflint'


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


def action_matrix_from_grobner_basis(p, G, ordering):
    # Construct a linear basis for the quotient algebra
    basis_monomials = quotient_algebra_basis(G, ordering)

    # Construct the action matrix for x
    remainders = [remainder(p*monomial, G, ordering) for monomial in basis_monomials]
    M, B = matrix_form(remainders, basis_monomials)
    return M.T
