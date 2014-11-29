import itertools
import fractions
import numpy as np
import scipy.linalg

import echelon
from polynomial import Polynomial, Term, evaluate_monomial, gbasis, matrix_form, GrevlexOrdering, LexOrdering,\
    as_term, as_polynomial


class SolutionSet(object):
    def __init__(self, solutions):
        self.solutions = solutions


class PolynomialSolverError(Exception):
    pass


def spy(a, threshold=1e-8):
    a = np.atleast_2d(a)
    return '\n'.join('[' + ''.join('x' if abs(x) > threshold else ' ' for x in row) + ']'
                     for row in a)


def magdigit(x):
    if abs(x) < 1e-9:
        return ' '
    elif abs(x) > .1:
        return 'x'
    else:
        return str(int(np.floor(-np.log10(abs(x)))))


def spymag(a):
    a = np.atleast_2d(a)
    return '\n'.join('[' + ''.join(map(magdigit, row)) + ']'
                     for row in a)


def permutation_matrix(p):
    a = np.zeros((len(p), len(p)), int)
    for i, pi in enumerate(p):
        a[pi,i] = 1
    return a


def evaluate_poly_vector(v, x, dtype):
    return np.array([vi(*x) for vi in v], dtype)


def solve_monomial_equations(monomials, values):
    """Given a set of monomials and their values at some point (x1,..,xn), compute
    the possible values for x1,..,xn at that point (there will be up to 2^n
    possibilities)."""
    assert all(isinstance(m, tuple) for m in monomials)
    a = np.asarray(monomials, float)

    # First test whether each variable is present on its own
    naked_indices = [None] * len(monomials[0])
    for i, monomial in enumerate(monomials):
        if sum(monomial) == 1:
            naked_indices[monomial.index(1)] = i

    if all(i is not None for i in naked_indices):
        print '  Solved monomial equations the simple way'
        yield np.take(values, naked_indices)
        return

    if np.any(np.abs(values) < 1e-8):
        print 'Warning: some values were zero, cannot solve for these'
        return

    log_x, residuals, rank, svs = np.linalg.lstsq(a, np.log(np.abs(values)))
    if rank < a.shape[1]:
        print 'Warning: rank defficient monomial equations (incomplete basis?)'
        return

    if np.linalg.norm(residuals) < 1e-6:
        abs_x = np.exp(log_x)
        for signs in itertools.product((-1, 1), repeat=len(abs_x)):
            x = abs_x * signs
            err = sum((evaluate_monomial(m, x) - y)**2 for m, y in zip(monomials, values))
            if err < 1e-6:
                yield x


def solve_via_basis_selection(equations, expansion_monomials, lambda_poly, solutions=None, include_grobner=False):
    nvars = lambda_poly.num_vars

    print 'Equations:'
    for f in equations:
        print '  ', f
        if solutions is not None:
            for solution in solutions:
                print '    = %f at %s' % (f(*solution), solution)

    # Expand equations
    expanded_equations = list(equations)
    for f, expansions in zip(equations, expansion_monomials):
        for monomial in expansions:
            expanded_equations.append(f * monomial)

    if include_grobner:
        print 'Grobner basis:'
        gb_equations = [p.astype(fractions.Fraction) for p in equations]
        for f in gbasis(gb_equations, GrevlexOrdering()):
            print '  ', f
            expanded_equations.append(f)

    print 'Expanded equations:'
    for f in expanded_equations:
        print '  ', f
        if solutions is not None:
            for solution in solutions:
                print '    = %f at %s' % (f(*solution), solution)

    present = set(term.monomial for f in expanded_equations for term in f)
    original = set(term.monomial for f in equations for term in f)

    # Compute permissible monomials
    permissible = set()
    for m in original:
        p_m = lambda_poly * m
        if all(mi in present for mi in p_m.monomials):
            permissible.add(m)

    # Compute required monomials
    required = set()
    for m in permissible:
        p_m = lambda_poly * m
        for mi in p_m.monomials:
            if mi not in permissible:
                required.add(mi)

    nuissance = list(present.difference(set.union(permissible, required)))
    required = list(required)
    permissible = list(permissible)

    nn = len(nuissance)
    nr = len(required)
    npe = len(permissible)
    num_remaining = len(expanded_equations) - len(required) - len(nuissance)

    print 'Present monomials:', ', '.join(map(str, map(Term.from_monomial, present)))
    print 'Permissible monomials:', ', '.join(map(str, map(Term.from_monomial, permissible)))
    print 'Required monomials:', ', '.join(map(str, map(Term.from_monomial, required)))
    print 'Nuissance monomials:', ', '.join(map(str, map(Term.from_monomial, nuissance)))
    print 'Num equations:', len(expanded_equations)
    print 'Num equations after eliminating:', num_remaining

    if len(permissible) <= nvars:
        raise PolynomialSolverError('There are fewer permissible monomials than variables. Add more expansions.')

    if num_remaining <= 0:
        raise PolynomialSolverError('The number of required plus nuissance monomials exceeds the number of equations. '
                                    'Add more expansions.')

    # Construct the three column blocks from the expanded equations
    c_nuissance, x_nuissance = matrix_form(expanded_equations, nuissance)
    c_required, x_required = matrix_form(expanded_equations, required)
    c_permissible, x_permissible = matrix_form(expanded_equations, permissible)

    # Construct the complete coefficient matrix, making sure to cast to float (very important)
    c_complete = np.hstack((c_nuissance, c_required, c_permissible)).astype(float)
    x_complete = np.hstack((x_nuissance, x_required, x_permissible))

    print 'c_nuissance:'
    print spy(c_nuissance)
    print 'c_required:'
    print spy(c_required)
    print 'c_permissible:'
    print spy(c_permissible)

    print 'Full system:'
    print spy(c_complete)

    if solutions is not None:
        for s in solutions:
            print '  Evaluated at %s: %s' % (s, np.dot(c_complete, evaluate_poly_vector(x_complete, s, float)))

    # Eliminate the nuissance monomials
    u_complete, nuissance_rows_used = echelon.partial_row_echelon_form(c_complete, ncols=nn)
    c_elim = u_complete[nuissance_rows_used:, nn:]
    x_elim = x_complete[nn:]

    print 'Used %d rows to eliminate %d nuissance monomials' % (nuissance_rows_used, nn)

    print 'After first LU:'
    print spy(u_complete)

    print 'After dropping nuissance monomials:'
    print spy(c_elim)

    if solutions is not None:
        for s in solutions:
            print '  Evaluated at %s: %s' % (s, np.dot(c_elim, evaluate_poly_vector(x_elim, s, float)))

    # Put the required monomial columns on row echelon form
    u_elim, required_rows_used = echelon.partial_row_echelon_form(c_elim, ncols=nr, tol=1e-15)

    print 'Used %d rows to put %d required monomials on row echelon form' % (required_rows_used, nr)

    print 'After second LU:'
    print spy(u_elim)

    if solutions is not None:
        for s in solutions:
            print '  Evaluated at %s: %s' % (s, np.dot(u_elim, evaluate_poly_vector(x_elim, s, float)))

    # First block division
    u_r = u_elim[:nr, :nr]
    c_p1 = u_elim[:nr, nr:]
    c_p2 = u_elim[nr:, nr:]

    # Check u_r
    # note that it is fine for nuissance monomials to have zero on the diagonal
    # because we simply drop those monomials, but since we will invert the
    # required monomial submatrix, all those diagonal entries must be non-zero
    defficient_indices = np.flatnonzero(np.abs(np.diag(u_r)) < 1e-8)
    if len(defficient_indices) > 0:
        print 'Failed to eliminate the following required monomials:'
        for i in defficient_indices:
            print '  ', x_required[i]

    # Factorize c_p2
    q, r, ordering = scipy.linalg.qr(c_p2, pivoting=True)
    p = permutation_matrix(ordering)

    x_reordered = np.dot(p.T, x_permissible)
    c_p1_p = np.dot(c_p1, p)
    assert c_p1_p.shape == (nr, npe)

    print 'After QR:'
    print spy(r)

    if solutions is not None:
        for s in solutions:
            print '  Evaluated at %s: %s' % (s, np.dot(r, evaluate_poly_vector(x_reordered, s, float)))

    success = False
    max_eliminations = min(len(expanded_equations) - nuissance_rows_used - required_rows_used,
                           len(present) - nn - nr)

    if max_eliminations <= 0:
        print 'Too few rows present - something went wrong'

    for ne in range(0, max_eliminations):
        # Compute the basis
        eliminated = [poly.leading_term(LexOrdering()).monomial for poly in x_reordered[:ne]]
        basis = [poly.leading_term(LexOrdering()).monomial for poly in x_reordered[ne:]]

        # Check whether this basis is complete
        rank = np.linalg.matrix_rank(basis)
        if rank < nvars:
            print 'Basis is incomplete at ne=%d (rank=%d, basis=%s)' % (ne, rank, x_reordered[ne:])
            continue

        # Form c1, c2
        c_pp1 = c_p1_p[:nr, :ne]
        u_pp2 = r[:ne, :ne]
        c1 = np.vstack((np.hstack((u_r, c_pp1)),
                        np.hstack((np.zeros((ne, nr)), u_pp2))))

        c_b1 = c_p1_p[:nr, ne:]
        c_b2 = r[:ne, ne:]
        c2 = np.vstack((c_b1, c_b2))

        # Check rank of c1
        assert c1.shape[0] == c1.shape[1]
        condition = np.linalg.cond(c1)
        if condition < 1e+8:
            success = True
            print 'Success at ne=%d (condition=%f)' % (ne, condition)
            break
        else:
            print 'Conditioning is poor at ne=%d (condition=%f, basis=%s)' % (ne, condition, x_reordered[ne:])

    if not success:
        raise PolynomialSolverError('Could not find a valid basis')

    # Report
    print 'Num monomials: ', len(present)
    print 'Num nuissance: ', len(nuissance)
    print 'Num required: ', len(required)
    print 'Num eliminated by qr: ', ne
    print 'Basis size: ', len(basis)

    # Compute action matrix form for p*B
    p_basis = [lambda_poly*m for m in basis]
    action_b, _ = matrix_form(p_basis, basis)
    action_r, _ = matrix_form(p_basis, required + eliminated)

    soln = np.linalg.solve(c1, c2)
    action = action_b - np.dot(action_r, soln)

    print 'Basis:'
    print map(Term.from_monomial, basis)

    print 'Basis * p'
    for bi, row in zip(basis, action):
        lhs = bi*lambda_poly
        rhs = sum(as_polynomial(bj, nvars) * aj for bj, aj in zip(basis, row))
        print '  %s * (%s) = %s = %s' % (as_term(bi, nvars), lambda_poly, lhs, rhs)
        if solutions is not None:
            for s in solutions:
                print '    at %s, lhs=%s, rhs=%s' % (s, lhs(*s), rhs(*s))

    print 'Action matrix:'
    print action

    # Find indices within basis
    unit_index = basis.index(Polynomial.constant(1, nvars))
    #var_indices = [basis.index(var) for var in vars]

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(action)

    print 'Eigenvectors:'
    print eigvecs

    # Divide out the unit monomial row
    nrm = eigvecs[unit_index]
    mask = np.abs(nrm) > 1e-8
    monomial_values = (eigvecs[:, mask] / eigvecs[unit_index][mask]).T

    print 'Normalized eigenvectors:'
    print monomial_values

    # Test each solution
    solutions = []
    for values in monomial_values:
        #candidate = [eigvec[i]/eigvec[unit_index] for i in var_indices]
        for solution in solve_monomial_equations(basis, values):
            print 'Candidate solution:', solution
            values = [f(*solution) for f in equations]
            print '  System values:', values
            if np.linalg.norm(values) < 1e-8:
                solutions.append(solution)

    # Report final solutions
    base_vars = Polynomial.coordinates(lambda_poly.num_vars)
    print 'Solutions:'
    for solution in solutions:
        print '  ' + ' '.join('%s=%-10.4f' % (var, val) for var, val in zip(base_vars, solution))

    return SolutionSet(solutions)