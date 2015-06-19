import itertools
import fractions
import numpy as np
import scipy.linalg

import utils
import echelon
from polynomial import Polynomial, Term, evaluate_monomial, gbasis, matrix_form, GrevlexOrdering, LexOrdering,\
    as_term, as_polynomial, as_monomial, product


class SolutionSet(object):
    def __init__(self, solutions):
        self.solutions = solutions


class PolynomialSolverError(Exception):
    """Represents a failure of a solver to identify solutions to a polynomial system."""
    pass


class DiagnosticError(Exception):
    """Represents a failure detected using known solutions to a system of polynomial equations."""
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


def all_monomials(variables, degree):
    return list(map(product, itertools.product(list(variables)+[1], repeat=degree)))


def evaluate_poly_vector(v, x, dtype=float):
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
        yield np.take(values, naked_indices)
        return

    if np.any(np.abs(values) < 1e-8):
        print('Warning: some values were zero, cannot solve for these')
        return

    log_x, residuals, rank, svs = np.linalg.lstsq(a, np.log(np.abs(values)))
    if rank < a.shape[1]:
        print('Warning: rank defficient monomial equations (incomplete basis?)')
        return

    if np.linalg.norm(residuals) < 1e-6:
        abs_x = np.exp(log_x)
        for signs in itertools.product((-1, 1), repeat=len(abs_x)):
            x = abs_x * signs
            err = sum((evaluate_monomial(m, x) - y)**2 for m, y in zip(monomials, values))
            if err < 1e-6:
                yield x


def solve_via_basis_selection(equations, expansion_monomials, lambda_poly, diagnostic_solutions=None,
                              include_grobner=False, verbosity=1):
    ROW_ECHELON_TOLERANCE = 1e-10
    nvars = lambda_poly.num_vars
    lambda_poly = lambda_poly.astype(float)

    if verbosity >= 1:
        print('Equations:')
        for f in equations:
            print('  ', f)

    if diagnostic_solutions is not None:
        for f in equations:
            for solution in diagnostic_solutions:
                val = f(*solution)
                assert abs(val) < 1e-8, 'expected input equations to evaluate to zero at diagnostic solution'

    # Expand equations
    expanded_equations = list(equations)
    if utils.list_depth(expansion_monomials) == 3:
        assert len(equations) == len(expansion_monomials)
        for f, expansions in zip(equations, expansion_monomials):
            for monomial in expansions:
                expanded_equations.append(f * monomial)
    else:
        for f in equations:
            for monomial in expansion_monomials:
                expanded_equations.append(f * monomial)

    # Add grobner basis if requested
    if include_grobner:
        print('Grobner basis:')
        gb_equations = [p.astype(fractions.Fraction) for p in equations]
        grobner_basis = gbasis(gb_equations, GrevlexOrdering())
        expanded_equations.extend(grobner_basis)
        if verbosity >= 1:
            print('Grobner basis size: %d' % len(gb_equations))
            if verbosity >= 2:
                for f in grobner_basis:
                    print('  ', f)

    # Report expanded equations if requested
    if verbosity >= 1:
        print('Num expanded equations: %d' % len(expanded_equations))
        if verbosity >= 3:
            for f in expanded_equations:
                print('  ', f)

    # Test expanded equations against known solutions
    if diagnostic_solutions is not None:
        for f in expanded_equations:
            for solution in diagnostic_solutions:
                val = f(*solution)
                assert abs(val) < 1e-8, 'expected input equations to evaluate to zero at diagnostic solution'

    # Compute present monomials
    present = set(term.monomial for f in expanded_equations for term in f)
    original = set(term.monomial for f in equations for term in f)

    # Compute permissible monomials
    permissible = set()
    for monomial in original:
        p_m = lambda_poly * monomial
        if all(mi in present for mi in p_m.monomials):
            permissible.add(monomial)

    # Compute required monomials
    required = set()
    for monomial in permissible:
        p_m = lambda_poly * monomial
        for mi in p_m.monomials:
            if mi not in permissible:
                required.add(mi)

    nuissance = list(present.difference(set.union(permissible, required)))
    required = list(required)
    permissible = list(permissible)

    nn = len(nuissance)
    nr = len(required)

    # Report monomials in each set
    print('Present monomials (%d):' % len(present),\
        ', '.join(map(str, list(map(Term.from_monomial, present)))))
    print('Permissible monomials (%d):' % len(permissible),\
        ', '.join(map(str, list(map(Term.from_monomial, permissible)))))
    print('Required monomials (%d):' % len(required),\
        ', '.join(map(str, list(map(Term.from_monomial, required)))))
    print('Nuissance monomials (%d):' % len(nuissance),\
        ', '.join(map(str, list(map(Term.from_monomial, nuissance)))))

    if len(permissible) <= nvars:
        raise PolynomialSolverError('There are fewer permissible monomials than variables. Add more expansions.')

    # Do not try to check whether we have enough equations here because we do not yet know how many rows it will take
    # to eliminate the nuissance monomials (typically it takes less than the number of nuissance monomials).

    # Construct the three column blocks from the expanded equations
    c_nuissance, x_nuissance = matrix_form(expanded_equations, nuissance)
    c_required, x_required = matrix_form(expanded_equations, required)
    c_permissible, x_permissible = matrix_form(expanded_equations, permissible)

    # Construct the complete coefficient matrix, making sure to cast to float (very important)
    c_complete = np.hstack((c_nuissance, c_required, c_permissible)).astype(float)
    x_complete = x_nuissance + x_required + x_permissible

    if verbosity >= 3:
        print('c_nuissance:')
        print(spy(c_nuissance))
        print('c_required:')
        print(spy(c_required))
        print('c_permissible:')
        print(spy(c_permissible))

    if verbosity >= 2:
        print('Full system:')
        print(spy(c_complete))

    if diagnostic_solutions is not None:
        for solution in diagnostic_solutions:
            values = np.dot(c_complete, evaluate_poly_vector(x_complete, solution))
            if verbosity >= 2:
                print('  Evaluated at %s: %s' % (solution, values))
            if np.abs(values).max() > 1e-8:
                idx = np.abs(values).argmax()
                raise DiagnosticError('expected equation %d to evaluate to zero but received %f' % (idx, values[idx]))

    # Eliminate the nuissance monomials
    u_complete, nuissance_rows_used = echelon.partial_row_echelon_form(c_complete, ncols=nn)
    c_elim = u_complete[nuissance_rows_used:, nn:]
    x_elim = x_complete[nn:]

    if verbosity >= 1:
        print('Used %d rows to eliminate %d nuissance monomials, have %d left' % (nuissance_rows_used, nn, len(c_elim)))

    if verbosity >= 2:
        print('After putting nuissnace monomials on row echelon form:')
        print(spy(u_complete))
        print('After dropping nuissance monomials:')
        print(spy(c_elim))

    if diagnostic_solutions is not None:
        for solution in diagnostic_solutions:
            values = np.dot(c_elim, evaluate_poly_vector(x_elim, solution))
            if verbosity >= 2:
                print('  Evaluated at %s: %s' % (solution, values))
            if np.abs(values).max() > 1e-8:
                idx = np.abs(values).argmax()
                raise DiagnosticError('expected equation %d to evaluate to zero but received %f' % (idx, values[idx]))

    # Now we can check that we have enough equations left because the number of rows used to put the required monomials
    # on row ehcelon form must always equal the number of required monomials.
    if len(c_elim) < nr:
        raise PolynomialSolverError('there are %d equations remaining after eliminating nuissance monomails, '
                                    'but we need %d to eliminate the required monomials.' % (len(c_elim), nr))

    # Check that c_elim[:, :nr] has full column rank
    c_elim_rank = np.linalg.matrix_rank(c_elim[:, :nr])
    if c_elim_rank < nr:
        print('c_elim has %d rows and rank %d but nr=%d, predicting LU will fail.' % (c_elim.shape[0], c_elim_rank, nr))

    # Put the required monomial columns on row echelon form
    try:
        u_elim, required_rows_used = echelon.partial_row_echelon_form(c_elim,
                                                                      ncols=nr,
                                                                      tol=ROW_ECHELON_TOLERANCE,
                                                                      allow_rank_defficient=False)
    except echelon.RowEchelonError as ex:
        idx_complete = ex.col + nn
        complete_count = int(np.sum(c_complete[:, idx_complete] != 0))
        elim_count = int(np.sum(c_elim[:, ex.col] != 0))
        raise PolynomialSolverError(
            'failed to eliminate required monomial %s (col %d), '
            'which appeared in %d of %d expanded equations, and %d of %d after eliminating nuissance monomials' %
            (x_elim[ex.col], ex.col, complete_count, len(c_complete), elim_count, len(c_elim)))

    if verbosity >= 1:
        print('Used %d rows to put %d required monomials on row echelon form' % (required_rows_used, nr))
        if verbosity >= 2:
            print(spy(u_elim))

    if diagnostic_solutions is not None:
        for solution in diagnostic_solutions:
            values = np.dot(u_elim, evaluate_poly_vector(x_elim, solution))
            if verbosity >= 2:
                print('  Evaluated at %s: %s' % (solution, values))
            if np.abs(values).max() > 1e-8:
                idx = np.abs(values).argmax()
                raise DiagnosticError('expected equation %d to evaluate to zero but received %f' % (idx, values[idx]))

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
        raise PolynomialSolverError('Failed to eliminate required monomials: ' +
                                    ', '.join([str(x_required[i]) for i in defficient_indices]))

    # Factorize c_p2
    q, r, ordering = scipy.linalg.qr(c_p2, pivoting=True)
    p = permutation_matrix(ordering)

    x_reordered = np.dot(p.T, x_permissible)
    c_p1_p = np.dot(c_p1, p)
    assert c_p1_p.shape == (nr, len(permissible))

    if verbosity >= 2:
        print('After QR:')
        print(spy(r))

    if diagnostic_solutions is not None:
        for solution in diagnostic_solutions:
            values = np.dot(r, evaluate_poly_vector(x_reordered, solution))
            if verbosity >= 2:
                print('  Evaluated at %s: %s' % (solution, values))
            if np.abs(values).max() > 1e-8:
                idx = np.abs(values).argmax()
                raise DiagnosticError('expected equation %d to evaluate to zero but received %f' % (idx, values[idx]))

    success = False
    max_eliminations = min(len(expanded_equations) - nuissance_rows_used - required_rows_used,
                           len(present) - nn - nr)

    assert max_eliminations > 0, 'too few rows present - something went wrong'

    for ne in range(0, max_eliminations):
        # Compute the basis
        x_eliminated = x_reordered[:ne]
        x_basis = x_reordered[ne:]
        eliminated = [poly.as_monomial() for poly in x_eliminated]
        basis = [poly.as_monomial() for poly in x_basis]

        # Check whether this basis is complete
        rank = int(np.linalg.matrix_rank(basis))
        if rank < nvars:
            if verbosity >= 1:
                print('Basis is incomplete at ne=%d (rank=%d, basis=%s)' % (ne, rank, x_reordered[ne:]))
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
            if verbosity >= 1:
                print('Success at ne=%d (condition=%f)' % (ne, condition))
            break
        elif verbosity >= 1:
            print('Conditioning is poor at ne=%d (condition=%f, basis=%s)' % (ne, condition, x_reordered[ne:]))

    if not success:
        raise PolynomialSolverError('Could not find a valid basis')

    x_dependent = np.hstack((x_required, x_eliminated))
    dependent = required + eliminated

    # Report
    if verbosity >= 1:
        print('Num monomials: ', len(present))
        print('Num nuissance: ', len(nuissance))
        print('Num required: ', len(required))
        print('Num eliminated by qr: ', ne)
        print('Basis size: ', len(basis))

    # Verify that c1 * basis + c2 * required = 0 at diagnostic solutions
    if diagnostic_solutions is not None:
        c_post = np.hstack((c1, c2))
        x_post = np.hstack((x_required, x_eliminated, x_basis))
        for solution in diagnostic_solutions:
            values = np.dot(c_post, evaluate_poly_vector(x_post, solution))
            if verbosity >= 2:
                print('  Evaluated at %s: %s' % (solution, values))
            if np.abs(values).max() > 1e-8:
                idx = np.abs(values).argmax()
                raise DiagnosticError('expected equation %d to evaluate to zero but received %f' % (idx, values[idx]))

    # Report the basis
    if verbosity >= 1:
        print('Basis:')
        print(list(map(as_monomial, basis)))

    # Compute lambda_poly * basis
    p_basis = [lambda_poly*monomial for monomial in basis]
    c_action_b, _ = matrix_form(p_basis, basis)
    c_action_r, _ = matrix_form(p_basis, dependent)

    # Check that the monomials in p_basis are exactly basis+required+eliminated
    known_monomials = set(basis + dependent)
    for i, p_bi in enumerate(p_basis):
        for monomial in p_bi.monomials:
            if not monomial in known_monomials:
                raise PolynomialSolverError(
                    '%s in lambda_poly*%s (basis element %d) was not in the basis, required, or eliminated set' % \
                        (as_monomial(monomial), as_monomial(basis[i]), i))

    # Check that lambda * b = action_b * b + action_r * r at solution
    for bi, b_row, r_row in zip(basis, c_action_b, c_action_r):
        assert len(r_row) == len(dependent)
        assert len(b_row) == len(basis)
        lhs = bi*lambda_poly
        rhs = (sum(cj*rj for cj, rj in zip(r_row, x_dependent)) +
               sum(cj*bj for cj, bj in zip(b_row, x_basis)))
        if diagnostic_solutions is not None:
            for solution in diagnostic_solutions:
                lvalue = lhs(*solution)
                rvalue = rhs(*solution)
                if verbosity >= 2:
                    print('    at %s, lhs=%s, rhs=%s' % (solution, lvalue, rvalue))
                if abs(lvalue - rvalue) > 1e-8:
                    raise DiagnosticError('expected action equation lhs (=%f) to match rhs (=%f) at %s:\n'
                                          '  lhs=%s\n  rhs=%s' %
                                          (lvalue, rvalue, solution, lhs, rhs))

    # Compute action matrix form for lambda_poly * basis
    soln = np.linalg.solve(c1, c2)
    c_action = c_action_b - np.dot(c_action_r, soln)

    if verbosity >= 2:
        print('Solution for required monomials:')
        print(soln)
        for row, monomial in zip(soln, dependent):
            print('  %s = - %s * basis {at points on variety}' % (as_monomial(monomial), row))

    # Check that basis + soln * required = 0 at diagnostic solutions
    if diagnostic_solutions is not None:
        c_solved = np.hstack((np.eye(len(required) + len(eliminated)), soln))
        for solution in diagnostic_solutions:
            values = np.dot(c_solved, evaluate_poly_vector(x_post, solution))
            if verbosity >= 2:
                print('  Evaluated at %s: %s' % (solution, values))
            if np.abs(values).max() > 1e-8:
                idx = np.abs(values).argmax()
                raise DiagnosticError('expected equation %d to evaluate to zero but received %f' % (idx, values[idx]))

    # Check that lambda * b = action * b at solution (note that lambda is a polynomial, not a matrix here)
    if verbosity >= 2 or diagnostic_solutions is not None:
        if verbosity >= 2:
            print('Action matrix:')
            print(c_action)
        for bi, row in zip(basis, c_action):
            assert len(basis) == len(row)
            lhs = bi*lambda_poly
            rhs = sum(cj * bj for cj, bj in zip(row, x_basis))
            if verbosity >= 2:
                print('  %s * (%s) = %s = %s' % (as_term(bi, nvars), lambda_poly, lhs, rhs))
            if diagnostic_solutions is not None:
                for solution in diagnostic_solutions:
                    lvalue = lhs(*solution)
                    rvalue = rhs(*solution)
                    if verbosity >= 2:
                        print('    at %s, lhs=%s, rhs=%s' % (solution, lvalue, rvalue))
                    if abs(lvalue - rvalue) > 1e-8:
                        raise DiagnosticError('expected action equation lhs (=%f) to match rhs (=%f) at %s:\n'
                                              '  lhs=%s\n  rhs=%s' %
                                              (lvalue, rvalue, solution, lhs, rhs))

    # Find indices within basis
    unit_index = basis.index(Polynomial.constant(1, nvars))

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(c_action)

    if verbosity >= 2:
        print('Eigenvectors:')
        print(eigvecs)

    # Divide out the unit monomial row
    nrm = eigvecs[unit_index]
    mask = np.abs(nrm) > 1e-8
    monomial_values = (eigvecs[:, mask] / eigvecs[unit_index][mask]).T

    if verbosity >= 2:
        print('Normalized eigenvectors:')
        print(monomial_values)

    # Test each solution
    solutions = []
    for values in monomial_values:
        #candidate = [eigvec[i]/eigvec[unit_index] for i in var_indices]
        for solution in solve_monomial_equations(basis, values):
            values = [f(*solution) for f in equations]
            if verbosity >= 1:
                print('Candidate solution: %s -> Values = %s' % (solution, values))
            if np.linalg.norm(values) < 1e-8:
                solutions.append(solution)

    # Report final solutions
    base_vars = Polynomial.coordinates(lambda_poly.num_vars)
    if verbosity >= 1:
        print('Solutions:')
        for solution in solutions:
            print('  ' + ' '.join('%s=%s' % (var, val) for var, val in zip(base_vars, solution)))

    return SolutionSet(solutions)
