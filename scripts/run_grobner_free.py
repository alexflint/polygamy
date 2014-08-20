import numpy as np
import scipy.linalg

from polynomial import *

def main():
    np.random.seed(0)
    np.set_printoptions(linewidth=300, suppress=True)
    #run_two_vars()
    run_two_circles()
    #run_three_vars()
    #run_three_spheres()


def run_two_vars():
    x, y = Polynomial.coordinates(2)
    equations = [x**2 + y**2 - 1,
                 x-y]
    expansion_monomials = [
        [x, y],
        [x, y, x*y, x*x, y*y]
    ]
    #solve_truncated(equations, expansion_monomials)
    solve_via_basis_selection(equations, expansion_monomials, x+y)


def run_two_circles():
    x, y = Polynomial.coordinates(2)
    equations = [
        (x-1)**2 + (y+1)**2 - 9,
        (x-1)**2 + (y+3)**2 - 9,
        ]
    expansion_monomials = [
        [x, y, x*y, x*x, y*y, x*x*y],
        [x, y, x*y, x*x, y*y, x*x*y]
    ]
    #solve_truncated(equations, expansion_monomials)
    solve_via_basis_selection(equations, expansion_monomials, x+y)


def run_three_vars():
    x, y, z = Polynomial.coordinates(3)
    equations = [
        x**2 + y**2 + z**2 - 1,
        x - y,
        x - z
    ]
    expansion_monomials = [
        [],
        [x, y, z],
        [x, y, z]
    ]

    #solve_truncated(equations, expansion_monomials)
    solve_via_basis_selection(equations, expansion_monomials, x)


def all_monomials(variables, degree):
    return map(product, itertools.product(list(variables)+[1], repeat=degree))


def run_three_spheres():
    x, y, z = Polynomial.coordinates(3)
    equations = [
        x**2 + y**2 + z**2 - 9,
        (x-1)**2 + (y-1)**2 + z**2 - 9,
        (x-1)**2 + (y-1)**2 + (z-1)**2 - 9,
    ]
    expansion_monomials = [
        #[x, y, z, x*x, y*y, z*z, x*y, y*z, x*z],
        #[x, y, z, x*x, y*y, z*z, x*y, y*z, x*z],
        #[x, y, z, x*x, y*y, z*z, x*y, y*z, x*z],
        #list(all_monomials((x, y, z), 2)),
        #list(all_monomials((x, y, z), 2)),
        #list(all_monomials((x, y, z), 2)),
        #[x, x*y, x*z],
        #[x, x*y, x*z],
        #[x, x*y, x*z],
        [],
        [],
        [x, y, z],
    ]

    # Pick a polynomial to compute the action matrix for
    lambda_poly = x  # sum(np.random.rand() * var for var in vars)

    #solve_truncated(equations, expansion_monomials, lambda_poly)
    solve_via_basis_selection(equations, expansion_monomials, lambda_poly)


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


def swap_rows(a, i, j):
    # Note that a[i],a[j] = a[j],a[i] will _not_ work here
    temp = a[i].copy()
    a[i] = a[j]
    a[j] = temp


class RowEchelonError(Exception):
    def __init__(self, col):
        self.col = col
    def __str__(self):
        return 'elimination failed at column '+str(self.col)


def partial_lu(a, ncols):
    """Compute an LU decomposition whether only the first N columns of U are upper
    triangular."""
    assert a.shape[0] <= a.shape[1], \
        'partial_lu not implemented for matrices with nr > nc'
    p, l, u = scipy.linalg.lu(a)
    ll = l.copy()
    ll[:, ncols:] = np.eye(*l.shape)[:, ncols:]
    uu = scipy.linalg.solve_triangular(ll, np.dot(p.T, a), lower=True)
    return p, ll, uu


def partial_row_echelon_form(a, ncols, tol=1e-8):
    """Eliminate the first N columns of A, including pivoting."""
    a = np.asarray(a)
    assert ncols <= a.shape[1]
    if a.dtype.kind == 'i':
        a = a.astype(float)

    row = 0
    u = a.copy()
    for col in range(ncols):
        # move the row with the largest element in col i to the top
        pivot_row = row + np.argmax(np.abs(u[row:, col]))
        swap_rows(u, row, pivot_row)
        if abs(u[row, col]) < tol:
            # this column is already eliminated, which is fine
            u[col, col] = 0.
        else:
            u[row, col+1:] /= u[row, col]
            u[row, col] = 1.
            u[row+1:, col+1:] -= u[row+1:, col:col+1] * u[row, col+1:]
            u[row+1:, col] = 0.
            row += 1
    return u, row


def permutation_matrix(p):
    a = np.zeros((len(p), len(p)), int)
    for i, pi in enumerate(p):
        a[pi,i] = 1
    return a


def solve_monomial_equations(monomials, values):
    """Given a set of monomials and their values at some point (x1,..,xn), compute
    the possible values for x1,..,xn at that point (there will be up to 2^n
    possibilities)."""
    assert all(isinstance(m, tuple) for m in monomials)
    a = np.asarray(monomials, float)

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


def solve_via_basis_selection(equations, expansion_monomials, lambda_poly):
    vars = Polynomial.coordinates(lambda_poly.num_vars)

    print 'Equations:'
    for f in equations:
        print '  ', f

    # Expand equations
    expanded_equations = list(equations)
    for f, expansions in zip(equations, expansion_monomials):
        for monomial in expansions:
            expanded_equations.append(f * monomial)

    print 'Grobner basis:'
    gb_equations = [p.astype(fractions.Fraction) for p in equations]
    for f in gbasis(gb_equations, GrevlexOrdering()):
        print '  ', f
        expanded_equations.append(f)

    print 'Expanded equations:'
    for f in expanded_equations:
        print '  ', f

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

    if len(permissible) <= len(vars):
        print 'Error: There are fewer permissible monomials than variables. Add more expansions.'
        return

    if num_remaining <= 0:
        print 'Error: The number of required plus nuissance monomials exceeds the number of equations. Add more expansions.'
        return

    # Construct the three column blocks from the expanded equations
    c_nuissance, x_nuissance = matrix_form(expanded_equations, nuissance)
    c_required, x_required = matrix_form(expanded_equations, required)
    c_permissible, x_permissible = matrix_form(expanded_equations, permissible)
    c_complete = np.hstack((c_nuissance, c_required, c_permissible))

    print 'c_nuissance:'
    print spy(c_nuissance)
    print 'c_required:'
    print spy(c_required)
    print 'c_permissible:'
    print spy(c_permissible)

    print 'Full system:'
    print spy(c_complete)

    # Eliminate the nuissance monomials
    u_complete, rows_used = partial_row_echelon_form(c_complete, ncols=nn)
    c_elim = u_complete[rows_used:, nn:]

    print 'Used %d rows to eliminate %d nuissance monomials' % (rows_used, nn)

    print 'After first LU:'
    print spy(u_complete)

    print 'After dropping nuissance monomials:'
    print spy(c_elim)

    # Put the required monomial columns on row echelon form
    u_elim, rows_used = partial_row_echelon_form(c_elim, ncols=nr, tol=1e-15)

    print 'Used %d rows to put %d required monomials on row echelon form' % (rows_used, nr)

    print 'After second LU:'
    print spy(u_elim)

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
        #return

    # Factorize c_p2
    q, r, perm = scipy.linalg.qr(c_p2, pivoting=True)
    p = permutation_matrix(perm)

    x_reordered = np.dot(p.T, x_permissible)
    c_p1_p = np.dot(c_p1, p)
    assert c_p1_p.shape == (nr, npe)

    print 'After QR:'
    print spy(r)

    success = False
    max_eliminations = min(len(expanded_equations) - rows_used - nr, len(present) - nn - nr)
    for ne in range(0, max_eliminations):
        # Compute the basis
        eliminated = [poly.leading_term(LexOrdering()).monomial for poly in x_reordered[:ne]]
        basis = [poly.leading_term(LexOrdering()).monomial for poly in x_reordered[ne:]]

        # Check whether this basis is complete
        rank = np.linalg.matrix_rank(basis)
        if rank < len(vars):
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
        print 'Could not find a valid basis'
        return

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

    print 'Action matrix:'
    print action

    # Find indices within basis
    unit_index = basis.index(Polynomial.constant(1, len(vars)))
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
    print 'Solutions:'
    for solution in solutions:
        print '  ' + ' '.join('%s=%10.6f' % (var, val) for var, val in zip(vars, solution))


def solve_via_truncation(equations, expansion_monomials, lambda_poly):
    vars = Polynomial.coordinates(equations[0].num_vars)

    print 'Equations:'
    for f in equations:
        print '  ', f

    # Expand equations
    expanded_equations = list(equations)
    for f, expansions in zip(equations, expansion_monomials):
        for monomial in expansions:
            expanded_equations.append(f * monomial)

    print 'Expanded equations:'
    for f in expanded_equations:
        print '  ', f

    present = set(term.monomial for f in expanded_equations for term in f)

    # Compute permissible monomials
    permissible = set()
    for m in present:
        p_m = lambda_poly * m
        if all(mi in present for mi in p_m.monomials):
            permissible.add(m)

    basis = permissible

    # Compute required monomials
    p_basis = []
    required = set()
    for m in basis:
        p_m = lambda_poly * m
        p_basis.append(p_m)
        for mi in p_m.monomials:
            if mi not in basis:
                required.add(mi)

    nuissance = list(present.difference(set.union(basis, required)))
    required = list(required)
    basis = list(basis)
    permissible = list(permissible)

    print 'Present monomials:', ', '.join(map(str, map(Term.from_monomial, present)))
    print 'Permissible monomials:', ', '.join(map(str, map(Term.from_monomial, permissible)))
    print 'Required monomials:', ', '.join(map(str, map(Term.from_monomial, required)))
    print 'Nuissance monomials:', ', '.join(map(str, map(Term.from_monomial, nuissance)))

    # Construct the three column blocks from the expanded equations
    c_nuissance, _ = matrix_form(expanded_equations, nuissance)
    c_required, _ = matrix_form(expanded_equations, required)
    c_basis, _ = matrix_form(expanded_equations, basis)
    c = np.hstack((c_nuissance, c_required, c_basis))

    print 'c_nuissance:'
    print c_nuissance
    print 'c_required:'
    print c_required
    print 'c_basis:'
    print c_basis
    print 'c:'
    print c

    # Eliminate the nuissance monomials
    lambda_poly, l, u = scipy.linalg.lu(c)

    print 'u:'
    print u

    nn = len(nuissance)
    nb = len(basis)
    nr = len(required)
    c1 = u[nn:nn+nr, nn:nn+nr]
    c2 = u[nn:nn+nr, nn+nr:]

    print 'c1:'
    print c1

    # Check rank of c1
    rank = np.linalg.matrix_rank(c1)
    if rank < c1.shape[0]:
        print 'Error: c1 is only of rank %d (needed rank %d)' % (rank, c1.shape[0])
        return

    # Compute action matrix form for p*B
    print 'p_basis:'
    print p_basis
    action_b, _ = matrix_form(p_basis, basis)
    action_r, _ = matrix_form(p_basis, required)
    action = action_b - np.dot(action_r, np.linalg.solve(c1, c2))

    print 'action matrix:'
    print action

    # Find indices within basis
    unit_index = basis.index(Polynomial.constant(1, len(vars)))
    var_indices = [basis.index(var) for var in vars]

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(action)

    # Test each solution
    solutions = []
    for eigvec in eigvecs.T:
        candidate = [eigvec[i]/eigvec[unit_index] for i in var_indices]
        print 'Candidate solution:', candidate
        values = [f(*candidate) for f in equations]
        print '  System values:', values
        if np.linalg.norm(values) < 1e-8:
            solutions.append(candidate)

    print 'Basis size: ', len(basis)
    print 'Num required: ', len(required)
    print 'Num nuissance: ', len(nuissance)

    # Report final solutions
    print 'Solutions:'
    for solution in solutions:
        print '  ' + ' '.join('%s=%10.6f' % (var, val) for var, val in zip(vars, solution))


if __name__ == '__main__':
    main()
