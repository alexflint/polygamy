import numpy as np
import scipy.linalg

from polynomial import *

def main():
    np.set_printoptions(linewidth=300, suppress=True)
    run_two_vars()
    #run_three_vars()
    #run_three_spheres()


def run_two_vars():
    x, y = Polynomial.coordinates(2)
    equations = [x**2 + y**2 - 1,
                 x-y]
    expansion_monomials = [[x, y],
                           [x, y, x*y, x*x, y*y]]
    #solve_truncated(equations, expansion_monomials)
    solve_basis_selection(equations, expansion_monomials)


def run_two_vars_v2():
    x, y = Polynomial.coordinates(2)
    equations = [x**2 + y**2 - 1,
                 x-y]
    expansion_monomials = [[x, y, y],
                           [x, y, x**2, x*y, y**2]]
    #solve_truncated(equations, expansion_monomials)
    solve_basis_selection(equations, expansion_monomials)


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
        [x, y, z],
    ]
    solve_truncated(equations, expansion_monomials)


def run_three_spheres():
    x, y, z = Polynomial.coordinates(3)
    equations = [
        x**2 + y**2 + z**2 - 2,
        (x-1)**2 + (y-1)**2 + z**2 - 2,
        (x-1)**2 + (y-1)**2 + (z-1)**2 - 2,
    ]
    expansion_monomials = [
        [x, y, z, x*x, y*y, z*z, x*y, y*z, x*z],
        [x, y, z, x*x, y*y, z*z, x*y, y*z, x*z],
        [x, y, z, x*x, y*y, z*z, x*y, y*z, x*z],
    ]
    solve_truncated(equations, expansion_monomials)


def partial_lu(a, ncols):
    """Compute an LU decomposition whether only the first N columns of U are upper
    triangular."""
    p, l, u = scipy.linalg.lu(a)
    ll = l.copy()
    ll[:, ncols:] = np.eye(*l.shape)[:, ncols:]
    uu = scipy.linalg.solve_triangular(ll, np.dot(p.T, a), lower=True)
    return p, ll, uu


def permutation_matrix(p):
    a = np.zeros((len(p), len(p)), int)
    for i, pi in enumerate(p):
        a[pi,i] = 1
    return a


def solve_monomials(monomials, values):
    # Given a set of monomials and their values at some point (x1,..,xn), compute
    # the possible values forx1,..,xn at that point (there will be up to 2^n
    # possibilities)
    assert all(isinstance(m, tuple) for m in monomials)
    a = np.asarray(monomials, float)
    if any(np.abs(v) < 1e-8 for v in values):
        print 'Warning: some values were zero, cannot solve for these'
        return
    log_x, residuals, rank, svs = np.linalg.lstsq(a, np.log(np.abs(values)))
    if np.linalg.norm(residuals) < 1e-6:
        abs_x = np.exp(log_x)
        for signs in itertools.product((-1, 1), repeat=len(abs_x)):
            x = abs_x * signs
            err = sum((evaluate_monomial(m, x) - y)**2 for m, y in zip(monomials, values))
            if err < 1e-6:
                yield x


def solve_basis_selection(equations, expansion_monomials):
    np.random.seed(0)
    vars = Polynomial.coordinates(equations[0].num_vars)

    # Pick a polynomial to compute the action matrix for
    lambda_p = sum(np.random.rand() * var for var in vars)

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
        p_m = lambda_p * m
        if all(mi in present for mi in p_m.monomials):
            permissible.add(m)

    # Compute required monomials
    required = set()
    for m in permissible:
        p_m = lambda_p * m
        for mi in p_m.monomials:
            if mi not in permissible:
                required.add(mi)

    nuissance = list(present.difference(set.union(permissible, required)))
    required = list(required)
    permissible = list(permissible)

    nn = len(nuissance)
    nr = len(required)
    npe = len(permissible)

    print 'Present monomials:', ', '.join(map(str, map(Term.from_monomial, present)))
    print 'Permissible monomials:', ', '.join(map(str, map(Term.from_monomial, permissible)))
    print 'Required monomials:', ', '.join(map(str, map(Term.from_monomial, required)))
    print 'Nuissance monomials:', ', '.join(map(str, map(Term.from_monomial, nuissance)))

    # Construct the three column blocks from the expanded equations
    c_nuissance, _ = matrix_form(expanded_equations, nuissance)
    c_required, _ = matrix_form(expanded_equations, required)
    c_permissible, x_permissible = matrix_form(expanded_equations, permissible)
    c = np.hstack((c_nuissance, c_required, c_permissible))

    print 'c_nuissance:'
    print c_nuissance
    print 'c_required:'
    print c_required
    print 'c_permissible:'
    print c_permissible
    print 'c:'
    print c

    # Eliminate the nuissance monomials
    _, l, u = partial_lu(c, ncols=nn+nr)

    print 'u:'
    print u

    # First block division
    u_r = u[nn:nn+nr, nn:nn+nr]
    c_p1 = u[nn:nn+nr, nn+nr:]
    c_p2 = u[nn+nr:, nn+nr:]

    # Factorize c_p2
    print 'c_p2:', c_p2.shape
    q, r, perm = scipy.linalg.qr(c_p2, pivoting=True)
    p = permutation_matrix(perm)

    c_p1_p = np.dot(c_p1, p)
    assert c_p1_p.shape == (nr, npe)

    success = False
    max_eliminations = min(len(expanded_equations), len(present)) - nr - nn
    for ne in range(1, max_eliminations):
        # Form c1, c2
        c_pp1 = c_p1_p[:nr, :ne]
        u_pp2 = r[:ne, :ne]
        c1 = np.vstack((np.hstack((u_r, c_pp1)),
                        np.hstack((np.zeros((ne, nr)), u_pp2))))

        c_b1 = c_p1_p[:nr, ne:]
        c_b2 = r[:ne, ne:]
        c2 = np.vstack((c_b1, c_b2))

        # Compute the basis
        x_reordered = np.dot(p.T, x_permissible)
        eliminated = [poly.leading_term(LexOrdering()).monomial for poly in x_reordered[:ne]]
        basis = [poly.leading_term(LexOrdering()).monomial for poly in x_reordered[ne:]]

        # Check rank of c1
        assert c1.shape[0] == c1.shape[1]
        condition = np.linalg.cond(c1)
        if condition < 1e+5:
            success = True
            print 'Success at ne=%d (condition=%f)' % (ne, condition)
            break
        else:
            print 'Conditioning is too poor at ne=%d (condition=%f)' % (ne, condition)

    if not success:
        return

    # Report
    print 'Num monomials: ', len(present)
    print 'Num nuissance: ', len(nuissance)
    print 'Num required: ', len(required)
    print 'Num eliminated by qr: ', ne
    print 'Basis size: ', len(basis)

    # Compute p * x_basis
    p_basis = [lambda_p*m for m in basis]
    print 'p_basis:'
    print p_basis
    print 'num_vars:', p_basis[0].num_vars
    print 'basis:', map(Term.from_monomial, basis)

    # Compute action matrix form for p*B
    action_b, _ = matrix_form(p_basis, basis)
    action_r, _ = matrix_form(p_basis, required + eliminated)
    print 'c1:', c1.shape
    print 'c2:', c2.shape
    print 'action_b:', action_b.shape
    print 'action_r:', action_r.shape
    action = action_b - np.dot(action_r, np.linalg.solve(c1, c2))

    print 'action matrix:'
    print action

    # Find indices within basis
    unit_index = basis.index(Polynomial.constant(1, len(vars)))
    #var_indices = [basis.index(var) for var in vars]

    # Compute eigenvalues and eigenvectors
    eigvals, eigvecs = np.linalg.eig(action)

    # Divide out the unit monomial row
    nrm = eigvecs[unit_index]
    mask = np.abs(nrm) > 1e-8
    print mask
    print eigvecs
    monomial_values = (eigvecs[:, mask] / eigvecs[unit_index][mask]).T

    print 'Basis:'
    print map(Term.from_monomial, basis)

    print 'Normalized eigenvectors:'
    print monomial_values

    # Test each solution
    solutions = []
    for values in monomial_values:
        #candidate = [eigvec[i]/eigvec[unit_index] for i in var_indices]
        for solution in solve_monomials(basis, values):
            print 'Candidate solution:', solution
            values = [f(*solution) for f in equations]
            print '  System values:', values
            if np.linalg.norm(values) < 1e-8:
                solutions.append(solution)

    # Report final solutions
    print 'Solutions:'
    for solution in solutions:
        print '  ' + ' '.join('%s=%10.6f' % (var, val) for var, val in zip(vars, solution))



def solve_truncated(equations, expansion_monomials):
    np.random.seed(0)
    vars = Polynomial.coordinates(equations[0].num_vars)

    # Pick a polynomial to compute the action matrix for
    p = sum(np.random.rand() * var for var in vars)

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
        p_m = p * m
        if all(mi in present for mi in p_m.monomials):
            permissible.add(m)

    basis = permissible

    # Compute required monomials
    p_basis = []
    required = set()
    for m in basis:
        p_m = p * m
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
    p, l, u = scipy.linalg.lu(c)

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
