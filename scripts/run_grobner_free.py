import numpy as np
import scipy.linalg

from polynomial import *

def main():
    vars = parse('x', 'y')
    x, y = vars

    equations = [x**2 + y**2 - 1, x-y]
    expansion_monomials = [[], [x, y]]

    p = x + 2*y
    num_vars = len(vars)

    print 'F:'
    for f in equations:
        print '  ', f

    expanded_equations = list(equations)
    for f, expansions in zip(equations, expansion_monomials):
        for monomial in expansions:
            expanded_equations.append(f * monomial)

    print 'Expanded equations:'
    for f in expanded_equations:
        print '  ', f

    present = set(term.monomial for f in expanded_equations for term in f)

    permissible = set()
    for m in present:
        p_m = p * m
        if all(mi in present for mi in p_m.monomials):
            permissible.add(m)

    basis = permissible

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
    c1 = u[nn:nn+nb, nn:nn+nb]
    c2 = u[nn:nn+nb, nn+nb:]

    print 'c1:'
    print c1
    print np.linalg.inv(c1)

    # Check rank of c1
    rank = np.linalg.matrix_rank(c1)
    if rank < nb:
        print 'Error: c1 is only of rank %d' % rank

    # Compute action matrix form for p*B
    print 'p_basis:'
    print p_basis
    print 'basis:'
    print basis
    action_b, _ = matrix_form(p_basis, basis)
    action_r, _ = matrix_form(p_basis, required)
    action = action_b - np.dot(action_r, np.linalg.solve(c1, c2))

    print 'action matrix:'
    print action

    # Find indices within basis
    unit_index = basis.index(Polynomial.constant(1, num_vars))
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

    # Report final solutions
    print 'Solutions:'
    for solution in solutions:
        print '  ' + ' '.join('%s=%10.6f' % (var, val) for var, val in zip(vars, solution))


if __name__ == '__main__':
    main()
