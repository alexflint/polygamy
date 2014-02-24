import itertools
import numpy as np

from polysolve import *

# the root is at (x=1 y=2 z=3) and (x=10 y=20 z=30)

f1,f2,f3 = parse('(x+y+z-6) * (3*x + 2*y + z - 100)',
                 '(y+z-5) * (y**2+z-430)**2',
                 '(z-3) * (z-30)**2')

def solve_triangular(F, threshold=1e-3):
    fun = polynomial_vector(F)
    n = F[0].num_vars
    candidate_roots = []
    for i in reversed(range(n)):
        roots = []
        for root in itertools.product(*candidate_roots):
            fcur = F[i]
            for var_index,value in enumerate(root):
                fcur = fcur.evaluate_partial(n-var_index-1, value)
            cur_roots,cur_brackets = solve_univariate(fcur.squeeze())
            roots.extend(cur_roots)
        candidate_roots.append(roots)

    candidate_roots.reverse()

    polished_roots = []
    for root in itertools.product(*candidate_roots):
        r = fun(*root)
        if np.linalg.norm(r) < threshold:
            result = polish_multivariate_root(F, root)
            print result
            print '\n'
            polished_roots.append(result.x)

    return np.array(polished_roots)

print solve_triangular([f1,f2,f3], 1e-6)
