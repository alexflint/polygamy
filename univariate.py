import numpy as np

inf = float('inf')


def count_sign_changes(ys):
    signs = [cmp(y, 0) for y in ys if y != 0]  # ignore all zero evaluations
    return sum(signs[i] != signs[i+1] for i in range(len(signs)-1))


class SturmChain(object):
    def __init__(self, polynomial):
        assert polynomial.num_vars == 1
        self._chain = [ polynomial.copy(), polynomial.partial_derivative(0) ]
        while self._chain[-1].total_degree > 0:
            self._chain.append(-(self._chain[-2] % self._chain[-1]))
        self._compiled = [ p.compile() for p in self._chain ]

    def evaluate(self, x):
        if x == inf:
            return count_sign_changes(f.sign_at_infinity() for f in self._chain)
        elif x == -inf:
            return count_sign_changes(f.sign_at_minus_infinity() for f in self._chain)
        else:
            return count_sign_changes(f(x) for f in self._compiled)

    def count_roots_between(self, a, b):
        return self.evaluate(a) - self.evaluate(b)

    def count_roots(self):
        return self.count_roots_between(-inf, inf)


def binary_search_continuous(f, target, low, high):
    '''Perform a floating-point binary search over the interval
    [low,high], evaluating the monotonically increasing function f at
    each point until f(x)=target.'''
    low = float(low)
    high = float(high)

    while True:
        x = (low + high) / 2
        y = f(x)
        if y == target:
            return x
        elif y < target:
            low = x
        else:
            high = x


def isolate_roots(polynomial, lower=-inf, upper=inf):
    '''Given a polynomial with N roots, return N+1 floating
    (K[0],...,K[n]) point numbers such that the i-th root is between
    K[i] and K[i+1].'''
    assert lower < inf
    assert upper > -inf

    # Count the total number of roots we're after
    s = SturmChain(polynomial)
    num_roots = s.count_roots_between(lower, upper)

    # Find a finite lower bound
    if lower == -inf:
        lower = -1.004325326  # eek this is a hack to avoid hitting multiple roots
        while s.count_roots_between(lower, upper) < num_roots:
            lower *= 10

    # Find a finite upper bound
    if upper == inf:
        upper = 1.009548275  # eek this is a hack to avoid hitting multiple roots
        while s.count_roots_between(lower, upper) < num_roots:
            upper *= 10

    # Now isolate the roots
    brackets = [lower]
    for i in range(num_roots-1):
        bracket = binary_search_continuous(lambda x: s.count_roots_between(lower,x), 1, lower, upper)
        brackets.append(bracket)
        lower = bracket
    brackets.append(upper)

    return brackets


def bisect_bracket(f, a, b, tol, threshold=1e-10):
    '''Given a function f with a root between A and B, return a new
    interval (C,D) containing the root such that D-C < TOL.'''
    assert a < b
    assert tol > 0

    ya = f(a)
    yb = f(b)
    assert ya != 0
    assert yb != 0
    assert (ya<0) != (yb<0), 'a=%f, b=%f, ya=%f, yb=%f' % (a,b,ya,yb)

    while b-a > tol:
        c = (a + b) / 2
        yc = f(c)
        if abs(yc) < threshold:
            # When yc falls below numerical precision we can no longer
            # keep reducing the search window
            return a,b
        if (yc<0) == (ya<0):
            a = c
            ya = yc
        else:
            b = c
            yb = yc
    return (a,b)


def polish_root(polynomial, root, **kwargs):
    import scipy.optimize
    assert polynomial.num_vars == 1
    first_derivative = polynomial.partial_derivative(0)
    second_derivative = first_derivative.partial_derivative(0)
    return scipy.optimize.newton(func=polynomial.compile(),
                                 fprime=first_derivative.compile(),
                                 fprime2=second_derivative.compile(),
                                 x0=root,
                                 **kwargs)


def solve_via_sturm(polynomial, lower=-inf, upper=inf, tol=1e-8):
    '''Find all roots of a univariate polynomial. Also return upper
    and lower bounds for each root.'''
    bounds = isolate_roots(polynomial, lower, upper)
    f = polynomial.compile()
    roots = []
    brackets = []
    for i in range(len(bounds)-1):
        a,b = bisect_bracket(f, bounds[i], bounds[i+1], tol)
        roots.append((a+b)/2)
        brackets.append((a,b))
    return roots, brackets


def companion_matrix(polynomial):
    '''Compute the companion matrix for a univariate polynomial.'''
    assert polynomial.num_vars == 1
    d = polynomial.total_degree
    lt = polynomial.leading_term()
    C = np.zeros((d, d))
    C[1:,:-1] = np.eye(d-1)
    for term in polynomial:
        if term.total_degree != d:
            C[term.total_degree,-1] = -float(term.coef / lt.coef)
    return C


def solve_via_companion(polynomial):
    # Compute the companion matrix
    C = companion_matrix(polynomial)
    # Find eigenvalues
    v = sorted(a.real for a in np.linalg.eigvals(C))
    # Construct the sturm chain to determine the number of unique roots
    n = SturmChain(polynomial).count_roots()
    # Collapse the closest pair of roots until we have exactly n
    while len(v) > n:
        i = np.argmin(np.diff(v))
        del v[i]
    return v

