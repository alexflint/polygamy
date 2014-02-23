from __future__ import division

import abc
import numbers
import fractions
import collections
import StringIO
import itertools
import ast
import operator
import types
import math

inf = float('inf')

def product(xs):
    '''Compute the product of the elements of XS.'''
    return reduce(operator.mul, xs)

def compare_leftmost(A, B):
    assert len(A) == len(B)
    for a,b in zip(A, B):
        if a > b:
            return 1
        elif a < b:
            return -1
    return 0

def compare_rightmost(A, B):
    assert len(A) == len(B)
    for a,b in zip(A[::-1], B[::-1]):
        if a > b:
            return 1
        elif a < b:
            return -1
    return 0

def can_divide_monomial(A, B):
    '''True if B divides A.'''
    assert len(A) == len(B)
    for i in range(len(A)):
        if B[i] > A[i]:
            return False
    return True

def divide_monomial(A, B):
    '''Divide the monomial A by the monomial B.'''
    return tuple(A[i] - B[i] for i in range(len(A)))

def multiply_monomial(A, B):
    '''Multiply the monomial A by the monomial B.'''
    return tuple(A[i] + B[i] for i in range(len(A)))

def superscript_string(v):
    '''Construct a unicode string containing v as superscript characters.'''
    assert isinstance(v, numbers.Integral)
    chars = []
    if v < 0:
        chars.append(unichr(0x207B))
        v = -v
    for digit in map(int, str(v)):
        if digit == 1:
            chars.append(unichr(0x00B9))
        elif digit == 2:
            chars.append(unichr(0x00B2))
        elif digit == 3:
            chars.append(unichr(0x00B3))
        else:
            chars.append(unichr(0x2070 + digit))
    return ''.join(chars)

def subscript_string(v):
    '''Construct a unicode string containing v as superscript characters.'''
    assert isinstance(v, numbers.Integral)
    chars = []
    if v < 0:
        chars.append(unichr(0x208B))
        v = -v
    for digit in map(int, str(v)):
        chars.append(unichr(0x2080 + digit))
    return ''.join(chars)


def as_polynomial(x, num_vars):
    '''Convert scalars, terms, or monomials to polynomials.'''
    if isinstance(x, numbers.Real):
        # Interpret scalars as constant polynomials
        return Polynomial.create([Term(x, (0,)*num_vars)])
    elif isinstance(x, tuple):
        # Interpret tuples as monomials
        return Polynomial.create([Term(1, x)])
    elif isinstance(x, Term):
        # Interpret terms as length-1 polynomials
        return Polynomial.create([x])
    elif isinstance(x, Polynomial):
        return x
    else:
        raise TypeError('Cannot convert %s to polynomial' % type(x))

class DivisionError(Exception):
    pass

class TupleOrdering(object):
    '''Represents an ordering over n-tuples of integers.'''
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def __call__(self, a, b):
        "__Call__ two tuples and return -1, 0, or 1"
        pass

class LexOrdering(TupleOrdering):
    '''Implements "lex" monomial ordering.'''
    def __call__(self, a, b):
        return compare_leftmost(a, b)

class GrlexOrdering(TupleOrdering):
    '''Implements "grlex" monomial ordering.'''
    def __call__(self, a, b):
        if sum(a) > sum(b):
            return 1
        elif sum(a) < sum(b):
            return -1
        else:
            return compare_leftmost(a, b)

class GrevlexOrdering(TupleOrdering):
    '''Implements "grevlex" monomial ordering.'''
    def __call__(self, a, b):
        if sum(a) > sum(b):
            return 1
        elif sum(a) < sum(b):
            return -1
        else:
            return compare_rightmost(b, a)  # yes this is (b,a) not (a,b)

class MonomialOrdering(object):
    def __init__(self, variable_order, tuple_order):
        self._vars = { x:i for i,x in enumerate(variable_ordering) }
        self._ordering = tuple_ordering
    def __call__(self, a, b):
        # TODO: implement more ways to parse monomials
        assert len(a) == len(self._vars)
        assert len(b) == len(self._vars)
        return self._ordering(a, b)

class Term(object):
    def __init__(self, coef, monomial):
        self.monomial = monomial
        if isinstance(coef, numbers.Integral):
            self.coef = fractions.Fraction(coef)
        elif isinstance(coef, numbers.Real):
            self.coef = coef
        else:
            raise ValueError('Invalid coefficient: %s (type=%s)' % (coef, type(coef)))

    def __eq__(self, rhs):
        if not isinstance(rhs, Term):
            return False
        return self.coef == rhs.coef and self.monomial == rhs.monomial

    def _multiply_by(self, rhs):
        self.coef *= rhs.coef
        self.monomial = multiply_monomial(self.monomial, rhs.monomial)

    def _divide_by(self, rhs):
        if not rhs.divides(self):
            raise DivisionError('Cannot divide %s by %s' % (term,rhs))
        self.coef /= rhs.coef
        self.monomial = divide_monomial(self.monomial, rhs.monomial)

    def __mul__(self, rhs):
        result = self.copy()
        result._multiply_by(rhs)
        return result

    def __truediv__(self, rhs):
        result = self.copy()
        result._divide_by(rhs)
        return result

    def __neg__(self):
        return Term(-self.coef, self.monomial)

    def __call__(self, *x):
        '''Evaluate this term at x.'''
        assert len(x) == len(self.monomial)
        return float(self.coef) * product(xi**ai for xi,ai in zip(x,self.monomial))

    def divides(self, rhs):
        return can_divide_monomial(rhs.monomial, self.monomial)

    @property
    def total_degree(self):
        '''Return the sum of the exponents in this term.'''
        return sum(self.monomial)

    def rationalize(self):
        '''Return a copy of this term in which the coefficient is a Fraction object.'''
        return Term(fractions.Fraction(self.coef), self.monomial)

    def copy(self):
        '''Return a copy of this term.'''
        return Term(self.coef, self.monomial)

    def python_expression(self, varnames):
        '''Construct a python expression string representing this polynomial.'''
        assert len(varnames) == len(self.monomial)
        return '*'.join([str(self.coef)] + ['%s**%d' % (var,exponent)
                                            for var,exponent in zip(varnames,self.monomial)
                                            if exponent>0])

    def format(self, use_superscripts=True):
        '''Construct a string representation of this polynomial.'''
        strings = []
        if self.coef != 1 or self.total_degree == 0:
            strings.append(str(self.coef))
        for var_index,exponent in enumerate(self.monomial):
            if exponent >= 1:
                if len(self.monomial) <= 4:
                    var = 'xyzw'[var_index]
                elif use_superscripts:
                    var = 'x'+subscript_string(var_index+1)
                else:
                    var = 'x'+str(var_index+1)

                if exponent == 1:
                    strings.append(var)
                elif use_superscripts:
                    strings.append(var + superscript_string(exponent))
                else:
                    strings.append(var + '^' + str(exponent))

        if use_superscripts:
            return ''.join(strings)
        else:
            return '*'.join(strings)

    def __unicode__(self):
        return self.format()

    def __str__(self):
        return unicode(self).encode('utf-8')

class ComparableTerm(object):
    @classmethod
    def factory(cls, ordering):
        def build(term):
            return ComparableTerm(ordering, term)
        return build
    def __init__(self, ordering, term):
        self._ordering = ordering
        self._term = term
    def __cmp__(self, rhs):
        return self._ordering(self._term.monomial, rhs._term.monomial)

class Polynomial(object):
    def __init__(self, num_vars):
        self._num_vars = num_vars
        self._term_dict = {}

    @classmethod
    def create(cls, terms=[], num_vars=None):
        if num_vars is None:
            assert len(terms) > 0, \
                'for polynomials with zero terms you must pass num_vars'
            num_vars = len(terms[0].monomial)

        # there may be duplicate terms so add them one by one
        p = Polynomial(num_vars)
        for term in terms:
            p._add_term(term)
        return p

    @classmethod
    def zero(cls, num_vars):
        return Polynomial(num_vars)

    @classmethod
    def one(cls, num_vars):
        return Polynomial.create([Term(1,(0,)*num_vars)])

    def copy(self):
        p = Polynomial(self._num_vars)
        p._term_dict = { monomial: term.copy() for monomial,term in self._term_dict.iteritems() }
        return p

    @property
    def num_vars(self):
        return self._num_vars

    @property
    def terms(self):
        return self._term_dict.viewvalues()

    @property
    def total_degree(self):
        if len(self) == 0:
            return 0
        else:
            return max(term.total_degree for term in self.terms)

    def sorted_terms(self, ordering, reverse=False):
        return sorted(self.terms, key=ComparableTerm.factory(ordering), reverse=reverse)

    def leading_term(self, ordering):
        return max(self.terms, key=ComparableTerm.factory(ordering))

    def trailing_terms(self, ordering):
        return Polynomial.create(self.sorted_terms(ordering)[:-1], self._num_vars)

    def divides(self, rhs, ordering):
        return any([ self.leading_term(ordering).divides(term) for term in rhs.terms ])

    def can_divide_by(self, rhs, ordering):
        return rhs.divides(self)

    def divide_by(self, rhs, ordering):
        rhs = as_polynomial(rhs, self._num_vars)
        lt_rhs = rhs.leading_term(ordering)
        tt_rhs = rhs.trailing_terms(ordering)

        dividend = self.copy()
        remainder = Polynomial.zero(self._num_vars)
        quotient = Polynomial.zero(self._num_vars)

        while len(dividend) > 0:
            lt_dividend = dividend._pop_leading_term(ordering)
            if lt_rhs.divides(lt_dividend):
                factor = lt_dividend / lt_rhs
                quotient._add_term(factor)
                dividend -= tt_rhs * factor
            else:
                remainder._add_term(lt_dividend)

        return quotient,remainder

    def partial_derivative(self, var_index):
        '''Return a polynomial representing the partial derivative of
        this polynomial with respect to the its i-th variable.'''
        assert var_index >= 0
        assert var_index < self._num_vars

        result = Polynomial.zero(self._num_vars)
        for term in self.terms:
            if term.monomial[var_index] > 0:
                derivative_coef = term.coef * term.monomial[var_index]
                derivative_monomial = tuple(exponent - int(i==var_index) 
                                            for i,exponent in enumerate(term.monomial))
                result._add_term(Term(derivative_coef, derivative_monomial))
        return result

    def squeeze(self):
        '''Return a new polynomial with a (possibly) smaller number of
        variables formed by eliminating variables that do not appear
        in any term.'''
        mask = [ any(term.monomial[i]>0 for term in self.terms)
                 for i in range(self._num_vars) ]
        result = Polynomial.zero(sum(mask))
        for term in self.terms:
            result._add_term(Term(term.coef, tuple(v for i,v in enumerate(term.monomial) if mask[i])))
        return result

    def rationalize(self):
        '''Return a copy of this polynomial in which all coefficients
        are Fraction objects.'''
        return Polynomial.create([term.rationalize() for term in self.terms])

    def _pop_leading_term(self, ordering):
        return self._term_dict.pop(self.leading_term(ordering).monomial)

    def _add_term(self, term):
        assert isinstance(term, Term)
        assert len(term.monomial) == self._num_vars
        t = self._term_dict.setdefault(term.monomial, Term(0, term.monomial))
        t.coef += term.coef
        if t.coef == 0:
            del self._term_dict[term.monomial]

    def _add_terms(self, terms):
        for term in terms:
            self._add_term(term)

    def _negate_terms(self):
        for term in self._term_dict.itervalues():
            term.coef = -term.coef

    def _divide_terms_by(self, rhs):
        for term in self._term_dict.itervalues():
            term._divide_by(rhs)

    def __eq__(self, rhs):
        rhs = as_polynomial(rhs, self._num_vars)
        # dictionaries conveniently do an automatic deep comparison
        # including checking for missing elements
        return rhs._term_dict == self._term_dict

    def __len__(self):
        return len(self._term_dict)

    def __add__(self, rhs):
        rhs = as_polynomial(rhs, self._num_vars)
        result = self.copy()
        result._add_terms(rhs.terms)
        return result

    def __neg__(self):
        result = self.copy()
        result._negate_terms()
        return result

    def __sub__(self, rhs):
        rhs = as_polynomial(rhs, self._num_vars)
        result = rhs.copy()
        result._negate_terms()
        result._add_terms(self.terms)
        return result

    def __mul__(self, rhs):
        rhs = as_polynomial(rhs, num_vars=self._num_vars)
        result = Polynomial.zero(self._num_vars)
        for lterm,rterm in itertools.product(self.terms, rhs.terms):
            result._add_term(lterm*rterm)
        return result

    def __pow__(self, rhs):
        if not isinstance(rhs, numbers.Integral):
            raise TypeError('Cannot raise a polynomial to the power of a %s' % type(rhs))
        elif rhs < 0:
            raise TypeError('Cannot raise a polynomial to a negative power.')

        result = Polynomial.zero(self._num_vars)
        for terms in itertools.product(self.terms, repeat=rhs):
            result._add_term(product(terms))
        return result

    def __truediv__(self, rhs):
        # We only support division by a scalar. To perform polynomial
        # division, use __mod__ to compute the remainder, or
        # __floordiv__ to compute the quotient, or divide_by() to
        # compute both
        if isinstance(rhs, numbers.Rational):
            return self * fractions.Fraction(1, rhs)

    def __floordiv__(self, rhs):
        quotient,remainder = self.divide_by(rhs, GrevlexOrdering())
        return quotient

    def __mod__(self, rhs):
        quotient,remainder = self.divide_by(rhs, GrevlexOrdering())
        return remainder

    def __rmul__(self, lhs):
        try:
            return as_polynomial(lhs, self._num_vars) * self
        except TypeError:
            return NotImplemented

    def __radd__(self, lhs):
        try:
            return as_polynomial(lhs, self._num_vars) + self
        except TypeError:
            return NotImplemented

    def __rsub__(self, lhs):
        try:
            return as_polynomial(lhs, self._num_vars) - self
        except TypeError:
            return NotImplemented

    def __rmod__(self, lhs):
        try:
            return as_polynomial(lhs, self._num_vars) % self
        except TypeError:
            return NotImplemented

    def __rfloordiv__(self, lhs):
        try:
            return as_polynomial(lhs, self._num_vars) // self
        except TypeError:
            return NotImplemented

    def __call__(self, *x):
        '''Evaluate this polynomial at x, which should be an iterable
        of length self._num_vars.'''
        assert len(x) == self._num_vars
        return sum(term(*x) for term in self.terms)

    def evaluate_at_infinity(self):
        '''Compute the limiting value of this polynomial as x tends to infinity.'''
        assert self._num_vars == 1
        if len(self) == 0:
            return 0
        else:
            return self.leading_term(LexOrdering())(inf)

    def evaluate_at_minus_infinity(self):
        '''Compute the limiting value of this polynomial as x tends to minus infinity.'''
        assert self._num_vars == 1
        if len(self) == 0:
            return 0
        else:
            return self.leading_term(LexOrdering())(-inf)

    def compile(self):
        '''Return a python function that can be used to evaluate this polynomial very quickly.'''
        varnames = tuple('x'+str(i) for i in range(self._num_vars))
        expr = self.python_expression(varnames)
        source = 'def f(%s): return %s' % (','.join(varnames), expr)
        code = compile(source, '<polynomial>', mode='exec')
        namespace = {}
        exec code in namespace
        return namespace['f']
        
    def python_expression(self, varnames=None):
        '''Construct a representation of this polynomial as a python expression string.'''
        if varnames is None:
            varnames = tuple('x'+str(i) for i in range(self._num_vars))
        if len(self) == 0:
            return '0'
        else:
            return ' + '.join(term.python_expression(varnames) for term in self.terms)

    def format(self, ordering=GrevlexOrdering(), use_superscripts=True):
        '''Construct a string representation of this polynomial.'''
        if len(self) == 0:
            return '0'
        else:
            parts = []
            for term in self.sorted_terms(ordering, reverse=True):
                if term.coef < 0:
                    if len(parts) == 0:
                        parts.append('-')
                    else:
                        parts.append(' - ')
                    term = -term
                else:
                    if len(parts) != 0:
                        parts.append(' + ')
                parts.append(term.format(use_superscripts))
            return ''.join(parts)

    def __unicode__(self):
        return self.format()

    def __str__(self):
        return unicode(self).encode('utf8')

    def __repr__(self):
        return str(self)

def gcd(A, B):
    assert len(A) == len(B)
    return tuple(min(A[i],B[i]) for i in range(len(A)))

def lcm(A, B):
    assert len(A) == len(B)
    return tuple(max(A[i],B[i]) for i in range(len(A)))

def remainder(f, H, ordering):
    '''Compute the remainder of f on division by <H1,...,Hn> (the ideal generated by H).'''
    quotients = [ Polynomial.zero(h.num_vars) for h in H ]
    remainder = f.copy()
    i = 0
    while i < len(H):
        if H[i].divides(remainder, ordering):
            quotient,remainder = remainder.divide_by(H[i], ordering)
            quotients[i] += quotient
            i = 0
        else:
            i += 1

    return remainder

def s_poly(f, g, ordering):
    f = as_polynomial(f)
    g = as_polynomial(g)
    ltf = f.leading_term(ordering)
    ltg = g.leading_term(ordering)
    common = Term(1, lcm(ltf.monomial, ltg.monomial))
    return as_polynomial(common/ltf) * f + as_polynomial(common/ltg) * g

def gbasis(F):
    pass

def show_division(a, b):
    if isinstance(a, basestring):
        a,b = parse(a,b)
    q,r = a.divide_by(b)
    print '[%s] = (%s) * [%s] + (%s)' % (a, q, b, r)

def extract_symbols(module):
    return { node.id for node in ast.walk(module) if isinstance(node, ast.Name) }

def polynomial_jacobian(polynomial):
    partials = [ polynomial.partial_derivative(i) for i in range(polynomial.num_vars) ]
    return lambda x: [ partial(x) for partial in partials ]

def polynomial_hessian(polynomial):
    partials = [ polynomial.partial_derivative(i) for i in range(polynomial.num_vars) ]
    return lambda x: [ partial(x) for partial in partials ]

def polish_univariate_root(polynomial, root, **kwargs):
    import scipy.optimize
    assert polynomial.num_vars == 1
    first_derivative = polynomial.partial_derivative(0)
    second_derivative = first_derivative.partial_derivative(0)
    return scipy.optimize.newton(func=polynomial.compile(),
                                 fprime=first_derivative.compile(),
                                 fprime2=second_derivative.compile(),
                                 x0=root,
                                 **kwargs)

def count_sign_changes(ys):
    signs = [ cmp(y,0) for y in ys if y != 0 ]  # ignore all zero evaluations
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
            return count_sign_changes(f.evaluate_at_infinity() for f in self._chain)
        elif x == -inf:
            return count_sign_changes(f.evaluate_at_minus_infinity() for f in self._chain)
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

def bracket_univariate_roots(polynomial, lower=-inf, upper=inf):
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

def bisect_bracket(f, a, b, tol):
    '''Given a function f with a root between A and B, return a new
    interval (C,D) containing the root such that D-C < TOL.'''
    assert a < b
    assert tol > 0

    ya = f(a)
    yb = f(b)
    assert ya != 0
    assert yb != 0
    assert (ya<0) != (yb<0)

    while b-a > tol:
        c = (a + b) / 2
        yc = f(c)
        if (yc<0) == (ya<0):
            a = c
            ya = yc
        else:
            b = c
            yb = yc
    return (a,b)

def parse(*exprs, **kwargs):
    # Get symbols
    symbols = set.union(*[extract_symbols(ast.parse(expr.strip())) for expr in exprs])

    # Check variable order
    variable_order = kwargs.get('variable_order', None)
    if variable_order is None:
        variable_order = sorted(symbols)
    else:
        assert symbols.issubset(variable_order), 'variable_order contained the wrong symbols'

    # Construct polynomials corresponding to each variable
    n = len(variable_order)
    variables = { var : as_polynomial(tuple(int(x==i) for x in range(n)), n)
                  for i,var in enumerate(variable_order) }

    # Evaluate
    polynomials = tuple(eval(expr, variables) for expr in exprs)

    # Cast to singleton if necessary
    if len(exprs) == 1:
        return polynomials[0]
    else:
        return polynomials
