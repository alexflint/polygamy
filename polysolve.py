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
import numpy as np

import ring
import unicode_rendering

inf = float('inf')

class OrderingError(Exception):
    pass

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

def as_polynomial(x, num_vars, ctype=None):
    '''Convert scalars, terms, or monomials to polynomials.'''
    if isinstance(x, numbers.Real):
        # Interpret scalars as constant polynomials
        return Polynomial.create([Term(x, (0,)*num_vars)], num_vars, ctype)
    elif isinstance(x, tuple):
        # Interpret tuples as monomials
        return Polynomial.create([Term(1, x)], num_vars, ctype)
    elif isinstance(x, Term):
        # Interpret terms as length-1 polynomials
        return Polynomial.create([x], num_vars, ctype)
    elif isinstance(x, Polynomial):
        return x
    else:
        raise TypeError('Cannot convert %s to polynomial' % type(x))

def as_term(x, num_vars):
    '''Convert scalars, terms, or monomials to polynomials.'''
    if isinstance(x, numbers.Real):
        # Interpret scalars as constant polynomials
        return Term(x, (0,)*num_vars)
    elif isinstance(x, tuple):
        # Interpret tuples as monomials
        return Term(1, x)
    elif isinstance(x, Term):
        # Interpret terms as length-1 polynomials
        return x
    else:
        raise TypeError('Cannot convert %s to polynomial' % type(x))

class DivisionError(Exception):
    pass

class MonomialOrdering(object):
    '''Represents an ordering over n-tuples of integers.'''
    __metaclass__ = abc.ABCMeta
    @abc.abstractmethod
    def __call__(self, a, b):
        "__Call__ two tuples and return -1, 0, or 1"
        pass

class LexOrdering(MonomialOrdering):
    '''Implements "lex" monomial ordering.'''
    def __call__(self, a, b):
        return compare_leftmost(a, b)

class GrlexOrdering(MonomialOrdering):
    '''Implements "grlex" monomial ordering.'''
    def __call__(self, a, b):
        if sum(a) > sum(b):
            return 1
        elif sum(a) < sum(b):
            return -1
        else:
            return compare_leftmost(a, b)

class GrevlexOrdering(MonomialOrdering):
    '''Implements "grevlex" monomial ordering.'''
    def __call__(self, a, b):
        if sum(a) > sum(b):
            return 1
        elif sum(a) < sum(b):
            return -1
        else:
            return compare_rightmost(b, a)  # yes this is (b,a) not (a,b)

class DegreeOrdering(MonomialOrdering):
    '''Orders univariate monomials by their degree. This is not a true
    monomial ordering because it is only valid for univariate
    monomials.'''
    def __call__(self, a, b):
        assert len(a) == 1
        assert len(b) == 1
        return cmp(a[0], b[0])

class Term(object):
    def __init__(self, coef, monomial, ctype=None):
        self._monomial = monomial
        if ctype is None:
            self._coef = coef
            self._ctype = type(coef)
        else:
            self._coef = ctype(coef)
            self._ctype = ctype

    @property
    def coef(self):
        return self._coef

    @coef.setter
    def coef(self, value):
        self._coef = self._ctype(value)

    @property
    def monomial(self):
        return self._monomial

    @property
    def ctype(self):
        return self._ctype

    def astype(self, ctype):
        '''If ctype == self.ctype then return a reference to this
        object, otherwise return a copy of this term converted to the
        given type.'''
        if ctype == self.ctype:
            return self
        else:
            return Term(self.coef, self.monomial, ctype)

    def __eq__(self, rhs):
        rhs = as_term(rhs, len(self.monomial))
        return self.coef == rhs.coef and self.monomial == rhs.monomial

    def __ne__(self, rhs):
        return not (self == rhs)

    def _multiply_by(self, rhs):
        if isinstance(rhs, numbers.Real):
            self.coef *= rhs
        else:
            self.coef *= rhs.coef
            self._monomial = multiply_monomial(self.monomial, rhs.monomial)

    def _divide_by(self, rhs):
        if isinstance(rhs, numbers.Real):
            self.coef /= rhs
        else:
            if not rhs.divides(self):
                raise DivisionError('Cannot divide %s by %s' % (term,rhs))
            self.coef /= rhs.coef
            self._monomial = divide_monomial(self.monomial, rhs.monomial)

    def _negate(self):
        self.coef = -self.coef

    def __mul__(self, rhs):
        rhs = as_term(rhs, len(self.monomial))
        result = self.copy()
        result._multiply_by(rhs)
        return result

    def __truediv__(self, rhs):
        rhs = as_term(rhs, len(self.monomial))
        result = self.copy()
        result._divide_by(rhs)
        return result

    def __neg__(self):
        return Term(-self.coef, self.monomial, self.ctype)

    def __call__(self, *x):
        '''Evaluate this term at x.'''
        assert len(x) == len(self.monomial)
        return self.coef * product(xi**ai for xi,ai in zip(x,self.monomial))

    def evaluate_partial(self, var_index, value):
        '''Create a new term with by evaluating this term at the given
        value for the given variable. The result formally contains the
        same number of variables but the evaluated variable always has
        an exponent of zero.'''
        return Term(self.coef * value**self.monomial[var_index],
                    tuple(0 if i==var_index else a for i,a in enumerate(self.monomial)),
                    self.ctype)

    def divides(self, rhs):
        return can_divide_monomial(rhs.monomial, self.monomial)

    @property
    def total_degree(self):
        '''Return the sum of the exponents in this term.'''
        return sum(self.monomial)

    def copy(self):
        '''Return a copy of this term.'''
        return Term(self.coef, self.monomial, self.ctype)

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
                    var = 'x'+unicode_rendering.subscript(var_index+1)
                else:
                    var = 'x'+str(var_index+1)

                if exponent == 1:
                    strings.append(var)
                elif use_superscripts:
                    strings.append(var + unicode_rendering.superscript(exponent))
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

    def __repr__(self):
        return str(self)

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
    def __init__(self, num_vars, ctype=None):
        self._num_vars = num_vars
        self._ctype = ctype or fractions.Fraction
        self._term_dict = {}

    @classmethod
    def create(cls, terms=[], num_vars=None, ctype=None):
        if num_vars is None:
            terms = list(terms)
            assert len(terms) > 0, 'to create an empty polynomial you must pass num_vars'
            num_vars = len(terms[0].monomial)

        if ctype is None:
            terms = list(terms)
            if len(terms) > 0:
                ctype = terms[0].ctype

        # there may be duplicate terms so add them one by one
        p = Polynomial(num_vars, ctype)
        p._add_terms(terms)
        return p

    @property
    def num_vars(self):
        '''Return the number of variables in the polynomial ring in
        which this polynomial resides.'''
        return self._num_vars

    @property
    def ctype(self):
        return self._ctype

    @property
    def total_degree(self):
        '''Return the sum of the exponents of the highest-degree term in this polynomial.'''
        if len(self) == 0:
            return 0
        else:
            return max(term.total_degree for term in self)

    def copy(self):
        '''Return a copy of this polynomial.'''
        return Polynomial.create((term.copy() for term in self),
                                 self.num_vars,
                                 self.ctype)

    def astype(self, ctype):
        '''Return a copy of this polynomial in which each coefficient
        is cast to the given type.'''
        if ctype == self.ctype:
            return self
        else:
            return Polynomial.create((term.astype(ctype) for term in self),
                                     self.num_vars,
                                     ctype)

    def sorted_terms(self, ordering=None, reverse=False):
        '''Return a collection of Term objects representing terms in
        this polynomial, sorted by the given ordering (lowest ordered
        term first).'''
        return sorted(self,
                      key=ComparableTerm.factory(self._resolve_ordering(ordering)),
                      reverse=reverse)

    def leading_term(self, ordering=None):
        '''Return a Term object representing the term in this
        polynomial that is sorted first by the given ordering.'''
        return max(self,
                   key=ComparableTerm.factory(self._resolve_ordering(ordering)))

    def trailing_terms(self, ordering=None):
        '''Return a polynomial consisting of all terms in this
        polynomial other than the leading term.'''
        return Polynomial.create(self.sorted_terms(ordering)[:-1], self.num_vars, self.ctype)

    def divides(self, rhs, ordering=None):
        return any([ self.leading_term(ordering).divides(term) for term in rhs ])

    def can_divide_by(self, rhs, ordering=None):
        return rhs.divides(self)

    def divide_by(self, rhs, ordering=None):
        rhs = as_polynomial(rhs, self.num_vars)
        if rhs == 0:
            raise DivisionError('Cannot divide by zero')

        lt_rhs = rhs.leading_term(ordering)
        tt_rhs = rhs.trailing_terms(ordering)

        dividend = self.copy()
        remainder = Polynomial(self.num_vars, self.ctype)
        quotient = Polynomial(self.num_vars, self.ctype)

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
        assert var_index < self.num_vars

        result = Polynomial(self.num_vars, self.ctype)
        for term in self:
            if term.monomial[var_index] > 0:
                derivative_coef = term.coef * term.monomial[var_index]
                derivative_monomial = tuple(exponent - int(i==var_index) 
                                            for i,exponent in enumerate(term.monomial))
                result[derivative_monomial] += derivative_coef
        return result

    def squeeze(self):
        '''Return a new polynomial with a (possibly) smaller number of
        variables formed by eliminating variables that do not appear
        in any term.'''
        mask = [ any(term.monomial[i]>0 for term in self)
                 for i in range(self.num_vars) ]
        result = Polynomial(sum(mask), self.ctype)
        for term in self:
            squeezed_monomial = tuple(v for i,v in enumerate(term.monomial) if mask[i])
            result[squeezed_monomial] += term.coef
        return result

    def normalized(self, ordering=None):
        '''Return a copy of this polynomial in which the leading coefficient is 1.'''
        if len(self) == 0:
            return Polynomial(self.num_vars, self.ctype)  # return a copy, not a reference
        else:
            lt = self.leading_term(ordering)
            result = Polynomial(self.num_vars, self.ctype)
            result[lt.monomial] = 1
            result._add_terms(term/lt.coef for term in self if term is not lt)
            return result

    def _resolve_ordering(self, ordering=None):
        '''If ordering is None and this is a univariate polynomial
        then return a DegreeOrdering instance. Otherwise, check that
        it is a valid monomial ordering for this polynomial and return
        it if so, or raise an exception if not.'''
        if ordering is None:
            if self.num_vars == 1:
                return DegreeOrdering()
            else:
                raise OrderingError('you must provide a monomial ordering because this '+
                                    'polynomial is over more than one variable')
        else:
            if callable(ordering):
                return ordering
            else:
                raise OrderingError('monomial orderings must be callable')

    def _pop_leading_term(self, ordering=None):
        return self._term_dict.pop(self.leading_term(ordering).monomial)

    def _add_term(self, term):
        self[term.monomial] += term.coef

    def _add_terms(self, terms):
        for term in terms:
            self._add_term(term)

    def _negate_terms(self):
        for term in self:
            term._negate()

    def _divide_terms_by(self, rhs):
        for term in self:
            term._divide_by(rhs)

    def __eq__(self, rhs):
        rhs = as_polynomial(rhs, self.num_vars)
        # dictionaries conveniently do an automatic deep comparison
        # including checking for missing elements
        return rhs._term_dict == self._term_dict

    def __ne__(self, rhs):
        return not (self == rhs)

    def __nonzero__(self):
        return len(self) > 0

    def __add__(self, rhs):
        rhs = as_polynomial(rhs, self.num_vars)
        result = self.copy()
        result._add_terms(rhs)
        return result

    def __neg__(self):
        result = self.copy()
        result._negate_terms()
        return result

    def __sub__(self, rhs):
        rhs = as_polynomial(rhs, self.num_vars)
        result = rhs.copy()
        result._negate_terms()
        result._add_terms(self)
        return result

    def __mul__(self, rhs):
        rhs = as_polynomial(rhs, self.num_vars)
        result = Polynomial(self.num_vars, self.ctype)
        for lterm,rterm in itertools.product(self, rhs):
            result._add_term(lterm*rterm)
        return result

    def __pow__(self, rhs):
        if not isinstance(rhs, numbers.Integral):
            raise TypeError('cannot raise a polynomial to the power of a %s' % type(rhs))
        elif rhs < 0:
            raise TypeError('cannot raise a polynomial to a negative power.')

        result = Polynomial(self.num_vars, self.ctype)
        for terms in itertools.product(self, repeat=rhs):
            result._add_term(product(terms))
        return result

    def __truediv__(self, rhs):
        '''We only support division by a scalar. To perform polynomial
        division, use f%g to compute the remainder, f//g to compute
        the quotient, or divide_by() to compute both.'''
        assert isinstance(rhs, numbers.Rational), 'must use f%g or f//g for non-scalar division'
        return self * fractions.Fraction(1, rhs)

    def __floordiv__(self, rhs):
        # TODO: avoid putting a default in here - an OrderedPolynomial class perhaps?
        quotient,remainder = self.divide_by(rhs, GrevlexOrdering())
        return quotient

    def __mod__(self, rhs):
        # TODO: avoid putting a default in here - an OrderedPolynomial class perhaps?
        quotient,remainder = self.divide_by(rhs, GrevlexOrdering())
        return remainder

    def __rmul__(self, lhs):
        try:
            return as_polynomial(lhs, self.num_vars) * self
        except TypeError:
            return NotImplemented

    def __radd__(self, lhs):
        try:
            return as_polynomial(lhs, self.num_vars) + self
        except TypeError:
            return NotImplemented

    def __rsub__(self, lhs):
        try:
            return as_polynomial(lhs, self.num_vars) - self
        except TypeError:
            return NotImplemented

    def __rmod__(self, lhs):
        try:
            return as_polynomial(lhs, self.num_vars) % self
        except TypeError:
            return NotImplemented

    def __rfloordiv__(self, lhs):
        try:
            return as_polynomial(lhs, self.num_vars) // self
        except TypeError:
            return NotImplemented

    def __call__(self, *x):
        '''Return this polynomial evaluated at x, which should be an
        iterable of length num_vars.'''
        assert len(x) == self.num_vars
        return sum(term(*x) for term in self)

    def __len__(self):
        '''Return the number of terms in this polynomial.'''
        return len(self._term_dict)

    def __iter__(self):
        '''Return an iterator over the terms in this polynomial, in an
        arbitrary order. For predictable ordering, use
        polynomial.sorted_terms(...).'''
        return self._term_dict.itervalues()

    def __getitem__(self, monomial):
        '''Get the coefficient of the given monomial in this
        polynomial, or zero if this polynomial does not contain the
        given monomial.'''
        assert len(monomial) == self.num_vars
        term = self._term_dict.get(monomial, None)
        if term is None:
            return 0
        else:
            return term.coef

    def __setitem__(self, monomial, coef):
        '''Get the coefficient of the given monomial in this
        polynomial, or zero if this polynomial does not contain the
        given monomial.'''
        assert len(monomial) == self.num_vars
        term = self._term_dict.get(monomial, None)
        if term is None:
            term = Term(coef, monomial, self.ctype)
            self._term_dict[monomial] = term
        else:
            term.coef = coef
        if not term.coef:
            del self[monomial]

    def __delitem__(self, monomial):
        del self._term_dict[monomial]

    def __contains__(self, monomial):
        '''Return true if this polynomial contains a non-zero term
        with the given monomial.'''
        return monomial in self._term_dict

    def evaluate_partial(self, var, value):
        '''Evaluate this polynomial given a variable index and a value
        for that variable.  The result of this operation is always a
        new polynomial in the same number of variables, although the
        evaluated variable will not appear in any term.'''
        return Polynomial.create((term.evaluate_partial(var,value) for term in self),
                                 self.num_vars,
                                 self.ctype)

    def sign_at_infinity(self):
        '''Compute the limiting value of this polynomial as x tends to
        infinity.'''
        assert self.num_vars == 1
        if len(self) == 0:
            return 0
        else:
            return cmp(self.leading_term().coef, 0)

    def sign_at_minus_infinity(self):
        '''Compute the limiting value of this polynomial as x tends to
        minus infinity.'''
        assert self.num_vars == 1
        if len(self) == 0:
            return 0
        else:
            lt = self.leading_term()
            return cmp(lt.coef, 0) * (-1 if lt.monomial[0]%2 else 1)

    def compile(self):
        '''Return a python function that can be used to evaluate this
        polynomial quickly.'''
        varnames = tuple('x'+str(i) for i in range(self.num_vars))
        expr = self.python_expression(varnames)
        source = 'def f(%s): return %s' % (','.join(varnames), expr)
        code = compile(source, '<polynomial>', mode='exec')
        namespace = {}
        exec code in namespace
        return namespace['f']
        
    def python_expression(self, varnames=None):
        '''Construct a representation of this polynomial as a python
        expression string.'''
        if varnames is None:
            varnames = tuple('x'+str(i) for i in range(self.num_vars))
        if len(self) == 0:
            return '0'
        else:
            return ' + '.join(term.python_expression(varnames) for term in self)

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


#
# Utilities
#
def map_coefficients(f, polynomial):
    '''Return a new polynomial formed by replacing each coefficient in
    the given polynomial with f(coefficient).'''
    result = Polynomial.create(polynomial.num_vars, polynomial.ctype)
    for term in polynomial:
        result[term.monomial] = f(term.coef)
    return result

#
# Operations for systems of equations
#

def remainder(f, H, ordering=None):
    '''Compute the remainder of f on division by <H1,...,Hn> (the ideal generated by H).'''
    quotients = [ Polynomial(h.num_vars, h.ctype) for h in H ]
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

def matrix_form(F, ordering=None):
    '''Put the system of equations (f1=0,...,fn=0) into matrix form as
    C * X = 0, where C is a matrix of coefficients and X is a matrix
    of monomials.'''
    monomials = list(set(term.monomial for f in F for term in f))
    if isinstance(ordering, MonomialOrdering):
        monomials = sorted(monomials, key=lambda monomial: ComparableTerm(ordering, Term(1,monomial)))
    elif ordering is not None:
        monomials = ordering
    X = [ as_polynomial(monomial, F[0].num_vars) for monomial in monomials ]
    C = np.asarray([[ f[monomial] for monomial in monomials ] for f in F])
    return C,X





#
# Grobner basis computations
#

def lcm(A, B):
    assert len(A) == len(B)
    return tuple(max(A[i],B[i]) for i in range(len(A)))

def s_poly(f, g, ordering=None):
    f = as_polynomial(f)
    g = as_polynomial(g)
    ltf = f.leading_term(ordering)
    ltg = g.leading_term(ordering)
    common = Term(1, lcm(ltf.monomial, ltg.monomial))
    return as_polynomial(common/ltf) * f + as_polynomial(common/ltg) * g

def gbasis(F):
    pass




#
# Univariate solvers
#

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

def isolate_univariate_roots(polynomial, lower=-inf, upper=inf):
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

def polynomial_vector(polynomials):
    fs = [ p.compile() for p in polynomials ]
    return lambda *x: np.array([ f(*x) for f in fs ])

def polynomial_gradient(polynomial):
    return polynomial_vector(polynomial.partial_derivative(i) for i in range(polynomial.num_vars))

def polynomial_jacobian(polynomials):
    gradients = [ polynomial_gradient(p) for p in polynomials ]
    return lambda *x: np.array([ gradient(*x) for gradient in gradients ])

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

def polish_multivariate_root(polynomials, root, **kwargs):
    import scipy.optimize
    fun = polynomial_vector(polynomials)
    jac = polynomial_jacobian(polynomials)
    return scipy.optimize.root(fun=lambda x: fun(*x),
                               jac=lambda x: jac(*x),
                               x0=root,
                               **kwargs)

def solve_univariate_via_sturm(polynomial, lower=-inf, upper=inf, tol=1e-8):
    '''Find all roots of a univariate polynomial. Also return upper
    and lower bounds for each root.'''
    bounds = isolate_univariate_roots(polynomial, lower, upper)
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

def solve_univariate_via_companion(polynomial):
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
    


#
# Parsing polynomials from strings
#

def extract_symbols(module):
    return { node.id for node in ast.walk(module) if isinstance(node, ast.Name) }

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
    ctype = kwargs.get('ctype', None)
    variables = { }
    for i,var in enumerate(variable_order):
        monomial = tuple(int(x==i) for x in range(len(variable_order)))
        variables[var] = Polynomial.create([Term(1,monomial)], len(variable_order), ctype)

    # Evaluate
    polynomials = tuple(eval(expr, variables) for expr in exprs)

    # Cast to singleton if necessary
    if len(exprs) == 1:
        return polynomials[0]
    else:
        return polynomials
