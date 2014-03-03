from __future__ import division

import exceptions
import numbers

import ring

class ModularInverseError(exceptions.ArithmeticError):
    pass

class InverseOfZeroError(exceptions.ArithmeticError):
    pass

def multiplicative_inverse(r, n):
    '''Find the multiplicative inverse of r in the field of integers
    modulo n, ie an integer s such that s<n and s*r = 1 (mod n).'''
    if r == 0:
        raise InverseOfZeroError('cannot compute inverse of zero')
    gcd,a,b = ring.extended_gcd(r, n)
    if r*a % n != 1:
        raise ModularInverseError('%s has no inverse modulo %s' % (r,n))
    return a % n
    
class ModuloInteger(object):
    def __init__(self, r, n):
        '''Construct the integer r (mod n).'''
        assert isinstances(r, numbers.Integral)
        assert isinstances(n, numbers.Integral)
        self._r = r % n
        self._n = n

    @property
    def r(self):
        return self._r

    @property
    def n(self):
        return self._n

    @property
    def inverse(self):
        '''Return this object's multiplicative inverse, which is an
        integer s such that self*s = 1 (mod n).'''
        return multiplicative_inverse(self._r, self._n)

    def __add__(self, rhs):
        return ModuloInteger(self._r + int(rhs), self._n)
    def __radd__(self, lhs):
        return ModuloInteger(int(lhs) + self._r, self._n)

    def __sub__(self, rhs):
        return ModuloInteger(self._r - int(rhs), self._n)
    def __rsub__(self, lhs):
        return ModuloInteger(int(lhs) - self._r, self._n)

    def __neg__(self):
        return ModuloInteger(self._n - self._r, self._n)

    def __mul__(self, rhs):
        return ModuloInteger(self._r * int(rhs), self._n)
    def __rmul__(self, lhs):
        return ModuloInteger(int(lhs) * self._r, self._n)

    def __truediv__(self, rhs):
        return ModuloInteger(self._r * multiplicative_inverse(int(rhs), self._n), self._n)
    def __rtruediv__(self, lhs):
        return ModuloInteger(int(lhs) * multiplicative_inverse(self._r, self._n), self._n)

    def __div__(self, rhs):
        return ModuloInteger(self._r * multiplicative_inverse(int(rhs), self._n), self._n)
    def __rdiv__(self, lhs):
        return ModuloInteger(int(lhs) * multiplicative_inverse(self._r, self._n), self._n)

    def __floordiv__(self, rhs):
        return self / rhs
    def __rfloordiv__(self, lhs):
        return rhs / self

    def __int__(self):
        return int(self._r)
    def __long__(self):
        return long(self._r)

    def __cmp__(self, rhs):
        return cmp(self._r, int(rhs))
    def __nonzero__(self):
        return self._r != 0

    def __str__(self):
        return str(self._r)
    def __repr__(self):
        return '%s(%d,%d)' % (self.__class__.__name__, self._r, self._n)

numbers.Integral.register(ModuloInteger)
