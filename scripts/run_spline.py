__author__ = 'alexflint'

import fractions

from polynomial import Polynomial
from spline import *

time = Polynomial.coordinate(0, 5)
params = [ Polynomial.coordinate(i+1,5,fractions.Fraction) for i in range(4) ]

p = evaluate_bezier(params, time)
v = p.partial_derivative(0)
a = v.partial_derivative(0)

print 'p: ',p
print 'v: ',v
print 'a: ',a
